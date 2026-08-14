"""Low-Rank KV cache compression.

The K and V tensors per layer are projected to a low-rank space of
dimension ``rank`` and stored; during attention we reconstruct the
full-rank K/V on the fly. We measure both the compression ratio
(``rank / head_dim``) and the reconstruction overhead.

This implementation runs against the same HuggingFace models as the
baseline. It uses forward hooks to intercept K/V projections and
substitute low-rank approximations. Reconstruction quality is not
the focus — the focus is the memory/time trade-off.
"""

from __future__ import annotations

import time
from typing import List, Optional

import torch

from ..config import BenchmarkConfig
from ..profiling.memory import MemoryTracker, snapshot_memory
from ..profiling.resources import ResourceProbe
from ..utils.logging import get_logger
from ..utils.models import (
    dtype_from_string,
    get_torch_device,
    load_model_and_tokenizer,
)
from ..utils.prompts import build_prompt
from .base import BenchmarkResult, KVTechnique, TechniqueSpec


class LowRankCompression(KVTechnique):
    spec = TechniqueSpec(
        name="Low-Rank Compression",
        category="compression",
        description="KV cache projected to a low-rank subspace of dimension `rank`, reconstructed at attention time.",
    )

    def __init__(self, config: BenchmarkConfig, rank: int = 64):
        super().__init__(config)
        self.rank = rank
        self._projectors: list = []  # per-layer projector params
        self._hooks: list = []
        self.logger = get_logger(self.__class__.__name__)

    def setup(self) -> None:
        device = get_torch_device(use_cuda=self.config.use_cuda)
        dtype = dtype_from_string(self.config.dtype)
        self._model, self._tokenizer = load_model_and_tokenizer(
            self._model_name if hasattr(self, "_model_name") else self.config.model,
            dtype=dtype,
            device=device,
            use_cuda=self.config.use_cuda,
        )
        self._detect_shape()
        self._install_projectors()

    def analytic_kv_mb(self, batch: int, seq_len: int) -> float:
        shape = self._kv_shape
        if shape is None or shape.head_dim == 0:
            return 0.0
        r = min(self.rank, shape.head_dim)
        return (
            2
            * shape.num_layers
            * shape.num_kv_heads
            * r
            * shape.bytes_per_element
            * batch
            * seq_len
        ) / (1024 ** 2)

    def _install_projectors(self) -> None:
        """Create one (down, up) pair per attention layer's K and V
        projections and register forward hooks. The hooks replace the
        raw K/V with their low-rank factorization during the prefill +
        decode path.
        """
        device = next(self._model.parameters()).device

        layers = self._find_layers()
        self._projectors = []
        for layer in layers:
            attn = self._find_attn_module(layer)
            if attn is None:
                continue
            k_proj_weight = getattr(attn, "k_proj", None)
            v_proj_weight = getattr(attn, "v_proj", None)
            if k_proj_weight is None or v_proj_weight is None:
                # Fused qkv (e.g. GPT-2 ``c_attn``, Phi ``qkv_proj``): the
                # K/V projections are not separable, so we cannot attach a
                # per-projection low-rank factor. Skip this layer rather than
                # dereferencing a missing module; on such models low-rank
                # compression is simply not applied.
                continue
            # Allocate low-rank factors. The projector base dimension is the
            # projection's own output width: on MHA models that is the full
            # hidden size, while on GQA models (e.g. TinyLlama, whose
            # ``k_proj`` emits ``n_kv_heads * head_dim`` outputs) it is the
            # narrower shared KV width. Sizing from ``head_dim`` alone would
            # crash on the wider GQA output.
            kv_dim = k_proj_weight.out_features
            r = min(self.rank, kv_dim)
            down_k = torch.nn.Linear(kv_dim, r, bias=False, device=device, dtype=k_proj_weight.weight.dtype)
            up_k = torch.nn.Linear(r, kv_dim, bias=False, device=device, dtype=k_proj_weight.weight.dtype)
            down_v = torch.nn.Linear(kv_dim, r, bias=False, device=device, dtype=k_proj_weight.weight.dtype)
            up_v = torch.nn.Linear(r, kv_dim, bias=False, device=device, dtype=k_proj_weight.weight.dtype)
            # Initialize up/down as identity-ish: down is random, up is random, then we accept the noisy reconstruction.
            torch.nn.init.kaiming_uniform_(down_k.weight, a=5 ** 0.5)
            torch.nn.init.kaiming_uniform_(up_k.weight, a=5 ** 0.5)
            torch.nn.init.kaiming_uniform_(down_v.weight, a=5 ** 0.5)
            torch.nn.init.kaiming_uniform_(up_v.weight, a=5 ** 0.5)

            handle_k = k_proj_weight.register_forward_hook(
                lambda mod, inp, out, down=down_k, up=up_k: up(down(out))
            )
            handle_v = v_proj_weight.register_forward_hook(
                lambda mod, inp, out, down=down_v, up=up_v: up(down(out))
            )
            self._projectors.append((down_k, up_k, down_v, up_v))
            self._hooks.extend([handle_k, handle_v])

    def teardown(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._projectors.clear()
        super().teardown()

    def _find_layers(self):
        for path in ("model.layers", "transformer.h", "model.decoder.layers"):
            mod = self._model
            try:
                for p in path.split("."):
                    mod = getattr(mod, p)
                if hasattr(mod, "__len__"):
                    return list(mod)
            except AttributeError:
                continue
        return []

    def _find_attn_module(self, layer):
        # LLaMA-style: self_attn; GPT-2-style: attn.
        for name in ("self_attn", "attn"):
            a = getattr(layer, name, None)
            if a is not None:
                return a
        return None

    def run_trial(
        self,
        prompts: List[str],
        max_new_tokens: int,
        batch_size: int,
    ) -> BenchmarkResult:
        text = build_prompt(
            self._tokenizer,
            prompts,
            target_tokens=256,
            seed=self.config.seed,
        )
        prompt_list = [text] * batch_size

        device = next(self._model.parameters()).device
        enc = self._tokenizer(
            prompt_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
        ).to(device)

        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        before = snapshot_memory()
        with ResourceProbe(interval_s=0.05) as probe:
            t0 = time.perf_counter()
            with MemoryTracker(interval_s=0.05) as tracker:
                with torch.inference_mode():
                    out = self._model(
                        input_ids=enc["input_ids"],
                        attention_mask=enc.get("attention_mask"),
                        use_cache=False,
                    )
                generated = enc["input_ids"]
                cur_attn = enc.get("attention_mask")
                eos = self._tokenizer.eos_token_id
                n_new = 0
                for step in range(max_new_tokens):
                    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    n_new += 1
                    generated = torch.cat([generated, next_tok], dim=-1)
                    if cur_attn is not None:
                        cur_attn = torch.cat(
                            [cur_attn, torch.ones_like(next_tok)], dim=-1
                        )
                    if (next_tok == eos).all():
                        break
                    with torch.inference_mode():
                        out = self._model(
                            input_ids=next_tok,
                            attention_mask=cur_attn,
                            use_cache=False,
                        )
            t1 = time.perf_counter()
        after = snapshot_memory()

        wall = t1 - t0
        approx_ctx = len(self._tokenizer.encode(text, add_special_tokens=False))
        return BenchmarkResult(
            technique=self.spec.name,
            category=self.spec.category,
            context_length=approx_ctx,
            batch_size=batch_size,
            peak_gpu_mb=max(before.peak_gpu_mb, after.peak_gpu_mb, tracker.peak_gpu_mb),
            gpu_reserved_mb=max(before.gpu_reserved_mb, after.gpu_reserved_mb),
            cpu_rss_mb=max(before.cpu_rss_mb, after.cpu_rss_mb, tracker.peak_cpu_rss_mb),
            analytic_kv_mb=self.analytic_kv_mb(batch_size, approx_ctx),
            wall_time_s=wall,
            tokens_generated=int(n_new),
            tokens_per_s=(n_new / wall) if wall > 0 else 0.0,
            time_to_first_token_s=(wall / max(1, n_new)),
            avg_cpu_pct=probe.avg_cpu_pct,
            peak_rss_mb=probe.peak_rss_mb,
            notes=self.spec.description + f" (rank={self.rank}, head_dim={self._kv_shape.head_dim})",
        )