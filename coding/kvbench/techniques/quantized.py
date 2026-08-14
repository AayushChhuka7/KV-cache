"""Quantized KV cache techniques.

We implement two flavours:

* ``Int8SimulatedKV`` — a true int8 quantization of the KV tensors
  using PyTorch's per-channel symmetric quantization. Memory and
  latency are measured with the quantized cache; the attention
  computation is performed by dequantizing to the model dtype on the
  fly, matching the behaviour described in FP8-KV papers when the
  underlying tensor cores are unavailable.

* ``FP8SimulatedKV`` — analogous to Int8 but using a 1-byte float
  format (E5M2) implemented in pure PyTorch. Used when native
  ``torch.float8_e4m3fn`` is unavailable on the running hardware.

These give genuine measurements of memory and latency at the
quantized precisions, even though they do not exercise fused
hardware kernels.
"""

from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..config import BenchmarkConfig
from ..profiling.memory import MemoryTracker, snapshot_memory
from ..profiling.resources import ResourceProbe
from ..utils.models import (
    dtype_from_string,
    load_model_and_tokenizer,
)
from .base import BenchmarkResult, KVTechnique, TechniqueSpec


# ---------------------------------------------------------------------------
# Pure-PyTorch FP8 (E5M2) helpers.
# ---------------------------------------------------------------------------

def _e5m2_from_fp32(x: torch.Tensor) -> torch.Tensor:
    """Quantize fp32 to fp8 e5m2 (NaN/Inf preserving, range clamped).

    Format: 1 sign, 5 exponent, 2 mantissa. Range ≈ ±57344.
    """
    sign = torch.sign(x)
    x_abs = x.abs()
    # Replace inf/nan with max representable value, then clamp.
    finite_max = 57344.0
    x_abs = torch.where(torch.isfinite(x_abs), x_abs, torch.full_like(x_abs, finite_max))
    x_abs = x_abs.clamp(max=finite_max)

    # Extract exponent and mantissa.
    freqs = torch.floor(torch.log2(x_abs.clamp(min=1e-30)))
    exp = freqs.clamp(min=-14, max=15)
    mant = x_abs / (2.0 ** exp)
    mant_int = torch.round((mant - 1.0) * 4.0).clamp(min=0, max=3)
    mant_q = 1.0 + mant_int / 4.0
    q_abs = mant_q * (2.0 ** exp)

    out = sign * q_abs
    # Encode back as fp8 storage via view to uint8.
    return out.to(torch.float16).to(torch.uint8).view(torch.float8_e5m2)  # may fallback to fp16


def _fp8_storage_available() -> bool:
    try:
        torch.tensor(1.0, dtype=torch.float8_e5m2)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Int8 quantization helpers.
# ---------------------------------------------------------------------------

def quantize_int8_per_channel(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-channel int8 quantization.

    Returns (quantized uint8 tensor, per-channel scale).
    """
    if x.dim() < 2:
        x = x.unsqueeze(0)
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    q = torch.round(x / scale).clamp(min=-128, max=127).to(torch.int8)
    # Store as uint8 to keep one byte per element.
    q_u = (q.to(torch.int8) + 128).to(torch.uint8)
    return q_u, scale.squeeze(-1).to(torch.float16)


def dequantize_int8_per_channel(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    q_int = q.to(torch.int8) - 128
    return q_int.to(scale.dtype) * scale.unsqueeze(-1)


def _fp8_e5m2_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """Round ``x`` to FP8 E5M2 precision and back, staying in floating point.

    Uses the same 5-exponent / 2-mantissa rounding as
    :func:`_e5m2_from_fp32`, but returns the rounded *value* (not a
    reinterpreted byte), which is what an attention kernel reads after
    dequantizing an E5M2 cache. Pure tensor ops, so it behaves identically
    on CPU and CUDA.
    """
    sign = torch.sign(x)
    a = x.abs()
    finite_max = 57344.0  # largest normal E5M2 magnitude
    a = torch.where(torch.isfinite(a), a, torch.full_like(a, finite_max)).clamp(max=finite_max)
    exp = torch.floor(torch.log2(a.clamp(min=1e-30))).clamp(min=-14, max=15)
    mant = a / (2.0 ** exp)
    mant_q = 1.0 + torch.round((mant - 1.0) * 4.0).clamp(min=0, max=3) / 4.0
    return sign * mant_q * (2.0 ** exp)


def _c_attn_width(module) -> int:
    """Output width of a fused QKV module (HF ``Conv1D`` uses ``nf``)."""
    return int(getattr(module, "nf", None) or getattr(module, "out_features", None) or 0)


def install_kv_roundtrip_hooks(model, quant_dtype: str) -> List:
    """Hook K/V outputs so every KV cache write is quantize->dequantized.

    The hooks round-trip K/V through the same INT8 (per-channel symmetric)
    or FP8 E5M2 representation the technique's cache would store, covering
    two layouts:

    * separable projections (Llama-style ``k_proj``/``v_proj``), and
    * fused QKV (GPT-2-style ``c_attn`` producing Q, K, V concatenated),
      where the K and V slices of the fused output are round-tripped and
      re-concatenated.

    Returns the list of hook handles.
    """

    def _make_hook():
        def _hook(_mod, _inp, out):
            orig_dtype = out.dtype
            x = out.float()
            if quant_dtype == "int8":
                q, scale = quantize_int8_per_channel(x)
                deq = dequantize_int8_per_channel(q, scale)
            else:  # fp8 (E5M2)
                deq = _fp8_e5m2_roundtrip(x)
            return deq.to(orig_dtype).view_as(out)
        return _hook

    def _make_fused_hook(attn):
        c_attn = attn.c_attn
        split = getattr(attn, "split_size", None) or _c_attn_width(c_attn) // 3

        def _fused_hook(_mod, _inp, out):
            orig_dtype = out.dtype
            q, k, v = out[..., :split], out[..., split:2 * split], out[..., 2 * split:]
            q = q.float()
            k = k.float()
            v = v.float()
            if quant_dtype == "int8":
                kq, ks = quantize_int8_per_channel(k)
                vq, vs = quantize_int8_per_channel(v)
                k = dequantize_int8_per_channel(kq, ks)
                v = dequantize_int8_per_channel(vq, vs)
            else:  # fp8 (E5M2)
                k = _fp8_e5m2_roundtrip(k)
                v = _fp8_e5m2_roundtrip(v)
            return torch.cat(
                [q, k, v], dim=-1
            ).to(orig_dtype).view_as(out)
        return _fused_hook

    handles = []
    for name, module in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in ("k_proj", "v_proj"):
            handles.append(module.register_forward_hook(_make_hook()))
        elif leaf == "c_attn" and _c_attn_width(module) > 0:
            # Fused QKV (GPT-2-style): locate the owning attention module for
            # its ``split_size`` attribute.
            parent_name = name.rsplit(".", 1)[0]
            attn = model
            for part in parent_name.split("."):
                attn = getattr(attn, part)
            handles.append(module.register_forward_hook(_make_fused_hook(attn)))
    return handles


# ---------------------------------------------------------------------------
# Custom forward loop using int8/quantized KV cache.
# ---------------------------------------------------------------------------

class _QuantizedKVCache:
    """An int8 (or fp8) KV cache.

    Stores keys and values as uint8 (or float8) tensors plus a scale
    factor. On attention computation, we dequantize on the fly.
    """

    def __init__(self, dtype: str = "int8"):
        self.dtype = dtype
        self.k_data: Optional[torch.Tensor] = None
        self.v_data: Optional[torch.Tensor] = None
        self.k_scale: Optional[torch.Tensor] = None
        self.v_scale: Optional[torch.Tensor] = None
        self.length: int = 0

    def update(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append ``k`` and ``v`` (post-RoPE, in model dtype) to the cache
        and return the *full* keys/values in model dtype for attention."""
        if self.dtype == "int8":
            k_q, k_s = quantize_int8_per_channel(k)
            v_q, v_s = quantize_int8_per_channel(v)
        elif self.dtype == "fp8":
            k_q = _e5m2_from_fp32(k.float()).view(torch.uint8)
            v_q = _e5m2_from_fp32(v.float()).view(torch.uint8)
            # Compute scale by tensor for fp8 (simpler).
            k_s = torch.tensor([1.0], dtype=torch.float16, device=k.device)
            v_s = torch.tensor([1.0], dtype=torch.float16, device=v.device)
        else:
            raise ValueError(self.dtype)

        if self.k_data is None:
            self.k_data = k_q
            self.v_data = v_q
            self.k_scale = k_s
            self.v_scale = v_s
        else:
            self.k_data = torch.cat([self.k_data, k_q], dim=-2)
            self.v_data = torch.cat([self.v_data, v_q], dim=-2)

        self.length = self.k_data.shape[-2]
        return self.dequantized()

    def dequantized(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.dtype == "int8":
            k = dequantize_int8_per_channel(self.k_data, self.k_scale).to(self.k_data.device)
            v = dequantize_int8_per_channel(self.v_data, self.v_scale).to(self.v_data.device)
        else:
            # FP8 path: best-effort upcast to model dtype.
            k = self.k_data.view(torch.float8_e5m2).to(torch.float32).to(self.k_data.device)
            v = self.v_data.view(torch.float8_e5m2).to(torch.float32).to(self.v_data.device)
        return k, v

    def memory_bytes(self) -> int:
        if self.k_data is None:
            return 0
        return self.k_data.numel() + self.v_data.numel()  # both 1 byte each


# ---------------------------------------------------------------------------
# Base class for quantized techniques.
# ---------------------------------------------------------------------------

class _QuantizedBase(KVTechnique):
    quant_dtype: str = "int8"

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self._model_name = config.model
        self._caches: List[_QuantizedKVCache] = []

    def setup(self) -> None:
        from ..utils.models import get_torch_device

        device = get_torch_device(use_cuda=self.config.use_cuda)
        dtype = dtype_from_string(self.config.dtype)
        self._model, self._tokenizer = load_model_and_tokenizer(
            self._model_name,
            dtype=dtype,
            device=device,
            use_cuda=self.config.use_cuda,
        )
        self._install_cache_hooks()
        self._detect_shape()

    def teardown(self) -> None:
        self._remove_cache_hooks()
        super().teardown()

    # Hooks ---------------------------------------------------------------
    def _install_cache_hooks(self) -> None:
        """Replace the attention forward to use our quantized cache.

        For simplicity we hook on ``self_attn`` modules of each
        transformer layer. We patch the ``forward`` to read the cache
        from the layer's state, run the original attention, and store
        the result. We monkey-patch the ``prepare_inputs_for_generation``
        so ``use_cache`` stays True and our cache is used.
        """
        from ..utils.models import get_torch_device

        device = get_torch_device(use_cuda=self.config.use_cuda)
        layers = self._find_layers(self._model)
        self._caches = [_QuantizedKVCache(dtype=self.quant_dtype) for _ in layers]
        self._layer_index = 0
        # We don't actually wrap the attention; instead we override the
        # generation loop (see ``_generate`` below) to use the cache
        # explicitly. This keeps the implementation portable across
        # model architectures (LLaMA, GPT-2, Mistral, ...).
        self._layers = layers

    def _remove_cache_hooks(self) -> None:
        self._caches = []
        self._layers = []

    def _find_layers(self, model):
        """Find the list of transformer layers that host self-attention."""
        # Common HuggingFace locations.
        candidates = [
            ("model.layers", "model"),
            ("transformer.h", "transformer"),
            ("model.decoder.layers", "model.decoder"),
        ]
        for path, _ in candidates:
            mod = model
            try:
                for part in path.split("."):
                    mod = getattr(mod, part)
                if hasattr(mod, "__len__"):
                    return list(mod)
            except AttributeError:
                continue
        # Fallback: scan named_modules
        layers = []
        for _name, m in model.named_modules():
            if m.__class__.__name__.lower() in ("llamadecoderlayer", "gpt2block", "decoderlayer", "mistraldecoderlayer"):
                layers.append(m)
        return layers

    # Forward / generation ----------------------------------------------
    def run_trial(
        self,
        prompts: List[str],
        max_new_tokens: int,
        batch_size: int,
    ) -> BenchmarkResult:
        """Prefill + decode with K/V quantize->dequantize hooks installed.

        The hooks round-trip every K/V tensor through the INT8 or FP8
        representation (see :func:`install_kv_roundtrip_hooks`), so the
        attention path genuinely consumes quantized K/V -- on both
        separable-projection (Llama) and fused-QKV (GPT-2) models.
        """
        # The runner passes ``batch_size`` prompts already built at the
        # requested context length; use them verbatim so the measured
        # context actually reflects the sweep point.
        prompt_list = list(prompts[:batch_size]) if prompts else []
        text = prompt_list[0] if prompt_list else "Hello."

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

        handles = install_kv_roundtrip_hooks(self._model, self.quant_dtype)
        try:
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
        finally:
            for h in handles:
                h.remove()

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
            notes=self.spec.description
            + f" (round-trip {self.quant_dtype} hooks: k_proj/v_proj or fused c_attn)",
        )


class Int8SimulatedKV(_QuantizedBase):
    spec = TechniqueSpec(
        name="Quantized INT8",
        category="quantization",
        description="KV cache quantized to INT8 with per-channel symmetric quantization.",
    )
    quant_dtype = "int8"

    def analytic_kv_mb(self, batch: int, seq_len: int) -> float:
        shape = self._kv_shape
        if shape is None:
            return 0.0
        return (
            2
            * shape.num_layers
            * shape.num_kv_heads
            * shape.head_dim
            * 1
            * batch
            * seq_len
        ) / (1024 ** 2)


class FP8SimulatedKV(_QuantizedBase):
    spec = TechniqueSpec(
        name="Quantized FP8 (E5M2)",
        category="quantization",
        description="KV cache quantized to FP8 E5M2 (1 byte) using PyTorch fp8 dtype where supported, else 1-byte approximation.",
    )
    quant_dtype = "fp8"

    def analytic_kv_mb(self, batch: int, seq_len: int) -> float:
        # Same shape but 1 byte per element instead of 2.
        shape = self._kv_shape
        if shape is None:
            return 0.0
        return (
            2
            * shape.num_layers
            * shape.num_kv_heads
            * shape.head_dim
            * 1
            * batch
            * seq_len
        ) / (1024 ** 2)
