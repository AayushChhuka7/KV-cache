"""Baseline KV cache.

Loads a model in the requested dtype and runs standard autoregressive
generation with the default KV cache layout provided by HuggingFace.
"""

from __future__ import annotations

from typing import List

from ..config import BenchmarkConfig
from ..utils.models import (
    dtype_from_string,
    load_model_and_tokenizer,
)
from ..utils.prompts import build_prompt
from .base import BenchmarkResult, KVTechnique, TechniqueSpec


class BaselineKV(KVTechnique):
    spec = TechniqueSpec(
        name="Baseline FP16",
        category="baseline",
        description="HuggingFace default KV cache in FP16/FP32. No optimization.",
    )

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self._model_name = config.model

    def setup(self) -> None:
        from ..utils.models import get_torch_device
        device = get_torch_device(use_cuda=self.config.use_cuda)
        dtype = dtype_from_string(self.config.dtype)
        self.logger.info(
            "Loading baseline model %s in %s on %s",
            self._model_name, dtype, device,
        )
        self._model, self._tokenizer = load_model_and_tokenizer(
            self._model_name,
            dtype=dtype,
            device=device,
            use_cuda=self.config.use_cuda,
        )
        self._detect_shape()
        self.logger.info(
            "KV shape: L=%d H_kv=%d D=%d B=%d -> %d bytes/token",
            self._kv_shape.num_layers,
            self._kv_shape.num_kv_heads,
            self._kv_shape.head_dim,
            self._kv_shape.bytes_per_element,
            self._kv_shape.kv_per_token,
        )

    def analytic_kv_mb(self, batch: int, seq_len: int) -> float:
        return self._kv_shape.cache_size_mb(batch, seq_len) if self._kv_shape else 0.0

    def run_trial(
        self,
        prompts: List[str],
        max_new_tokens: int,
        batch_size: int,
    ) -> BenchmarkResult:
        assert self._model is not None and self._tokenizer is not None

        ctx_tokens = prompts[0] if isinstance(prompts, list) and prompts else "Hello."
        # Build a deterministic prompt of length ctx_tokens by using the
        # trial's request context. The runner passes the context length
        # indirectly via the chosen prompt set; here we approximate by
        # encoding a short paragraph and expanding.
        text = build_prompt(
            self._tokenizer,
            prompts,
            target_tokens=max(64, len(ctx_tokens) if isinstance(ctx_tokens, str) else 256),
            seed=self.config.seed,
        )
        prompts_batch = [text] * batch_size

        def _generate(prompts: List[str], max_new_tokens: int):
            return self._hf_generate(
                self._model, self._tokenizer, prompts, max_new_tokens
            )

        # Approximate context length as tokenized length of the prompt.
        approx_ctx = len(self._tokenizer.encode(text, add_special_tokens=False))

        return self._measure_trial(
            generate_fn=_generate,
            context_length=approx_ctx,
            batch_size=batch_size,
        )
