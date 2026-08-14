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

        # The runner passes ``batch_size`` prompts already built at the
        # requested context length; use them verbatim so the measured
        # context actually reflects the sweep point.
        prompt_list = list(prompts[:batch_size]) if prompts else []
        text = prompt_list[0] if prompt_list else "Hello."

        def _generate(_prompts: List[str], max_new_tokens: int):
            # ``_measure_trial`` passes ``config.prompts``; ignore it and
            # use the context-length prompt list built for this cell.
            return self._hf_generate(
                self._model, self._tokenizer, prompt_list, max_new_tokens
            )

        # Approximate context length as tokenized length of the prompt.
        approx_ctx = len(self._tokenizer.encode(text, add_special_tokens=False))

        return self._measure_trial(
            generate_fn=_generate,
            context_length=approx_ctx,
            batch_size=batch_size,
        )
