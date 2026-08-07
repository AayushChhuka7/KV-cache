"""Grouped Query Attention (GQA) technique.

We load a HuggingFace model that natively uses GQA (e.g.
``TinyLlama/TinyLlama-1.1B-Chat-v1.0``, Llama-3.x, Mistral, Gemma, ...)
and measure the same metrics as the baseline. Because the model has
fewer KV heads than query heads, the KV cache memory drops by a factor
of approximately ``num_query_heads / num_kv_heads``.
"""

from __future__ import annotations

from typing import List

from ..config import BenchmarkConfig
from ..utils.models import (
    dtype_from_string,
    get_torch_device,
    load_model_and_tokenizer,
)
from ..utils.prompts import build_prompt
from .base import BenchmarkResult, KVTechnique, TechniqueSpec


class GQAModel(KVTechnique):
    spec = TechniqueSpec(
        name="GQA",
        category="attention",
        description="Grouped-Query Attention model: fewer KV heads than query heads; native HuggingFace GQA model.",
    )

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self._model_name = config.gqa_model

    def setup(self) -> None:
        device = get_torch_device(use_cuda=self.config.use_cuda)
        dtype = dtype_from_string(self.config.dtype)
        self.logger.info("Loading GQA model %s in %s", self._model_name, dtype)
        self._model, self._tokenizer = load_model_and_tokenizer(
            self._model_name,
            dtype=dtype,
            device=device,
            use_cuda=self.config.use_cuda,
        )
        self._detect_shape()
        cfg = self._model.config
        n_q = getattr(cfg, "num_attention_heads", None) or 0
        n_kv = getattr(cfg, "num_key_value_heads", None) or n_q
        if n_q and n_kv and n_kv < n_q:
            self.spec.extra["group_size"] = n_q // n_kv
            self.spec.description = (
                f"GQA: {n_q} query heads sharing {n_kv} KV heads "
                f"(group size {n_q // n_kv}); native HuggingFace model."
            )

    def analytic_kv_mb(self, batch: int, seq_len: int) -> float:
        return self._kv_shape.cache_size_mb(batch, seq_len) if self._kv_shape else 0.0

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
        prompts_batch = [text] * batch_size

        def _generate(prompts: List[str], max_new_tokens: int):
            return self._hf_generate(
                self._model, self._tokenizer, prompts, max_new_tokens
            )

        approx_ctx = len(self._tokenizer.encode(text, add_special_tokens=False))
        return self._measure_trial(
            generate_fn=_generate,
            context_length=approx_ctx,
            batch_size=batch_size,
        )
