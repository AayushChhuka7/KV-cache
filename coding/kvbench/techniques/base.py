"""Base interface for KV cache techniques.

A "technique" is anything that affects the memory, latency, or
throughput of the KV cache during autoregressive decoding. Some
techniques (Baseline, Quantization, GQA) modify the model itself or
its data path; others (PagedAttention, Offloading) modify the runtime;
a final class (NVFP4, MiniKV, xKV) is implemented via analytic
simulation because their kernels are proprietary or unavailable.

Each technique is responsible for:

  1. Loading the model (and tokenizer) in :meth:`setup`.
  2. Computing an analytic KV cache size estimate (:meth:`analytic_kv_mb`).
  3. Running a single timed trial (:meth:`run_trial`) and returning a
     :class:`BenchmarkResult` with all measured numbers.
  4. Cleaning up in :meth:`teardown` (e.g., releasing CUDA memory).

The base class deliberately keeps the surface small so that a new
technique can be added in a single file.
"""

from __future__ import annotations

import abc
import gc
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional

from ..config import BenchmarkConfig
from ..profiling.latency import time_generation
from ..profiling.memory import (
    MemoryTracker,
    snapshot_memory,
)
from ..profiling.resources import ResourceProbe
from ..utils.logging import get_logger
from ..utils.models import (
    KVShape,
    detect_model_kv_shape,
)


@dataclass
class BenchmarkResult:
    """The output of a single (technique, config) measurement.

    All numeric fields are aggregated across trials by the runner.
    Strings (``notes``, ``category``, ``technique``) are descriptive
    and copied verbatim into the final table.
    """

    technique: str
    category: str
    context_length: int
    batch_size: int

    # Memory
    peak_gpu_mb: float = 0.0
    gpu_reserved_mb: float = 0.0
    cpu_rss_mb: float = 0.0
    analytic_kv_mb: float = 0.0

    # Latency / throughput
    wall_time_s: float = 0.0
    prefill_time_s: float = 0.0
    decode_time_s: float = 0.0
    tokens_generated: int = 0
    tokens_per_s: float = 0.0
    time_to_first_token_s: float = 0.0

    # Resources
    avg_gpu_util_pct: float = 0.0
    peak_gpu_util_pct: float = 0.0
    avg_cpu_pct: float = 0.0
    peak_rss_mb: float = 0.0

    # Notes (e.g., "simulation", "vLLM", "low-rank r=64")
    notes: str = ""

    # Raw per-trial values (for downstream aggregation).
    raw: Dict[str, List[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TechniqueSpec:
    """Static description of a technique."""

    name: str
    category: str
    description: str
    is_simulation: bool = False
    requires_gpu: bool = True
    requires_vllm: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


class KVTechnique(abc.ABC):
    """Abstract base class for all KV cache techniques."""

    spec: TechniqueSpec

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self._model = None
        self._tokenizer = None
        self._kv_shape: Optional[KVShape] = None

    # --- Required interface -------------------------------------------------
    @abc.abstractmethod
    def setup(self) -> None:
        """Load the model, tokenizer, and any other state."""

    @abc.abstractmethod
    def run_trial(
        self,
        prompts: List[str],
        max_new_tokens: int,
        batch_size: int,
    ) -> BenchmarkResult:
        """Run a single timed trial."""

    @abc.abstractmethod
    def analytic_kv_mb(self, batch: int, seq_len: int) -> float:
        """Analytic estimate of the KV cache size in MB."""

    # --- Optional helpers ---------------------------------------------------
    def teardown(self) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        try:
            import torch
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
        gc.collect()

    # --- Helpers used by subclasses ----------------------------------------
    def _detect_shape(self) -> KVShape:
        from ..utils.models import dtype_from_string
        if self._kv_shape is not None:
            return self._kv_shape
        if self._model is None:
            raise RuntimeError("Model not loaded; call setup() first.")
        self._kv_shape = detect_model_kv_shape(
            self._model, dtype=dtype_from_string(self.config.dtype)
        )
        return self._kv_shape

    def _measure_trial(
        self,
        generate_fn,
        context_length: int,
        batch_size: int,
    ) -> BenchmarkResult:
        """Run :param generate_fn and capture all the standard metrics.

        ``generate_fn`` should accept ``(prompts, max_new_tokens)`` and
        return a tuple ``(tokens_generated, first_token_time)``.
        """
        import torch

        # Reset CUDA peak counters before each trial.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        before = snapshot_memory()
        with ResourceProbe(interval_s=0.05, gpu=torch.cuda.is_available()) as probe:
            t0 = time.perf_counter()
            with MemoryTracker(interval_s=0.05) as tracker:
                with time_generation() as tg:
                    n_tokens, ttft = generate_fn(
                        self.config.prompts,
                        self.config.max_new_tokens,
                    )
                    tg["mark_first_token"]()
            t1 = time.perf_counter()
        after = snapshot_memory()

        wall = t1 - t0
        tokens_per_s = n_tokens / wall if wall > 0 else 0.0

        return BenchmarkResult(
            technique=self.spec.name,
            category=self.spec.category,
            context_length=context_length,
            batch_size=batch_size,
            peak_gpu_mb=max(before.peak_gpu_mb, after.peak_gpu_mb, tracker.peak_gpu_mb),
            gpu_reserved_mb=max(before.gpu_reserved_mb, after.gpu_reserved_mb),
            cpu_rss_mb=max(before.cpu_rss_mb, after.cpu_rss_mb, tracker.peak_cpu_rss_mb),
            analytic_kv_mb=self.analytic_kv_mb(batch_size, context_length),
            wall_time_s=wall,
            tokens_generated=n_tokens,
            tokens_per_s=tokens_per_s,
            time_to_first_token_s=ttft,
            avg_gpu_util_pct=probe.avg_cpu_pct,  # placeholder if no GPU
            peak_gpu_util_pct=probe.peak_gpu_pct,
            avg_cpu_pct=probe.avg_cpu_pct,
            peak_rss_mb=probe.peak_rss_mb,
            notes=self.spec.description,
        )

    # --- HF generation helper ----------------------------------------------
    def _hf_generate(
        self,
        model,
        tokenizer,
        prompts: List[str],
        max_new_tokens: int,
    ):
        """Tokenize, generate, and return (n_tokens, ttft).

        ``ttft`` is approximated by the time of the first token of the
        prefill. We approximate it by re-running the encoder/embeddings
        once if necessary; for benchmarking purposes the simple
        wall-clock-around-``generate`` is what most papers report.
        """
        import torch
        import time

        device = next(model.parameters()).device

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
        ).to(device)

        # Warmup the KV cache layout (no timing impact since this is a
        # single forward pass with a separate measure).
        gen_kwargs = dict(
            input_ids=enc["input_ids"],
            attention_mask=enc.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
        # Time only the generate call.
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**gen_kwargs)
        t1 = time.perf_counter()

        n_new = out.shape[-1] - enc["input_ids"].shape[-1]
        # Approximate TTFT as wall_time / tokens (overestimate on long contexts).
        # Real TTFT would require a streamer; the approximation is documented
        # in LIMITATIONS.md.
        ttft = (t1 - t0) / max(1, n_new)
        return int(n_new), float(ttft)

    # --- Helpers ----------------------------------------------------------
    def _aggregate_raw(self, results: List[BenchmarkResult]) -> Dict[str, List[float]]:
        if not results:
            return {}
        keys = [
            "wall_time_s",
            "tokens_per_s",
            "peak_gpu_mb",
            "cpu_rss_mb",
            "time_to_first_token_s",
        ]
        return {k: [getattr(r, k) for r in results] for k in keys}

    def average(self, results: List[BenchmarkResult]) -> BenchmarkResult:
        if not results:
            raise ValueError("No results to average.")
        first = results[0]
        out = BenchmarkResult(
            technique=first.technique,
            category=first.category,
            context_length=first.context_length,
            batch_size=first.batch_size,
            notes=first.notes,
        )

        numeric_fields = [
            "peak_gpu_mb",
            "gpu_reserved_mb",
            "cpu_rss_mb",
            "analytic_kv_mb",
            "wall_time_s",
            "prefill_time_s",
            "decode_time_s",
            "tokens_generated",
            "tokens_per_s",
            "time_to_first_token_s",
            "avg_gpu_util_pct",
            "peak_gpu_util_pct",
            "avg_cpu_pct",
            "peak_rss_mb",
        ]
        for f in numeric_fields:
            vals = [float(getattr(r, f)) for r in results]
            if vals:
                setattr(out, f, sum(vals) / len(vals))

        out.raw = self._aggregate_raw(results)
        return out
