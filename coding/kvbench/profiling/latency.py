"""Latency and throughput measurement."""

from __future__ import annotations

import contextlib
import statistics
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Iterable, List, Optional


@dataclass
class LatencyResult:
    """Timing breakdown for a generation call."""

    wall_time_s: float
    prefill_time_s: float = 0.0
    decode_time_s: float = 0.0
    tokens_generated: int = 0
    tokens_per_s: float = 0.0
    time_to_first_token_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GenerationTimer:
    """High-resolution wall-clock + CUDA-event timer.

    Uses ``torch.cuda.Event`` when CUDA is available, otherwise falls back
    to ``time.perf_counter``. The prefill / decode phases are captured
    with events placed inside the model's generation loop when supported.
    """

    def __init__(self, use_cuda_events: bool = True):
        self.use_cuda_events = bool(use_cuda_events)
        self._cuda_available = False
        try:
            import torch
            self._cuda_available = torch.cuda.is_available() and use_cuda_events
        except Exception:
            self._cuda_available = False

    def now(self) -> float:
        if self._cuda_available:
            import torch
            return torch.cuda.Event(enable_timing=True).record()
        return time.perf_counter()


@contextlib.contextmanager
def time_generation(
    timer: Optional[GenerationTimer] = None,
):
    """Context manager that yields a callable for recording the first-token
    timestamp and reports the wall-clock duration.

    Example
    -------
    >>> with time_generation() as rec:
    ...     out = model.generate(..., streamer=rec.streamer())
    """
    timer = timer or GenerationTimer()
    state = {"first_token_at": None, "start": time.perf_counter()}

    def mark_first_token():
        if state["first_token_at"] is None:
            state["first_token_at"] = time.perf_counter()

    state["mark_first_token"] = mark_first_token
    yield state
    state["end"] = time.perf_counter()


class ThroughputMeter:
    """Accumulates per-token timings to compute steady-state throughput.

    Useful when the framework needs to separate prefill cost from the
    autoregressive per-token decode cost.
    """

    def __init__(self) -> None:
        self.per_token_times: List[float] = []

    def record(self, dt: float) -> None:
        self.per_token_times.append(float(dt))

    def tokens_per_s(self) -> float:
        if not self.per_token_times:
            return 0.0
        total = sum(self.per_token_times)
        if total <= 0:
            return 0.0
        return len(self.per_token_times) / total

    def median_per_token_s(self) -> float:
        if not self.per_token_times:
            return 0.0
        return statistics.median(self.per_token_times)

    def summary(self) -> Dict[str, float]:
        if not self.per_token_times:
            return {"tokens": 0, "mean_s": 0.0, "median_s": 0.0, "tokens_per_s": 0.0}
        mean = statistics.mean(self.per_token_times)
        med = statistics.median(self.per_token_times)
        return {
            "tokens": len(self.per_token_times),
            "mean_s": mean,
            "median_s": med,
            "tokens_per_s": 1.0 / mean if mean > 0 else 0.0,
        }
