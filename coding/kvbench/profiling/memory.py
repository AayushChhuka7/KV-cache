"""Memory measurement helpers.

We track three families of memory:

* ``peak_gpu_mb``  — peak GPU memory allocated by PyTorch, or 0 if no GPU.
* ``gpu_reserved_mb`` — peak GPU memory reserved by the caching allocator.
* ``cpu_rss_mb``   — peak resident set size of the Python process.

We additionally provide an analytic estimator based on a model's KV
shape, which is useful for very low-level comparisons.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterator, List, Optional

import psutil

from ..utils.models import KVShape


@dataclass
class MemoryReading:
    """A single memory snapshot."""

    peak_gpu_mb: float = 0.0
    gpu_reserved_mb: float = 0.0
    gpu_util_pct: float = 0.0  # 0-100, may be 0 if NVML is unavailable
    cpu_rss_mb: float = 0.0
    cpu_vms_mb: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _read_torch_gpu_mb() -> tuple[float, float]:
    try:
        import torch
        if torch.cuda.is_available():
            return (
                float(torch.cuda.max_memory_allocated()) / (1024 ** 2),
                float(torch.cuda.max_memory_reserved()) / (1024 ** 2),
            )
    except Exception:
        pass
    return 0.0, 0.0


def _read_nvml_util_pct() -> float:
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        return float(util.gpu)
    except Exception:
        return 0.0


def snapshot_memory(include_gpu_util: bool = True) -> MemoryReading:
    """Return a single memory snapshot."""
    peak, reserved = _read_torch_gpu_mb()
    proc = psutil.Process(os.getpid())
    rss = proc.memory_info().rss / (1024 ** 2)
    vms = proc.memory_info().vms / (1024 ** 2)
    gpu_util = _read_nvml_util_pct() if include_gpu_util else 0.0
    return MemoryReading(
        peak_gpu_mb=peak,
        gpu_reserved_mb=reserved,
        gpu_util_pct=gpu_util,
        cpu_rss_mb=rss,
        cpu_vms_mb=vms,
    )


def resident_set_size_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def cache_size_from_shape(
    shape: KVShape,
    batch: int,
    seq_len: int,
    bytes_per_element: Optional[int] = None,
) -> float:
    """Analytic estimate of KV cache memory in megabytes."""
    if bytes_per_element is None:
        bytes_per_element = shape.bytes_per_element
    return (
        2
        * shape.num_layers
        * shape.num_kv_heads
        * shape.head_dim
        * bytes_per_element
        * batch
        * seq_len
    ) / (1024 ** 2)


def measure_cache_memory(
    model,
    forward_fn,
    shape: KVShape,
    batch: int,
    seq_len: int,
    bytes_per_element: Optional[int] = None,
) -> Dict[str, float]:
    """Compare the measured peak GPU memory to the analytic estimate.

    ``forward_fn`` should accept a tensor of shape (batch, seq_len) and
    return the model outputs. We run a single forward pass on a dummy
    input after resetting the memory stats and capture both numbers.
    """
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    ids = torch.zeros((batch, seq_len), dtype=torch.long, device=next(model.parameters()).device)
    with torch.no_grad():
        forward_fn(ids)

    measured = snapshot_memory()
    analytic = cache_size_from_shape(shape, batch, seq_len, bytes_per_element)
    return {
        "measured_peak_gpu_mb": measured.peak_gpu_mb,
        "measured_gpu_reserved_mb": measured.gpu_reserved_mb,
        "cpu_rss_mb": measured.cpu_rss_mb,
        "analytic_kv_mb": analytic,
    }


class MemoryTracker:
    """Background sampler that records memory at a fixed interval.

    Usage
    -----
    >>> tracker = MemoryTracker(interval_s=0.05)
    >>> with tracker:
    ...     do_work()
    >>> tracker.readings  # list[MemoryReading]
    """

    def __init__(self, interval_s: float = 0.05, include_gpu_util: bool = False):
        self.interval_s = float(interval_s)
        self.include_gpu_util = bool(include_gpu_util)
        self.readings: List[MemoryReading] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "MemoryTracker":
        self._stop.clear()
        self.readings.clear()
        self._thread = threading.Thread(
            target=self._run, name="MemoryTracker", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.readings.append(snapshot_memory(include_gpu_util=self.include_gpu_util))
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    @property
    def peak_gpu_mb(self) -> float:
        return max((r.peak_gpu_mb for r in self.readings), default=0.0)

    @property
    def peak_cpu_rss_mb(self) -> float:
        return max((r.cpu_rss_mb for r in self.readings), default=0.0)

    @property
    def peak_gpu_util_pct(self) -> float:
        return max((r.gpu_util_pct for r in self.readings), default=0.0)

    @property
    def avg_gpu_util_pct(self) -> float:
        vals = [r.gpu_util_pct for r in self.readings if r.gpu_util_pct > 0]
        if not vals:
            return 0.0
        return sum(vals) / len(vals)
