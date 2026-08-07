"""Resource probes — CPU / GPU stats independent of torch.cuda APIs."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import psutil


@dataclass
class CpuStats:
    """A single CPU sample."""

    cpu_percent: float
    rss_mb: float
    vms_mb: float
    num_threads: int
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GpuStats:
    """A single GPU sample (via NVML or zero-filled when unavailable)."""

    index: int
    util_pct: float
    mem_used_mb: float
    mem_total_mb: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_gpu_stats() -> List[GpuStats]:
    """Best-effort NVML GPU stats; empty list if pynvml is unavailable."""
    out: List[GpuStats] = []
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            out.append(
                GpuStats(
                    index=i,
                    util_pct=float(util.gpu),
                    mem_used_mb=mem.used / (1024 ** 2),
                    mem_total_mb=mem.total / (1024 ** 2),
                )
            )
    except Exception:
        return out
    return out


class ResourceProbe:
    """Background CPU/GPU sampler."""

    def __init__(self, interval_s: float = 0.1, gpu: bool = True):
        self.interval_s = float(interval_s)
        self.gpu = bool(gpu)
        self.cpu_samples: List[CpuStats] = []
        self.gpu_samples: List[GpuStats] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc = psutil.Process(os.getpid())

    def __enter__(self) -> "ResourceProbe":
        self.cpu_samples.clear()
        self.gpu_samples.clear()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="ResourceProbe", daemon=True
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
                mi = self._proc.memory_info()
                self.cpu_samples.append(
                    CpuStats(
                        cpu_percent=float(self._proc.cpu_percent(interval=None)),
                        rss_mb=mi.rss / (1024 ** 2),
                        vms_mb=mi.vms / (1024 ** 2),
                        num_threads=self._proc.num_threads(),
                    )
                )
                if self.gpu:
                    self.gpu_samples.extend(get_gpu_stats())
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    @property
    def peak_rss_mb(self) -> float:
        return max((s.rss_mb for s in self.cpu_samples), default=0.0)

    @property
    def avg_cpu_pct(self) -> float:
        vals = [s.cpu_percent for s in self.cpu_samples if s.cpu_percent > 0]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def peak_gpu_pct(self) -> float:
        return max((s.util_pct for s in self.gpu_samples), default=0.0)
