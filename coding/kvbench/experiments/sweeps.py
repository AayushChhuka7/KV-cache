"""Sweep construction helpers.

A "sweep" is a list of ``(technique, params)`` pairs to run. Each
sweep is a flat list — the runner iterates over it.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

from ..config import BenchmarkConfig
from ..techniques import (
    BaselineKV,
    FP8SimulatedKV,
    GQAModel,
    Int8SimulatedKV,
    LowRankCompression,
    MiniKVSimulator,
    NVFP4Simulator,
    PagedAttentionVLLM,
    CPUOffloading,
    XKVSimulator,
)


def build_default_techniques(config: BenchmarkConfig) -> List:
    """Return the canonical list of techniques for the paper comparison."""
    out = [
        BaselineKV(config),
        Int8SimulatedKV(config),
        FP8SimulatedKV(config),
        GQAModel(config),
        LowRankCompression(config, rank=64),
        PagedAttentionVLLM(config),
        CPUOffloading(config, gpu_window=256),
    ]
    if config.run_simulations:
        out.extend(
            [
                NVFP4Simulator(config),
                MiniKVSimulator(config),
                XKVSimulator(config),
            ]
        )
    return out


def build_default_sweep(
    config: BenchmarkConfig,
) -> List[Tuple[object, dict]]:
    """Return a list of ``(technique_factory, kwargs)`` to run.

    We use factories so that each trial starts with a fresh model
    instance — HuggingFace ``generate`` keeps internal state that
    would otherwise leak between trials.
    """
    sweep: List[Tuple[object, dict]] = []

    def _factory(cls, **kwargs):
        def _make():
            return cls(config, **kwargs)
        return _make

    sweep.append((_factory(BaselineKV), {}))
    sweep.append((_factory(Int8SimulatedKV), {}))
    sweep.append((_factory(FP8SimulatedKV), {}))
    sweep.append((_factory(GQAModel), {}))
    sweep.append((_factory(LowRankCompression, rank=64), {}))
    sweep.append((_factory(LowRankCompression, rank=128), {}))
    sweep.append((_factory(PagedAttentionVLLM), {}))
    sweep.append((_factory(CPUOffloading, gpu_window=256), {}))
    sweep.append((_factory(CPUOffloading, gpu_window=1024), {}))

    if config.run_simulations:
        sweep.append((_factory(NVFP4Simulator), {}))
        sweep.append((_factory(MiniKVSimulator), {}))
        sweep.append((_factory(XKVSimulator), {}))

    return sweep


def sweep_context_lengths(
    config: BenchmarkConfig,
) -> Iterable[int]:
    return list(config.context_lengths)


def sweep_batch_sizes(
    config: BenchmarkConfig,
) -> Iterable[int]:
    return list(config.batch_sizes)