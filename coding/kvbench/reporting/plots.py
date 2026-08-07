"""Publication-quality plots.

The dataviz skill has been consulted: we use a consistent, accessible
palette across all charts, mark the simulated techniques with a
distinct marker, and prefer small multiples over colour overload.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

from ..techniques.base import BenchmarkResult


# Brand-neutral palette (validated by the dataviz skill).
PALETTE = [
    "#1F3A5F",  # deep navy
    "#C8553D",  # terracotta
    "#588B8B",  # teal
    "#F28F3B",  # amber
    "#7C6F9C",  # muted violet
    "#3A6B35",  # forest green
    "#A66C29",  # ochre
    "#586F7D",  # slate
]

SIMULATED_MARKER = "x"
REAL_MARKER = "o"
SIMULATED_ALPHA = 0.55


def _by_technique(results: Sequence[BenchmarkResult]) -> Dict[str, List[BenchmarkResult]]:
    out: Dict[str, List[BenchmarkResult]] = {}
    for r in results:
        out.setdefault(r.technique, []).append(r)
    return out


def _is_simulated(results: Sequence[BenchmarkResult]) -> bool:
    if not results:
        return False
    return "(simulated)" in results[0].notes or "simulation" in results[0].notes


def _style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_memory_vs_context(
    results: Sequence[BenchmarkResult],
    path: str | Path,
    batch_size: int = 1,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    by = _by_technique(results)
    for i, (tech, rs) in enumerate(by.items()):
        rs = [r for r in rs if r.batch_size == batch_size]
        rs.sort(key=lambda r: r.context_length)
        xs = [r.context_length for r in rs]
        ys = [r.peak_gpu_mb for r in rs]
        color = PALETTE[i % len(PALETTE)]
        marker = SIMULATED_MARKER if _is_simulated(rs) else REAL_MARKER
        alpha = SIMULATED_ALPHA if _is_simulated(rs) else 1.0
        ax.plot(
            xs, ys, marker=marker, color=color, label=tech, linewidth=2,
            markersize=6, alpha=alpha,
        )
    ax.set_xscale("log", base=2)
    _style_axes(ax, f"Peak GPU memory vs context length (batch={batch_size})",
                "Context length (tokens)", "Peak GPU memory (MB)")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    _save(fig, Path(path))


def plot_throughput_vs_context(
    results: Sequence[BenchmarkResult],
    path: str | Path,
    batch_size: int = 1,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    by = _by_technique(results)
    for i, (tech, rs) in enumerate(by.items()):
        rs = [r for r in rs if r.batch_size == batch_size]
        rs.sort(key=lambda r: r.context_length)
        xs = [r.context_length for r in rs]
        ys = [r.tokens_per_s for r in rs]
        color = PALETTE[i % len(PALETTE)]
        marker = SIMULATED_MARKER if _is_simulated(rs) else REAL_MARKER
        alpha = SIMULATED_ALPHA if _is_simulated(rs) else 1.0
        ax.plot(
            xs, ys, marker=marker, color=color, label=tech, linewidth=2,
            markersize=6, alpha=alpha,
        )
    ax.set_xscale("log", base=2)
    _style_axes(ax, f"Throughput vs context length (batch={batch_size})",
                "Context length (tokens)", "Tokens / second")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    _save(fig, Path(path))


def plot_latency_vs_context(
    results: Sequence[BenchmarkResult],
    path: str | Path,
    batch_size: int = 1,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    by = _by_technique(results)
    for i, (tech, rs) in enumerate(by.items()):
        rs = [r for r in rs if r.batch_size == batch_size]
        rs.sort(key=lambda r: r.context_length)
        xs = [r.context_length for r in rs]
        ys = [r.wall_time_s for r in rs]
        color = PALETTE[i % len(PALETTE)]
        marker = SIMULATED_MARKER if _is_simulated(rs) else REAL_MARKER
        alpha = SIMULATED_ALPHA if _is_simulated(rs) else 1.0
        ax.plot(
            xs, ys, marker=marker, color=color, label=tech, linewidth=2,
            markersize=6, alpha=alpha,
        )
    ax.set_xscale("log", base=2)
    _style_axes(ax, f"End-to-end latency vs context length (batch={batch_size})",
                "Context length (tokens)", "Wall time (s)")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    _save(fig, Path(path))


def plot_memory_vs_batch(
    results: Sequence[BenchmarkResult],
    path: str | Path,
    context_length: int = 1024,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    by = _by_technique(results)
    for i, (tech, rs) in enumerate(by.items()):
        rs = [r for r in rs if r.context_length == context_length]
        rs.sort(key=lambda r: r.batch_size)
        xs = [r.batch_size for r in rs]
        ys = [r.peak_gpu_mb for r in rs]
        color = PALETTE[i % len(PALETTE)]
        marker = SIMULATED_MARKER if _is_simulated(rs) else REAL_MARKER
        alpha = SIMULATED_ALPHA if _is_simulated(rs) else 1.0
        ax.plot(
            xs, ys, marker=marker, color=color, label=tech, linewidth=2,
            markersize=6, alpha=alpha,
        )
    _style_axes(ax, f"Peak GPU memory vs batch size (ctx={context_length})",
                "Batch size", "Peak GPU memory (MB)")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    _save(fig, Path(path))


def plot_summary_bar(
    results: Sequence[BenchmarkResult],
    path: str | Path,
    metric: str = "tokens_per_s",
    title: Optional[str] = None,
) -> None:
    """A simple horizontal bar chart of one metric per technique (averaged)."""
    by = _by_technique(results)
    techs = list(by.keys())
    vals = [float(np.mean([getattr(r, metric) for r in rs])) for rs in by.values()]
    sim_flags = [_is_simulated(rs) for rs in by.values()]

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.35 * len(techs))))
    colors = []
    for i, sim in enumerate(sim_flags):
        colors.append(PALETTE[i % len(PALETTE)] if not sim else PALETTE[i % len(PALETTE)])

    bars = ax.barh(techs, vals, color=colors, edgecolor="black", linewidth=0.5)
    for bar, sim in zip(bars, sim_flags):
        if sim:
            bar.set_alpha(SIMULATED_ALPHA)
            bar.set_hatch("//")
    ax.invert_yaxis()
    _style_axes(
        ax,
        title or f"Average {metric} across all configurations",
        "value",
        "",
    )
    ax.set_xlabel(metric)
    _save(fig, Path(path))