"""Build paper-style results tables from a list of BenchmarkResult."""

from __future__ import annotations

import csv
import io
import statistics
from typing import Dict, Iterable, List, Optional, Sequence

from ..techniques.base import BenchmarkResult


def build_results_table(
    results: Sequence[BenchmarkResult],
    *,
    baseline_technique: str = "Baseline FP16",
    metrics: Sequence[str] = (
        "peak_gpu_mb",
        "cpu_rss_mb",
        "analytic_kv_mb",
        "wall_time_s",
        "tokens_per_s",
        "time_to_first_token_s",
    ),
) -> List[Dict[str, object]]:
    """Return a list of dicts ready to be tabulated.

    Each row corresponds to one (technique, context_length, batch_size)
    cell. The ``memory_saving`` column expresses the *KV cache* memory
    saving versus the baseline at the same context length and batch
    size. The ``throughput_change`` column is the relative change in
    tokens/sec (positive = better).
    """
    if not results:
        return []

    # Group baseline by (ctx, bs) for delta computation.
    baseline_by_cell: Dict[tuple, BenchmarkResult] = {
        (r.context_length, r.batch_size): r for r in results if r.technique == baseline_technique
    }

    rows: List[Dict[str, object]] = []
    for r in results:
        ctx, bs = r.context_length, r.batch_size
        baseline = baseline_by_cell.get((ctx, bs))
        mem_saving = ""
        thr_change = ""
        if baseline is not None:
            if baseline.analytic_kv_mb > 0:
                mem_saving = (
                    100.0 * (1.0 - r.analytic_kv_mb / baseline.analytic_kv_mb)
                )
            if baseline.tokens_per_s > 0:
                thr_change = (
                    100.0
                    * (r.tokens_per_s - baseline.tokens_per_s)
                    / baseline.tokens_per_s
                )

        row: Dict[str, object] = {
            "Technique": r.technique,
            "Category": r.category,
            "Context": r.context_length,
            "Batch": r.batch_size,
            "Peak GPU MB": round(r.peak_gpu_mb, 1),
            "CPU RSS MB": round(r.cpu_rss_mb, 1),
            "KV Cache MB": round(r.analytic_kv_mb, 2),
            "Latency (s)": round(r.wall_time_s, 3),
            "Tokens/s": round(r.tokens_per_s, 2),
            "TTFT (s)": round(r.time_to_first_token_s, 3),
            "Memory Saving (%)": (
                "" if mem_saving == "" else round(float(mem_saving), 1)
            ),
            "Throughput Δ (%)": (
                "" if thr_change == "" else round(float(thr_change), 1)
            ),
            "Notes": r.notes,
        }
        rows.append(row)

    return rows


def table_to_markdown(
    rows: List[Dict[str, object]],
    columns: Optional[Sequence[str]] = None,
) -> str:
    if not rows:
        return "_(no results)_\n"

    columns = list(columns or list(rows[0].keys()))
    widths = [max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns]
    fmt = "| " + " | ".join("{:<%d}" % w for w in widths) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    out = [fmt.format(*columns), sep]
    for r in rows:
        out.append(fmt.format(*[str(r.get(c, "")) for c in columns]))
    return "\n".join(out) + "\n"


def table_to_csv(rows: List[Dict[str, object]]) -> str:
    if not rows:
        return ""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def aggregate_per_technique(
    results: Sequence[BenchmarkResult],
) -> List[Dict[str, object]]:
    """Return one row per technique with statistics averaged across
    (context_length, batch_size, trials).
    """
    by_tech: Dict[str, List[BenchmarkResult]] = {}
    for r in results:
        by_tech.setdefault(r.technique, []).append(r)

    rows: List[Dict[str, object]] = []
    for tech, rs in by_tech.items():
        def avg(field: str) -> float:
            vals = [float(getattr(r, field)) for r in rs if getattr(r, field) is not None]
            return statistics.mean(vals) if vals else 0.0

        rows.append(
            {
                "Technique": tech,
                "Category": rs[0].category,
                "Avg Peak GPU MB": round(avg("peak_gpu_mb"), 1),
                "Avg CPU RSS MB": round(avg("cpu_rss_mb"), 1),
                "Avg KV MB": round(avg("analytic_kv_mb"), 2),
                "Avg Latency (s)": round(avg("wall_time_s"), 3),
                "Avg Tokens/s": round(avg("tokens_per_s"), 2),
                "Avg TTFT (s)": round(avg("time_to_first_token_s"), 3),
                "Notes": rs[0].notes,
            }
        )
    return rows