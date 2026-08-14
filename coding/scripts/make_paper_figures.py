"""Generate publication-quality figures for the paper from REAL measurements.

Reads the per-cell CSV produced by a full kvbench sweep
(``coding/results/paper_sweep/results.csv`` by default) and renders the
six plots used by the paper (Figures 2-4) with the kvbench reporting
module. Nothing is fabricated here: every line is a measured (technique,
context, batch) cell.

Panel-specific honesty rules:
  * Peak-GPU-memory, throughput, and latency panels include only the
    techniques that executed a real model on hardware. The analytical
    simulators (NVFP4 / MiniKV / xKV) allocate nothing and report no
    latency or throughput, so a zero line would be misleading.
  * The KV-cache-size summary includes the simulators: their analytic
    cache size is their genuine, documented metric.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure we can import the kvbench package.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kvbench.techniques.base import BenchmarkResult
from kvbench.reporting.plots import (
    plot_memory_vs_context,
    plot_throughput_vs_context,
    plot_latency_vs_context,
    plot_memory_vs_batch,
    plot_summary_bar,
)

# Canonical technique order (matches the paper's Section IV listing).
TECHNIQUE_ORDER = [
    "Baseline FP16",
    "Quantized INT8",
    "Quantized FP8 (E5M2)",
    "GQA",
    "Low-Rank Compression",
    "PagedAttention (vLLM)",
    "CPU Offloading",
    "NVFP4 (simulated)",
    "MiniKV (simulated)",
    "xKV (simulated)",
]

REAL_ONLY = {t for t in TECHNIQUE_ORDER if "(simulated)" not in t}

CATEGORY = {
    "Baseline FP16": "baseline",
    "Quantized INT8": "quantization",
    "Quantized FP8 (E5M2)": "quantization",
    "GQA": "attention",
    "Low-Rank Compression": "compression",
    "PagedAttention (vLLM)": "memory_management",
    "CPU Offloading": "memory_management",
    "NVFP4 (simulated)": "quantization",
    "MiniKV (simulated)": "quantization_eviction",
    "xKV (simulated)": "compression",
}


def load_results(csv_path: Path) -> list[BenchmarkResult]:
    """Load the per-cell sweep CSV into BenchmarkResult objects."""
    import csv as _csv

    rows: list[BenchmarkResult] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            tech = row["Technique"]
            rows.append(
                BenchmarkResult(
                    technique=tech,
                    category=row.get("Category", CATEGORY.get(tech, "")),
                    context_length=int(row["Context"]),
                    batch_size=int(row["Batch"]),
                    peak_gpu_mb=float(row["Peak GPU MB"]),
                    cpu_rss_mb=float(row["CPU RSS MB"]),
                    analytic_kv_mb=float(row["KV Cache MB"]),
                    wall_time_s=float(row["Latency (s)"]),
                    tokens_generated=32,
                    tokens_per_s=float(row["Tokens/s"]),
                    time_to_first_token_s=float(row["TTFT (s)"]),
                    notes=row.get("Notes", ""),
                )
            )

    order = {t: i for i, t in enumerate(TECHNIQUE_ORDER)}
    rows.sort(key=lambda r: (order.get(r.technique, 99), r.context_length, r.batch_size))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default=ROOT / "results" / "paper_sweep" / "results.csv",
        help="Per-cell CSV from a full sweep (default: coding/results/paper_sweep/results.csv)",
    )
    ap.add_argument("--out", default=None, help="Output directory (default: project root, next to main.tex)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out) if args.out else Path(__file__).resolve().parent.parent.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = load_results(csv_path)
    real_rows = [r for r in all_rows if r.technique in REAL_ONLY]

    print(f"Loaded {len(all_rows)} cells from {csv_path} "
          f"({len(real_rows)} real, {len(all_rows) - len(real_rows)} simulated)")

    plot_memory_vs_context(real_rows, out_dir / "memory_vs_context.png", batch_size=1)
    plot_throughput_vs_context(real_rows, out_dir / "throughput_vs_context.png", batch_size=1)
    plot_latency_vs_context(real_rows, out_dir / "latency_vs_context.png", batch_size=1)
    plot_memory_vs_batch(real_rows, out_dir / "memory_vs_batch.png", context_length=512)
    plot_summary_bar(real_rows, out_dir / "summary_tokens_per_s.png", metric="tokens_per_s",
                     title="Average tokens per second across all configurations")
    plot_summary_bar(all_rows, out_dir / "summary_kv_mb.png", metric="analytic_kv_mb",
                     title="Average analytical KV cache size (MB) across all configurations",
                     log_scale=True)

    print(f"Wrote 6 figures to: {out_dir}")


if __name__ == "__main__":
    main()
