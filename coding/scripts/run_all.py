"""Run all benchmark experiments and write a unified report.

This is the entry point used by ``run_all.sh`` / ``run_all.bat`` and by
the README. It produces:

  results/<timestamp>/
      report.md          # human-readable summary
      results.json       # raw data + per-trial measurements
      results.csv        # per-cell table
      results_aggregated.csv
      *.png              # plots
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kvbench.config import BenchmarkConfig
from kvbench.experiments.runner import ExperimentRunner
from kvbench.experiments.sweeps import build_default_techniques
from kvbench.reporting.markdown import write_results_markdown
from kvbench.reporting.plots import (
    plot_memory_vs_batch,
    plot_memory_vs_context,
    plot_throughput_vs_context,
    plot_latency_vs_context,
    plot_summary_bar,
)
from kvbench.utils.logging import get_logger, log_section
from kvbench.utils.seeding import seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/all")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-simulations", action="store_true")
    parser.add_argument("--ctx", type=int, nargs="+", default=None)
    parser.add_argument("--bs", type=int, nargs="+", default=None)
    parser.add_argument("--model", default=None, help="Override the baseline model id.")
    parser.add_argument("--model-mha", default=None)
    parser.add_argument("--model-gqa", default=None)
    args = parser.parse_args()

    cfg = BenchmarkConfig()
    if args.trials:
        cfg.trials = args.trials
    if args.no_simulations:
        cfg.run_simulations = False
    if args.ctx:
        cfg.context_lengths = args.ctx
    if args.bs:
        cfg.batch_sizes = args.bs
    if args.model:
        cfg.model = args.model
    if args.model_mha:
        cfg.model = args.model_mha
    if args.model_gqa:
        cfg.gqa_model = args.model_gqa

    seed_everything(cfg.seed)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = ExperimentRunner(cfg, out_dir)
    techniques = build_default_techniques(cfg)

    t0 = time.perf_counter()
    with log_section(f"Running {len(techniques)} techniques"):
        runner.run_full_sweep(techniques)
    t1 = time.perf_counter()

    logger = get_logger()
    logger.info("Total wall time: %.1fs", t1 - t0)
    runner.save("all")
    write_results_markdown(runner.results, out_dir, config_dict=cfg.to_dict())

    if not args.no_plots and cfg.generate_plots:
        try:
            plot_memory_vs_context(runner.results, out_dir / "memory_vs_context.png")
            plot_throughput_vs_context(runner.results, out_dir / "throughput_vs_context.png")
            plot_latency_vs_context(runner.results, out_dir / "latency_vs_context.png")
            plot_memory_vs_batch(runner.results, out_dir / "memory_vs_batch.png")
            plot_summary_bar(runner.results, out_dir / "summary_tokens_per_s.png", metric="tokens_per_s")
            plot_summary_bar(runner.results, out_dir / "summary_kv_mb.png", metric="analytic_kv_mb")
            plot_summary_bar(runner.results, out_dir / "summary_peak_gpu_mb.png", metric="peak_gpu_mb")
        except Exception as e:  # noqa: BLE001
            logger.exception("Plot generation failed: %s", e)

    print("Done. See", out_dir / "report.md")


if __name__ == "__main__":
    main()