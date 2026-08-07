"""Run only the baseline benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make package importable when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kvbench.config import BenchmarkConfig
from kvbench.experiments.runner import ExperimentRunner
from kvbench.reporting.markdown import write_results_markdown
from kvbench.reporting.plots import (
    plot_memory_vs_context,
    plot_throughput_vs_context,
    plot_latency_vs_context,
    plot_summary_bar,
)
from kvbench.techniques.baseline import BaselineKV
from kvbench.utils.logging import log_section, get_logger
from kvbench.utils.seeding import seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/baseline")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--model", default=None)
    parser.add_argument("--ctx", type=int, nargs="+", default=None)
    parser.add_argument("--bs", type=int, nargs="+", default=None)
    args = parser.parse_args()

    cfg = BenchmarkConfig()
    cfg.trials = args.trials
    if args.model:
        cfg.model = args.model
    if args.ctx:
        cfg.context_lengths = args.ctx
    if args.bs:
        cfg.batch_sizes = args.bs

    seed_everything(cfg.seed)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = ExperimentRunner(cfg, out_dir)
    baseline = BaselineKV(cfg)
    with log_section("Baseline benchmark"):
        runner.run_full_sweep([baseline])

    runner.save("baseline")
    write_results_markdown(runner.results, out_dir, config_dict=cfg.to_dict())

    if cfg.generate_plots:
        plot_memory_vs_context(runner.results, out_dir / "memory_vs_context.png")
        plot_throughput_vs_context(runner.results, out_dir / "throughput_vs_context.png")
        plot_latency_vs_context(runner.results, out_dir / "latency_vs_context.png")
        plot_summary_bar(runner.results, out_dir / "summary_tokens_per_s.png")

    print("Done. See", out_dir / "report.md")


if __name__ == "__main__":
    main()