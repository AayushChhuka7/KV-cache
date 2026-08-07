"""Run CPU offloading experiment with multiple GPU windows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
from kvbench.techniques.offloading import CPUOffloading
from kvbench.utils.logging import log_section
from kvbench.utils.seeding import seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/offloading")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--model", default=None)
    parser.add_argument("--windows", type=int, nargs="+", default=[128, 256, 1024])
    args = parser.parse_args()

    cfg = BenchmarkConfig()
    cfg.trials = args.trials
    if args.model:
        cfg.model = args.model

    seed_everything(cfg.seed)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = ExperimentRunner(cfg, out_dir)
    techs = [BaselineKV(cfg)] + [CPUOffloading(cfg, gpu_window=w) for w in args.windows]
    with log_section("CPU offloading experiment"):
        runner.run_full_sweep(techs)

    runner.save("offloading")
    write_results_markdown(runner.results, out_dir, config_dict=cfg.to_dict())

    if cfg.generate_plots:
        plot_memory_vs_context(runner.results, out_dir / "memory_vs_context.png")
        plot_throughput_vs_context(runner.results, out_dir / "throughput_vs_context.png")
        plot_latency_vs_context(runner.results, out_dir / "latency_vs_context.png")
        plot_summary_bar(runner.results, out_dir / "summary_tokens_per_s.png")

    print("Done. See", out_dir / "report.md")


if __name__ == "__main__":
    main()