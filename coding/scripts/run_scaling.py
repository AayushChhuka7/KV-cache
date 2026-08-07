"""Sweep context length and batch size for the baseline + selected techniques."""

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
    plot_memory_vs_batch,
)
from kvbench.techniques.baseline import BaselineKV
from kvbench.techniques.gqa import GQAModel
from kvbench.techniques.quantized import Int8SimulatedKV
from kvbench.utils.logging import log_section
from kvbench.utils.seeding import seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/scaling")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--ctx", type=int, nargs="+", default=[128, 512, 1024, 2048, 4096]
    )
    parser.add_argument("--bs", type=int, nargs="+", default=[1, 2, 4, 8])
    args = parser.parse_args()

    cfg = BenchmarkConfig()
    cfg.trials = args.trials
    if args.model:
        cfg.model = args.model
    cfg.context_lengths = args.ctx
    cfg.batch_sizes = args.bs

    seed_everything(cfg.seed)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = ExperimentRunner(cfg, out_dir)
    techs = [BaselineKV(cfg), Int8SimulatedKV(cfg), GQAModel(cfg)]
    with log_section("Scaling sweep (context × batch)"):
        runner.run_full_sweep(
            techs,
            batch_sizes=cfg.batch_sizes,
            context_lengths=cfg.context_lengths,
        )

    runner.save("scaling")
    write_results_markdown(runner.results, out_dir, config_dict=cfg.to_dict())

    if cfg.generate_plots:
        for bs in cfg.batch_sizes:
            plot_memory_vs_context(runner.results, out_dir / f"memory_vs_context_bs{bs}.png", batch_size=bs)
            plot_throughput_vs_context(runner.results, out_dir / f"throughput_vs_context_bs{bs}.png", batch_size=bs)
            plot_latency_vs_context(runner.results, out_dir / f"latency_vs_context_bs{bs}.png", batch_size=bs)
        for ctx in cfg.context_lengths:
            plot_memory_vs_batch(runner.results, out_dir / f"memory_vs_batch_ctx{ctx}.png", context_length=ctx)

    print("Done. See", out_dir / "report.md")


if __name__ == "__main__":
    main()