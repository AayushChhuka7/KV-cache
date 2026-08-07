"""Run the analytical simulators (NVFP4, MiniKV, xKV) and merge with
the baseline's measured numbers to produce a comparable table.

Because the simulators are analytic, no model is loaded — but we do
still load the *config* of the requested baseline model so that the
analytic KV cache sizes are correct.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kvbench.config import BenchmarkConfig
from kvbench.experiments.runner import ExperimentRunner
from kvbench.reporting.markdown import write_results_markdown
from kvbench.reporting.plots import plot_summary_bar
from kvbench.techniques.baseline import BaselineKV
from kvbench.techniques.simulations import (
    MiniKVSimulator,
    NVFP4Simulator,
    XKVSimulator,
)
from kvbench.utils.logging import log_section
from kvbench.utils.seeding import seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/simulations")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    cfg = BenchmarkConfig()
    cfg.trials = args.trials
    if args.model:
        cfg.model = args.model
    cfg.run_simulations = True

    seed_everything(cfg.seed)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = ExperimentRunner(cfg, out_dir)
    techs = [
        BaselineKV(cfg),
        NVFP4Simulator(cfg),
        MiniKVSimulator(cfg),
        XKVSimulator(cfg),
    ]
    with log_section("Simulated techniques (NVFP4, MiniKV, xKV) vs baseline"):
        runner.run_full_sweep(techs)

    runner.save("simulations")
    write_results_markdown(runner.results, out_dir, config_dict=cfg.to_dict())

    if cfg.generate_plots:
        plot_summary_bar(runner.results, out_dir / "summary_kv_mb.png", metric="analytic_kv_mb")

    print("Done. See", out_dir / "report.md")


if __name__ == "__main__":
    main()