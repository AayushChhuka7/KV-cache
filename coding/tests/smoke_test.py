"""End-to-end smoke test of the experiment runner.

Runs only the analytical simulators (NVFP4, MiniKV, xKV) and the
baseline's KV-shape estimation, then writes the report. No model
weights are downloaded.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kvbench.config import BenchmarkConfig
from kvbench.experiments.runner import ExperimentRunner
from kvbench.reporting.markdown import write_results_markdown
from kvbench.techniques.simulations import (
    MiniKVSimulator,
    NVFP4Simulator,
    XKVSimulator,
)
from kvbench.utils.seeding import seed_everything


def main():
    cfg = BenchmarkConfig(trials=1)
    cfg.context_lengths = [256, 1024]
    cfg.batch_sizes = [1, 2]
    cfg.run_simulations = True

    seed_everything(cfg.seed)
    out_dir = Path("results/_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = ExperimentRunner(cfg, out_dir)
    techs = [NVFP4Simulator(cfg), MiniKVSimulator(cfg), XKVSimulator(cfg)]

    runner.run_full_sweep(techs)
    runner.save("smoke")
    write_results_markdown(runner.results, out_dir, config_dict=cfg.to_dict())
    print("Smoke OK; see", out_dir / "report.md")


if __name__ == "__main__":
    main()