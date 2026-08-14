"""Run the full sweep with per-technique checkpointing.

Saves results.json + results.csv + results_aggregated.csv after every
technique, so an interrupted run can be resumed with --skip-existing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kvbench.config import BenchmarkConfig
from kvbench.experiments.runner import ExperimentRunner
from kvbench.experiments.sweeps import build_default_techniques
from kvbench.reporting.markdown import write_results_markdown
from kvbench.techniques.base import BenchmarkResult


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/paper_sweep")
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--force", action="store_true", help="Re-run techniques even if present in results.json")
    args = p.parse_args()

    cfg = BenchmarkConfig()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = ExperimentRunner(cfg, out_dir)

    prior_results: list = []
    prior_meta: dict = {}
    json_candidates = [out_dir / "all.json", out_dir / "results.json"]
    existing_json = next((p for p in json_candidates if p.exists()), None)
    if existing_json is not None:
        with open(existing_json, encoding="utf-8") as f:
            data = json.load(f)
        prior_results = data.get("results", [])
        prior_meta = data.get("technique_meta", {})
        print(f"Existing {existing_json.name}: {len(prior_results)} rows")

    to_run = [t for t in build_default_techniques(cfg)]
    if args.only:
        to_run = [t for t in to_run if t.spec.name in args.only]
    done = {r["technique"] for r in prior_results}

    ran: set = set()
    for t in to_run:
        if t.spec.name in done and not args.force:
            print("SKIP", t.spec.name)
            continue
        ran.add(t.spec.name)
        t0 = time.perf_counter()
        runner.run_full_sweep([t])
        print(f"SAVED {t.spec.name} in {time.perf_counter() - t0:.1f}s")

    kept = [r for r in prior_results if r["technique"] not in ran]
    kept_meta = {k: v for k, v in prior_meta.items() if k not in ran}

    if ran or to_run:
        runner.results = [BenchmarkResult(**r) for r in kept] + runner.results
        runner.technique_meta = {**kept_meta, **runner.technique_meta}
        runner.save("all")
        write_results_markdown(runner.results, out_dir, config_dict=cfg.to_dict())
    else:
        print("Nothing to run")

    print("ALL DONE")


if __name__ == "__main__":
    main()
