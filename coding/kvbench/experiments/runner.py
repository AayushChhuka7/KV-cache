"""Experiment runner.

Orchestrates the full sweep:

  for each technique:
      setup()
      for each (context length, batch size):
          for each trial:
              run_trial(...)
      teardown()

and writes a JSON dump of all per-trial results plus a CSV summary.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ..config import BenchmarkConfig
from ..profiling.memory import snapshot_memory
from ..techniques.base import BenchmarkResult, KVTechnique
from ..utils.logging import get_logger, log_section
from ..utils.prompts import build_prompt
from ..utils.seeding import seed_everything


class ExperimentRunner:
    """High-level orchestrator."""

    def __init__(self, config: BenchmarkConfig, output_dir: str | Path):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("ExperimentRunner")
        self.results: List[BenchmarkResult] = []
        self.raw: List[Dict] = []
        self.technique_meta: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # Top-level entry points
    # ------------------------------------------------------------------
    def run_full_sweep(
        self,
        techniques: List[KVTechnique],
        batch_sizes: Optional[Iterable[int]] = None,
        context_lengths: Optional[Iterable[int]] = None,
    ) -> List[BenchmarkResult]:
        """Run the entire sweep and return aggregated results."""
        batch_sizes = list(batch_sizes or self.config.batch_sizes)
        context_lengths = list(context_lengths or self.config.context_lengths)

        for tech in techniques:
            self._run_technique(tech, batch_sizes, context_lengths)
            tech.teardown()
        return self.results

    def run_single(
        self,
        technique: KVTechnique,
        batch_size: int = 1,
        context_length: int = 256,
    ) -> BenchmarkResult:
        """Run a single (technique, batch, ctx) cell."""
        return self._run_cell(technique, batch_size, context_length)

    # ------------------------------------------------------------------
    # Per-technique orchestration
    # ------------------------------------------------------------------
    def _run_technique(
        self,
        technique: KVTechnique,
        batch_sizes: List[int],
        context_lengths: List[int],
    ) -> None:
        with log_section(f"Technique: {technique.spec.name}", self.logger):
            try:
                technique.setup()
            except Exception as e:  # noqa: BLE001
                self.logger.exception("setup() failed for %s: %s", technique.spec.name, e)
                self.technique_meta[technique.spec.name] = {"error": str(e)}
                return

            # Extend learned position embeddings (e.g. GPT-2's 1024-token
            # ``wpe``) to cover the sweep's longest context plus the decode
            # steps, so large-context cells measure the requested length
            # instead of truncating at the model's native limit.
            model = getattr(technique, "_model", None)
            tokenizer = getattr(technique, "_tokenizer", None)
            if model is not None:
                from ..utils.models import extend_position_embeddings

                need = max(context_lengths) + self.config.max_new_tokens + 8
                new_n = extend_position_embeddings(model, tokenizer, min_seq_len=need)
                if new_n > 0:
                    self.logger.info(
                        "Extended position embeddings to %d tokens for %s",
                        new_n, technique.spec.name,
                    )

            self.technique_meta[technique.spec.name] = {
                "category": technique.spec.category,
                "description": technique.spec.description,
                "is_simulation": getattr(technique, "is_simulation", False),
                "extra": technique.spec.extra,
            }

            for ctx in context_lengths:
                for bs in batch_sizes:
                    cell_results = self._run_cell_trials(
                        technique, ctx, bs, self.config.trials
                    )
                    if not cell_results:
                        # Every trial in this cell failed (e.g. CUDA OOM at
                        # large (ctx, batch) on small GPUs). Skip the cell
                        # rather than crashing the whole sweep; it is
                        # recorded as "not measured" in the log and is
                        # simply absent from the per-cell CSV.
                        self.logger.warning(
                            "  ctx=%-5d  bs=%-2d  NOT MEASURED (all %d trial(s) failed)",
                            ctx, bs, self.config.trials,
                        )
                        self.technique_meta.setdefault(
                            technique.spec.name, {}
                        ).setdefault("not_measured", []).append(
                            f"ctx={ctx},bs={bs}"
                        )
                        continue
                    avg = technique.average(cell_results)
                    self.results.append(avg)
                    self.raw.extend(r.to_dict() for r in cell_results)
                    self.logger.info(
                        "  ctx=%-5d  bs=%-2d  wall=%6.2fs  tok/s=%6.2f  GPU=%6.1f MB  CPU=%6.1f MB",
                        avg.context_length,
                        avg.batch_size,
                        avg.wall_time_s,
                        avg.tokens_per_s,
                        avg.peak_gpu_mb,
                        avg.cpu_rss_mb,
                    )

    def _run_cell_trials(
        self,
        technique: KVTechnique,
        context_length: int,
        batch_size: int,
        trials: int,
    ) -> List[BenchmarkResult]:
        # Build the prompts for this cell. The technique's
        # ``run_trial`` accepts a list of prompts — we always pass the
        # config.prompts, but scale each one to roughly ``context_length``
        # tokens via build_prompt.
        cell_prompts: List[str] = []
        if hasattr(technique, "_tokenizer") and technique._tokenizer is not None:
            for _ in self.config.prompts[: max(1, batch_size)]:
                cell_prompts.append(
                    build_prompt(
                        technique._tokenizer,
                        self.config.prompts,
                        target_tokens=context_length,
                        seed=self.config.seed,
                    )
                )
        else:
            cell_prompts = list(self.config.prompts[: max(1, batch_size)])

        out: List[BenchmarkResult] = []
        for trial_idx in range(trials):
            seed = self.config.seed + trial_idx
            seed_everything(seed, deterministic=False)
            t0 = time.perf_counter()
            try:
                result = technique.run_trial(
                    cell_prompts,
                    self.config.max_new_tokens,
                    batch_size,
                )
            except Exception as e:  # noqa: BLE001
                self.logger.exception(
                    "trial failed (%s, ctx=%d, bs=%d, trial=%d): %s",
                    technique.spec.name, context_length, batch_size, trial_idx, e,
                )
                continue
            t1 = time.perf_counter()
            result.raw.setdefault("wall_time_s", []).append(t1 - t0)
            out.append(result)
        return out

    def _run_cell(
        self,
        technique: KVTechnique,
        batch_size: int,
        context_length: int,
    ) -> BenchmarkResult:
        technique.setup()
        try:
            cells = self._run_cell_trials(technique, context_length, batch_size, self.config.trials)
        finally:
            technique.teardown()
        return technique.average(cells)

    # ------------------------------------------------------------------
    # Optional accuracy pass (perplexity)
    # ------------------------------------------------------------------
    def run_perplexity_eval(
        self,
        models: Optional[List[str]] = None,
        output_csv: Optional[str | Path] = None,
    ) -> List[Dict]:
        """Run the WikiText-2 perplexity pass for the real techniques.

        Invoked after the main memory/latency sweep when the caller opts in
        (see ``scripts/run_all.py --with-perplexity``). Re-seeds with the
        same seed used by the rest of the run so the accuracy numbers are
        reproducible alongside the paper's other measurements.
        """
        # Imported lazily so the package still imports on machines without
        # the datasets dependency when perplexity is not requested.
        from ..techniques.perplexity_eval import run_perplexity_evaluation

        seed_everything(self.config.seed)
        out = Path(output_csv) if output_csv else self.output_dir / "perplexity_results.csv"
        with log_section("Perplexity evaluation (WikiText-2 subset)", self.logger):
            rows = run_perplexity_evaluation(
                self.config,
                output_csv=out,
                models=models,
            )
        self.logger.info("Perplexity results written to %s", out)
        return rows

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, name: str = "results") -> List[Path]:
        out: List[Path] = []
        results_path = self.output_dir / f"{name}.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": self.config.to_dict(),
                    "technique_meta": self.technique_meta,
                    "results": [r.to_dict() for r in self.results],
                    "raw_trials": self.raw,
                },
                f,
                indent=2,
            )
        out.append(results_path)
        return out