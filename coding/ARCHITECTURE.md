# Architecture

This document explains the internal design of `kvbench`. It is meant
for contributors who want to add a new technique, a new metric, or
a new reporting format.

## Module map

```
kvbench/
├── config.py                   # BenchmarkConfig dataclass — single source of truth
├── utils/                      # stateless helpers
│   ├── seeding.py              # seed_everything — random/numpy/torch
│   ├── logging.py              # get_logger + log_section (banner context manager)
│   ├── prompts.py              # deterministic prompt builders
│   └── models.py               # model loader, KVShape, dtype mapping
├── profiling/                  # measurement primitives
│   ├── memory.py               # MemoryTracker + snapshot_memory + analytic KV size
│   ├── latency.py              # wall-clock / CUDA-event timer
│   └── resources.py            # background CPU/NVML probe
├── techniques/                 # one file per technique
│   ├── base.py                 # KVTechnique + BenchmarkResult + TechniqueSpec
│   ├── baseline.py             # BaselineKV
│   ├── quantized.py            # Int8SimulatedKV, FP8SimulatedKV
│   ├── gqa.py                  # GQAModel
│   ├── compression.py          # LowRankCompression
│   ├── paged.py                # PagedAttentionVLLM (vLLM + CPU fallback)
│   ├── offloading.py           # CPUOffloading
│   └── simulations.py          # NVFP4Simulator, MiniKVSimulator, XKVSimulator
├── experiments/
│   ├── runner.py               # ExperimentRunner — top-level orchestrator
│   └── sweeps.py               # default technique + sweep factories
└── reporting/
    ├── tables.py               # build_results_table / aggregate_per_technique
    ├── plots.py                # matplotlib charts (palette + simulated markers)
    └── markdown.py             # write_results_markdown — produces report.md
```

## Data flow

```
                  ┌────────────────────────────────┐
                  │  scripts/run_*.py (CLI entry)  │
                  └────────────────┬───────────────┘
                                   │ builds BenchmarkConfig
                                   ▼
                  ┌────────────────────────────────┐
                  │     ExperimentRunner           │
                  │  for tech in techniques:       │
                  │    setup()                     │
                  │    for ctx in ctx_lengths:     │
                  │      for bs in batch_sizes:    │
                  │        for trial in trials:    │
                  │          run_trial(prompts)    │
                  │          seed_everything(...)  │
                  └────────────────┬───────────────┘
                                   │
                                   ▼
                  ┌────────────────────────────────┐
                  │     KVTechnique.run_trial      │
                  │  (each technique implements)  │
                  └────────────────┬───────────────┘
                                   │ returns BenchmarkResult
                                   ▼
                  ┌────────────────────────────────┐
                  │  Reporting (tables, plots,     │
                  │  markdown report)              │
                  └────────────────────────────────┘
```

## Adding a new technique

1. Subclass `KVTechnique` in a new file under `kvbench/techniques/`.
2. Set `spec = TechniqueSpec(name=..., category=..., description=...)`.
3. Implement:
   * `setup(self)` — load the model / state.
   * `run_trial(self, prompts, max_new_tokens, batch_size)` — return a `BenchmarkResult`.
   * `analytic_kv_mb(self, batch, seq_len)` — return the KV cache size in MB.
4. (Optional) Override `teardown()` for cleanup.
5. Add your technique to `build_default_techniques()` in
   `kvbench/experiments/sweeps.py`.

## Adding a new metric

1. Add a field to `BenchmarkResult` in `kvbench/techniques/base.py`.
2. Add it to the `numeric_fields` list inside `average()`.
3. Capture it inside `_measure_trial()` (or directly in your
   technique's `run_trial`).
4. Optionally include it in the table-builder columns in
   `kvbench/reporting/tables.py` and in the markdown writer in
   `kvbench/reporting/markdown.py`.

## Seeding

`seed_everything(seed, deterministic=False)` is called inside
`_run_cell_trials` for each trial. It seeds:

* Python `random`
* `numpy.random`
* `torch.manual_seed`
* `torch.cuda.manual_seed_all` (when CUDA is available)

When `deterministic=True`, it additionally enables cuDNN's deterministic
algorithms. This is currently used only by tests.

## Profiling

* `snapshot_memory()` returns a `MemoryReading` with peak GPU memory,
  GPU reserved memory, and CPU RSS. It does **not** start a tracker; it
  just reads counters.
* `MemoryTracker` is a context manager that starts a background thread
  sampling `snapshot_memory` every `interval_s` seconds. Peak values
  are exposed as properties (`peak_gpu_mb`, `peak_cpu_rss_mb`).
* `GenerationTimer` wraps a CUDA event or `time.perf_counter` to record
  wall time and (optionally) per-phase timing.
* `ResourceProbe` is the multi-purpose CPU+GPU sampler used by the
  runner.

## Reporting

`write_results_markdown(results, output_dir, config_dict=...)`:

1. Builds the per-technique summary via `aggregate_per_technique`.
2. Builds the per-cell table via `build_results_table`, which also
   computes the relative `Memory Saving (%)` and `Throughput Δ (%)`
   versus the baseline at the same (ctx, bs).
3. Writes `report.md`, `results.csv`, and `results_aggregated.csv`.

`plot_*.py` use a single shared palette and a different marker for
simulated techniques so the figures are honest at a glance.

## Design choices

* **Pure-PyTorch fallbacks** — every technique that requires a custom
  kernel (vLLM, BitsAndBytes) has a CPU-only simulation that captures
  the *qualitative* behaviour described in the original papers. This
  keeps the framework runnable on consumer hardware and inside CI.
* **Factories in sweeps** — techniques are constructed once per cell
  to avoid leaking HuggingFace KV-cache state between trials.
* **Single config source** — `BenchmarkConfig` is the only mutable
  knob. Everything else (CLI args, env vars, YAML) feeds into it.
