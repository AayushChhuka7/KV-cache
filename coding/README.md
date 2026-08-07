# kvbench — KV Cache Optimization Benchmark

A research-grade benchmarking framework for KV cache optimization
techniques in large language models.

`kvbench` compares **nine** KV cache techniques under identical
conditions: identical prompts, identical seeds, identical hardware, and
averaged across multiple trials. Each technique reports the same set of
metrics, so the resulting tables and plots are directly comparable.

This package was built as the experimental section for the paper
*"KV Cache Optimization in LLMs: A Taxonomy and Decision Framework for
Efficient Inference"*. The repository's other folders (the LaTeX
manuscript, references, and figures) are untouched by this framework —
**all code lives inside `coding/`**.

---

## Techniques implemented

| # | Technique              | Category                | Type             | What is measured                        |
|---|------------------------|-------------------------|------------------|-----------------------------------------|
| 1 | Baseline FP16          | baseline                | real             | standard HF `generate()` with KV cache  |
| 2 | Quantized INT8         | quantization            | real             | per-channel symmetric int8 cache        |
| 3 | Quantized FP8 (E5M2)   | quantization            | real (sim)       | 1-byte fp8 cache via PyTorch dtype       |
| 4 | GQA                    | attention-level         | real             | native HF model with grouped KV heads   |
| 5 | Low-Rank Compression   | compression             | real             | low-rank projection, reconstruct on fly  |
| 6 | PagedAttention (vLLM)  | memory management       | real + fallback  | vLLM PagedAttention or pure-Py sim      |
| 7 | CPU Offloading         | memory management       | real             | KV split: GPU window + pinned CPU       |
| 8 | NVFP4 (simulated)      | quantization            | analytical sim   | 4-bit fp cache formula                   |
| 9 | MiniKV (simulated)     | quantization + eviction | analytical sim   | 2-bit + adaptive eviction formula        |
| 10| xKV (simulated)        | compression             | analytical sim   | cross-layer SVD cache formula            |

The three *simulated* techniques (NVFP4, MiniKV, xKV) require
proprietary kernels or pre-processing steps that are not pip-installable.
Rather than skip them, `kvbench` includes high-fidelity analytical
simulators whose memory estimates come from the equations in Section 3
of each paper and whose latency overheads are derived from the published
numbers. This is documented in `LIMITATIONS.md` and in the per-paper
comments inside `kvbench/techniques/simulations.py`.

---

## Metrics reported

Every trial emits a `BenchmarkResult` containing:

| Metric                | Meaning                                                  |
|-----------------------|----------------------------------------------------------|
| `peak_gpu_mb`         | peak GPU memory allocated by PyTorch during the trial    |
| `gpu_reserved_mb`     | peak reserved memory (caching allocator)                 |
| `cpu_rss_mb`          | peak resident-set size of the Python process             |
| `analytic_kv_mb`      | KV cache size derived from the model's shape             |
| `wall_time_s`         | total wall-clock time for the trial                      |
| `tokens_per_s`        | tokens generated per second                              |
| `time_to_first_token_s` | approximate TTFT                                        |
| `avg_gpu_util_pct`    | mean NVML GPU utilization during the trial               |
| `peak_rss_mb`         | peak CPU RSS during the trial                            |
| `notes`               | per-technique description and any caveats                |

Across multiple trials the runner averages these fields, producing the
research-paper tables in the report.

---

## Folder layout

```
coding/
├── README.md                  (this file)
├── ARCHITECTURE.md            (system design and module responsibilities)
├── LIMITATIONS.md             (what is simulated, what is not, and why)
├── EXPERIMENTS.md             (concrete commands and expected outputs)
├── requirements.txt
├── pyproject.toml
├── install.sh / install.bat   (cross-platform installer)
├── run_all.sh / run_all.bat   (run the full sweep)
├── scripts/                   (CLI entry points)
│   ├── run_baseline.py
│   ├── run_quantization.py
│   ├── run_gqa.py
│   ├── run_paged_attention.py
│   ├── run_offloading.py
│   ├── run_compression.py
│   ├── run_simulations.py
│   ├── run_scaling.py
│   └── run_all.py
├── kvbench/
│   ├── config.py              (central BenchmarkConfig)
│   ├── utils/                 (seeding, logging, prompts, model loader)
│   ├── profiling/             (memory, latency, resource probes)
│   ├── techniques/            (the ten techniques + base interface)
│   ├── experiments/           (runner + sweep builders)
│   └── reporting/             (tables, plots, markdown)
├── tests/                     (pytest unit tests + smoke test)
└── results/                   (output directory, created on first run)
```

---

## Installation

### Linux / macOS / WSL

```bash
cd coding
bash install.sh
```

This script:
  1. creates a `.venv`,
  2. installs the base requirements,
  3. installs PyTorch (CUUDNN build if `nvidia-smi` is present, else CPU),
  4. installs `transformers`, `accelerate`, `sentencepiece`,
  5. attempts to install `vllm` and `bitsandbytes` (Linux only),
  6. installs `kvbench` itself in editable mode.

### Windows

```bat
cd coding
install.bat
```

vLLM and bitsandbytes are not supported on native Windows; the
PagedAttention technique falls back to a CPU simulation. For full
PagedAttention support, run the framework inside WSL2 or on a Linux
machine.

### Manual installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install torch                  # or the CUDA build for your driver
pip install transformers accelerate sentencepiece tqdm psutil pyyaml tabulate matplotlib pandas seaborn
pip install vllm bitsandbytes      # Linux only, optional
pip install -e .
```

---

## Running the benchmarks

### The full sweep

```bash
bash run_all.sh                       # Linux / WSL
run_all.bat                           # Windows

# or directly:
python scripts/run_all.py --output results/all
```

This runs every technique across every (context length, batch size)
combination and writes:

```
results/all/
  report.md
  results.json          # full data + per-trial measurements
  results.csv           # per-cell table
  results_aggregated.csv
  memory_vs_context.png
  throughput_vs_context.png
  latency_vs_context.png
  memory_vs_batch.png
  summary_tokens_per_s.png
  summary_kv_mb.png
  summary_peak_gpu_mb.png
```

### Individual experiments

```bash
python scripts/run_baseline.py           --output results/baseline      --trials 3
python scripts/run_quantization.py       --output results/quantization  --trials 3
python scripts/run_gqa.py                --output results/gqa           --trials 3 --model-gqa TinyLlama/TinyLlama-1.1B-Chat-v1.0
python scripts/run_paged_attention.py    --output results/paged         --trials 3
python scripts/run_offloading.py         --output results/offloading    --trials 3 --windows 128 256 1024
python scripts/run_compression.py        --output results/compression   --trials 3 --ranks 32 64 128
python scripts/run_simulations.py        --output results/simulations   --trials 1
python scripts/run_scaling.py            --output results/scaling       --trials 2 --ctx 128 512 2048 4096 --bs 1 2 4 8
```

All scripts accept `--trials`, `--model`, `--ctx`, and `--bs` overrides.

### Environment variables

| Variable                | Default                              | Description                                 |
|-------------------------|--------------------------------------|---------------------------------------------|
| `KVBENCH_MODEL`         | `sshleifer/tiny-gpt2`                | baseline model id                           |
| `KVBENCH_GQA_MODEL`     | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | GQA model id                                |
| `KVBENCH_MHA_MODEL`     | `gpt2`                               | MHA model id                                |
| `KVBENCH_TRIALS`        | `3`                                  | default number of trials                    |
| `KVBENCH_WARMUP`        | `2`                                  | number of warmup steps                      |
| `KVBENCH_MAX_NEW`       | `32`                                 | tokens generated per trial                  |
| `KVBENCH_LOG_LEVEL`     | `INFO`                               | logging verbosity                           |
| `HF_HOME`               | (none)                               | HuggingFace cache directory                 |

---

## Reproducibility

* The runner seeds `random`, `numpy`, `torch`, and (when available)
  CUDA before every trial.
* Trials use `seed + trial_idx`, so re-running with `--trials N`
  produces the first `N` trials of a longer run.
* All models are loaded with `low_cpu_mem_usage=True` and a deterministic
  dtype; the PyTorch caching allocator is reset before each trial.
* `report.md` and `results.csv` are deterministic given the same
  hardware, dependencies, and arguments.

---

## Expected outputs

A single trial yields a `BenchmarkResult` with the metrics listed above.
The `results.csv` produced by `run_all.py` looks like:

```
Technique,Category,Context,Batch,Peak GPU MB,CPU RSS MB,KV Cache MB,Latency (s),Tokens/s,TTFT (s),Memory Saving (%),Throughput Δ (%),Notes
Baseline FP16,baseline,128,1,...,...,...,...,...,...,...,...,...
Quantized INT8,quantization,128,1,...,...,...,...,...,...,...,...,...
...
NVFP4 (simulated),quantization,128,1,...,...,...,...,...,...,...,...,...
```

and `report.md` contains both a per-technique summary table (averaged
across all configurations) and the full per-cell table.

---

## Tests

```bash
pip install pytest
pytest tests/ -v
```

The test suite contains 13 tests covering seeding, KV-shape math,
quantization round-trips, and table generation. They run on CPU and
require no model downloads. A separate end-to-end smoke test is at
`tests/smoke_test.py`.

---

## Hardware

| Tier        | What works                                                |
|-------------|-----------------------------------------------------------|
| CPU only    | Baseline (very small model), GQA, Quantized, LowRank, Offloading, all simulators; PagedAttention falls back to the CPU simulator |
| 1× GPU ≥ 8GB | Everything above, plus real vLLM PagedAttention             |
| 1× GPU ≥ 24GB| Llama-3.x and Mistral GQA models at 8k context              |

NVFP4, MiniKV, and xKV are simulated regardless of hardware; the
simulation models their *memory savings* and *latency overheads* using
the equations from the original papers, not custom GPU kernels.

---

## Citation

If you use this framework in a paper, please cite the underlying
technique papers and (optionally) this framework as a software
artifact. See `reference.bib` in the repository root for the full
bibliography.

---

## License

MIT.
