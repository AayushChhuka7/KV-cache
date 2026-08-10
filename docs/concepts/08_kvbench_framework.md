# 08 — `kvbench`: The Benchmarking Framework

## Why we built it

We needed a **single tool** that:

1. Runs every one of the 10 techniques under **identical conditions**.
2. Measures **the same set of metrics** for every technique.
3. Produces **reproducible** numbers (same seeds, same prompts, same
   hardware).
4. Works on **commodity hardware** (no H100 cluster required).
5. Distinguishes **real** measurement from **analytical simulation**
   in the output, so the reader is never misled.

So we wrote `kvbench` in Python on top of PyTorch and HuggingFace.

## Stack

- **Python 3.10+**
- **PyTorch** (CUDA if available, CPU fallback)
- **HuggingFace `transformers`** for model loading
- **`accelerate`** for offloading
- **`vllm`** (Linux only) for real PagedAttention
- **`bitsandbytes`** (Linux only) for INT8
- **`matplotlib`** for publication-quality plots
- **`pandas`** for CSV aggregation

## Architecture

```
kvbench/
├── techniques/
│   ├── base.py            # KVTechnique abstract interface
│   ├── baseline_fp16.py   # standard HF generate()
│   ├── int8.py            # per-channel symmetric int8
│   ├── fp8.py             # torch.float8_e5m2 if available
│   ├── gqa.py             # GQA model loader
│   ├── low_rank.py        # random projection
│   ├── paged.py           # vLLM or pure-PyTorch simulation
│   ├── offload.py         # GPU-window + pinned CPU
│   ├── nvfp4.py           # analytical sim
│   ├── minikv.py          # analytical sim
│   └── xkv.py             # analytical sim
├── experiments/
│   └── runner.py          # sweep over (technique, ctx, batch)
├── reporting/
│   └── plots.py           # publication-quality figures
└── metrics/
    └── collect.py         # peak GPU MB, CPU RSS, timing, tokens/s
```

## The `KVTechnique` interface

Every technique is a subclass of `KVTechnique` and provides three
methods:

```python
class KVTechnique:
    def setup(self):
        """Load model, tokenizer, or just config (for simulators)."""

    def analytic_kv_mb(self, batch: int, seq_len: int) -> float:
        """Return KV cache size in MB computed from Eq. 1 + params."""

    def run_trial(self, prompts, max_new_tokens, batch_size) -> BenchmarkResult:
        """Run one trial and return all measured numbers."""
```

This uniform interface means the runner can sweep techniques without
knowing their internals.

## The runner

`runner.py` iterates over the Cartesian product:

```
for technique in techniques:
    for ctx in [128, 512, 2048, 4096]:
        for batch in [1, 2, 4]:
            for trial in range(num_trials):
                seed_everything(trial)
                result = technique.run_trial(prompts, ctx, batch)
                write_csv(...)
```

Before every trial, we seed Python `random`, NumPy, and PyTorch. This
is essential for reproducibility — without it, the same technique
gives different numbers on different runs.

## How each technique is implemented

### Real techniques (we run them on the GPU/CPU)

- **Baseline FP16** — `model.generate()` with the HF cache. The
  default.
- **INT8** — `_replace_with_quant_layers_` style: wrap K/V
  projections with a symmetric per-channel quantizer-dequantizer.
- **FP8 (E5M2)** — store K/V as `torch.float8_e5m2` if the GPU
  supports it; otherwise as 1-byte approximation.
- **GQA** — load `TinyLlama-1.1B-Chat-v1.0` (32 query heads, 4 KV
  heads).
- **Low-rank** — wrap K/V with a `nn.Linear(D, r)` projection on
  write and `nn.Linear(r, D)` on read. Random Kaiming-uniform init.
- **PagedAttention** — try `vllm.LLM` first; fall back to a
  pure-PyTorch page-table simulation if vLLM is unavailable.
- **Offloading** — keep GPU window of size $W$, spill older slices
  to pinned CPU memory via `torch.cuda.HostAllocator` (or the CPU
  equivalent).

### Simulated techniques (we cannot run them)

- **NVFP4** — `analytic_kv_mb = analytic_kv_mb_fp16 / 4`.
- **MiniKV** — `analytic_kv_mb = 0.2 * analytic_kv_mb_fp16` (based on
  the paper's >80% reduction).
- **xKV** — `analytic_kv_mb = analytic_kv_mb_fp16 / 8` (based on the
  paper's reported 8×).

These three are clearly **marked** in every table and plot with a
distinct marker (`^` in the scatter plots) and a label "(simulated)"
in the legends.

## What gets written out

- **Per-cell CSV** — one row per (technique, ctx, batch, trial).
- **Per-technique CSV** — aggregated over trials.
- **Plots** — six PNGs in publication quality.

## Why pure-PyTorch PagedAttention is a fallback

`vllm` is the real implementation. It has CUDA kernels for the
page-table lookup, the block copy, and the attention computation.
None of that works on Windows. So we wrote a PyTorch fallback that:
- allocates a pool of fixed-size pages,
- maintains a per-request page table,
- reads the right pages for each attention step.

The fallback is **much slower** than real vLLM (no CUDA kernel
fusion), but it produces the same **qualitative** memory profile
(no over-allocation per sequence). We say so explicitly in the
limitations.

## Common viva questions

1. **Q: Why did you build your own framework instead of using an
   existing one?**
   A: Existing frameworks (vLLM's benchmarks, HuggingFace's
   `optimum-benchmark`) don't cover all 10 techniques in a single
   interface. We needed a unified interface to make the comparison
   fair.

2. **Q: How do you ensure fair comparison?**
   A: Same prompts, same seeds, same hardware, same metrics, same
   number of trials. The runner seeds every random number generator
   before every trial.

3. **Q: Why simulate instead of run NVFP4, MiniKV, xKV?**
   A: Their kernels are proprietary / one-shot preprocessing. We
   computed analytical estimates from the papers' own formulas and
   flagged them clearly in every table and plot.

4. **Q: Does your framework work on CPU?**
   A: Yes. PyTorch falls back to CPU when CUDA is not available.
   Numbers are slower but the framework runs.

5. **Q: Did you write tests?**
   A: Yes — `coding/tests/` has unit tests for the metrics
   collection, the analytic size formula, and the spec interfaces.
