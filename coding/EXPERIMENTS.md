# EXPERIMENTS.md — concrete commands and expected outputs

This document is a reference for running the benchmark on real
hardware. Each subsection lists the exact command, what it produces,
and what to look for in the output.

## 1. Smoke test (no model downloads)

```bash
python tests/smoke_test.py
```

Produces:

```
results/_smoke/
  report.md
  smoke.json
```

The smoke test runs only the analytical simulators (NVFP4, MiniKV,
xKV), so it finishes in a few seconds without downloading anything.
Use it to verify the install.

## 2. Baseline only

```bash
python scripts/run_baseline.py --output results/baseline --trials 3
```

Loads `KVBENCH_MODEL` (default `sshleifer/tiny-gpt2`), runs a sweep
across the default context lengths (128, 512, 2048, 4096) and batch
sizes (1, 2, 4), and writes `report.md`.

## 3. Quantization sweep

```bash
python scripts/run_quantization.py --output results/quantization --trials 3
```

Runs **Baseline FP16**, **Quantized INT8**, **Quantized FP8 (E5M2)**.
Expect to see:

* KV cache memory drops by ~ 50% for INT8 and ~ 50% for FP8
  (both store 1 byte per element vs 2 for FP16).
* `tokens_per_s` may *increase* slightly for INT8/FP8 because
  less memory bandwidth is consumed per token; or may *decrease*
  if the dequantization kernel is slow on your GPU.
* Peak GPU memory is *not* expected to drop by 50% — the model's
  weights dominate, and quantization only reduces the per-token
  cache footprint.

## 4. GQA vs MHA

```bash
python scripts/run_gqa.py --output results/gqa --trials 3 \
    --model-mha gpt2 \
    --model-gqa TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

Two HuggingFace models are loaded: an MHA model and a GQA model.
Expect:

* The GQA model has a smaller `analytic_kv_mb` by a factor equal to
  `num_attention_heads / num_key_value_heads`.
* Tokens/sec is usually higher for GQA at long contexts because
  the attention kernel has less data to read.

## 5. PagedAttention (vLLM)

```bash
python scripts/run_paged_attention.py --output results/paged --trials 3
```

Requires `vllm` (Linux + NVIDIA). The script will print a warning
and use the CPU fallback if vLLM is unavailable.

With vLLM you should see:

* The peak GPU memory is closer to the *actual* cache size, not
  the worst-case reserved size, because blocks are allocated on
  demand.
* `tokens_per_s` is typically much higher than the baseline at
  large batch sizes because vLLM batches more efficiently.

## 6. CPU Offloading

```bash
python scripts/run_offloading.py --output results/offloading --trials 3 --windows 128 256 1024
```

Three offloading configurations (GPU window = 128 / 256 / 1024) are
compared against the baseline. Expect:

* Smaller GPU windows → lower peak GPU memory, higher wall time
  (more transfers).
* Larger GPU windows → closer to baseline behaviour.

## 7. Low-Rank Compression

```bash
python scripts/run_compression.py --output results/compression --trials 3 --ranks 32 64 128
```

Three ranks are compared. Expect:

* KV memory scales linearly with `rank`. A `rank=32` cache is
  approximately `32/head_dim` the size of the baseline (e.g., 50%
  smaller for `head_dim=64`).
* Tokens/sec drops because of the reconstruction overhead.

## 8. Simulated techniques

```bash
python scripts/run_simulations.py --output results/simulations --trials 1
```

Runs the baseline + the three simulators. The simulators'
`wall_time_s` and `tokens_per_s` columns are zero — only the
`KV Cache MB` and `Memory Saving (%)` columns are populated.

## 9. Scaling sweep (context × batch)

```bash
python scripts/run_scaling.py --output results/scaling --trials 2 \
    --ctx 128 512 1024 2048 4096 --bs 1 2 4 8
```

Runs Baseline, INT8, and GQA across a 5 × 4 grid of (ctx, bs).
Produces the most informative plots: memory vs context for each
batch size, and memory vs batch for each context length.

## 10. Full sweep

```bash
python scripts/run_all.py --output results/all --trials 3
```

Runs every technique across every (ctx, bs). Produces a single
`report.md` plus all the CSVs and plots.

Flags:

* `--no-plots` — skip matplotlib.
* `--no-simulations` — exclude NVFP4 / MiniKV / xKV.
* `--ctx 128 512` — override the context-length sweep.
* `--bs 1 2` — override the batch-size sweep.

---

## Reading the report

`results/<exp>/report.md` contains three blocks:

1. **Configuration** — the exact `BenchmarkConfig` used (yaml).
2. **Per-Technique Summary** — averaged across all (ctx, bs, trial)
   combinations. Use this for headline numbers.
3. **Per-Cell Results** — one row per (technique, ctx, bs). Use this
   to spot cells where a technique degrades unexpectedly.

`results.csv` is the per-cell table in machine-readable form;
`results_aggregated.csv` is the per-technique summary.

## Reproducing a paper figure

```bash
python scripts/run_all.py --output results/all --trials 3 --no-simulations
```

then open `results/all/memory_vs_context.png` for the headline
memory-savings figure, or `summary_kv_mb.png` for the bar chart of
average KV cache size per technique.
