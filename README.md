# KV Cache Optimization in LLMs: A Taxonomy and Decision Framework for Efficient Inference

This repository contains the paper, the benchmarking framework, and the measured data behind:

> **"KV Cache Optimization in LLMs: A Taxonomy and Decision Framework for Efficient Inference"**
> Aayush Chhuka, Aditya Bajracharya, Sudip Dhungana
> National College of Engineering, Lalitpur, Nepal

The manuscript itself is `main.tex` (compiled PDF: `main.pdf`).

---

## What this paper is

During autoregressive inference, LLMs store the keys and values of already-generated tokens in a key-value (KV) cache to avoid recomputing them. The cache grows linearly with context length and becomes the main bottleneck for GPU memory, memory bandwidth, and inference latency.

This paper does two things:

1. **A taxonomy.** Ten KV cache optimization techniques are organized into four classes based on _which term of the KV-size equation they reduce_:

   | Category                 | Techniques                                              |
   | ------------------------ | ------------------------------------------------------- |
   | Attention-level redesign | GQA (Grouped-Query Attention)                           |
   | Quantization             | INT8, FP8 (E5M2), NVFP4, MiniKV                         |
   | Compression              | Low-rank compression, xKV (cross-layer SVD)             |
   | Memory management        | PagedAttention (vLLM), CPU Offloading, Shared Attention |

2. **A decision framework backed by measurements.** All techniques are compared under identical conditions (same prompts, seeds, hardware, trials) using a common metric set, so the choice of technique can be driven by the actual bottleneck of the target system rather than by marketing claims.

## What was actually measured

All experiments ran on a single commodity machine:

- **GPU:** NVIDIA GeForce GTX 1650, 4 GB VRAM (sm_75)
- **CPU:** Intel Core i5-11300H, 16 GB RAM, Windows 11
- **Framework:** PyTorch 2.11.0+cu128, HuggingFace transformers

**Models:** `sshleifer/tiny-gpt2` (baseline; 2 layers, 16 bytes/token cache) and `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (GQA model, 32 query heads / 4 KV heads).

**Metrics per trial:** peak GPU memory, CPU resident-set size, wall-clock latency, throughput (tokens/s), time-to-first-token, and analytical KV cache size. Sweep grid: context 128/512/2048/4096 × batch 1/2/4 × 3 trials.

**Accuracy:** WikiText-2 (raw test split, 500 lines, 512-token windows) perplexity for the five techniques that run a real model: Baseline FP16, INT8, FP8, GQA, low-rank.

### Key results (analytical KV cache size, ctx=256, batch=1, tiny-gpt2)

| Technique             | KV cache  | Reduction vs FP16                       |
| --------------------- | --------- | --------------------------------------- |
| Baseline FP16         | 0.0039 MB | —                                       |
| Quantized INT8        | 0.0020 MB | 50.0%                                   |
| Quantized FP8 (E5M2)  | 0.0020 MB | 50.0%                                   |
| Low-rank compression  | 0.0039 MB | 0% (full-rank projectors at head_dim=1) |
| PagedAttention (vLLM) | 0.0020 MB | 48.4% (block-granular allocation)       |
| CPU Offloading        | 0.0039 MB | 0% (fits the 256-token GPU window)      |
| NVFP4 (simulated)     | 0.0010 MB | 75.0%                                   |
| MiniKV (simulated)    | 0.0001 MB | 98.1%                                   |
| xKV (simulated)       | 0.0001 MB | 96.9%                                   |
| GQA (TinyLlama-1.1B)  | 5.50 MB   | 8× vs same-shape MHA                    |

### Perplexity (TinyLlama-1.1B, WikiText-2 subset)

| Technique            | Perplexity | Change vs baseline                         |
| -------------------- | ---------- | ------------------------------------------ |
| Baseline FP16        | 11.25      | —                                          |
| Quantized INT8       | 11.25      | +0.02%                                     |
| Quantized FP8 (E5M2) | 11.33      | +0.70%                                     |
| GQA                  | 11.25      | 0.00%                                      |
| Low-rank compression | 2704.57    | +23947.6% (random uncalibrated projectors) |

**Main findings:** quantization trades a sub-1% perplexity change for half the cache; GQA's savings are architectural and measured to be lossless; compression only pays off with calibrated (SVD) projectors, not random ones; memory-management techniques are lossless but only help above the GPU window size.

## Repository layout

```
├── main.tex / main.pdf      The paper (LaTeX source + compiled PDF)
├── reference.bib            Bibliography
├── make_figures.py          Regenerates the paper's figures from measured CSV data
├── *.png                    The paper's figures (memory/throughput/latency curves, summary bars)
├── coding/                  kvbench — the benchmarking framework (see coding/README.md)
│   ├── kvbench/             framework source (techniques, runner, profiling, reporting)
│   ├── scripts/             CLI entry points (run_all.py, per-technique runners, sweep_checkpointed.py)
│   ├── results/             all measured data (paper_sweep/, perplexity_*, tinygpt2_memory_real/)
│   ├── tests/               pytest unit tests (CPU-only, no model downloads)
│   └── README.md            full framework documentation: install, run, reproducibility
├── docs/concepts/           Explainer notes: what a KV cache is, the
```

All code lives under `coding/`; the paper sources are at the repository root.

---

## Running the framework

Full documentation is in `coding/README.md`. Quick start:

```bash
cd coding
bash install.sh              # Linux / macOS / WSL
# or: install.bat            # Windows (vLLM not available; PagedAttention falls back to CPU sim)

python scripts/run_all.py --output results/all --trials 3
```

Perplexity evaluation:

```bash
python scripts/run_all.py --output results/final --trials 1 --with-perplexity --ctx 256 512 2048 --bs 1 2 4
```

Checkpointed sweep with resume support (what produced `coding/results/paper_sweep/`):

```bash
python scripts/sweep_checkpointed.py --output results/paper_sweep --trials 3 --ctx 128 512 2048 4096 --bs 1 2 4
```

Tests (CPU-only, no model downloads):

```bash
pytest tests/ -v
```

Reproducibility: every trial seeds Python `random`, NumPy, and PyTorch with `seed + trial_idx`; all models use identical prompts and `low_cpu_mem_usage=True`.

---

## Citing this work

```bibtex
@article{chhuka2026kvcache,
  title={KV Cache Optimization in LLMs: A Taxonomy and Decision Framework for Efficient Inference},
  author={Chhuka, Aayush and Bajracharya, Aditya and Dhungana, Sudip},
  year={2026}
}
```

If you use `kvbench` in your own work, cite the underlying technique papers as well (GQA, vLLM/PagedAttention, xKV, MiniKV, NVFP4, MixQuant, FlexGen, etc. — full list in `reference.bib`).

---
