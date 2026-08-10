# 11 — Problems We Faced and How We Solved Them

This is the section that teachers ask when they want to know
"what problems did you actually face during the project?". Be
honest — every project has problems. What matters is how you
solved them.

## Problem 1 — Three techniques are not pip-installable

**What happened.** NVFP4, MiniKV, and xKV all depend on **proprietary
GPU kernels** or **one-shot offline preprocessing** that are not
publicly available. We tried:

- Searching PyPI for `nvfp4`, `minikv`, `xkv` — none found.
- Contacting the NVIDIA developer program for the NVFP4 kernel
  (no response in time).
- Installing the xKV preprocessing script — it requires a
  proprietary calibration dataset.

**The fix.** We built **analytical simulators** for these three
techniques. Each simulator computes the KV cache size from
Equation 1 using the technique's reported compression factor:

- NVFP4: $KV = KV_{\text{FP16}} / 4$.
- MiniKV: $KV = 0.2 \times KV_{\text{FP16}}$ (matches the >80%
  reduction).
- xKV: $KV = KV_{\text{FP16}} / 8$ (matches the 8× reduction).

We documented this in `coding/LIMITATIONS.md` and in the paper's
Limitations section. Every table and plot marks these rows with
"(simulated)" or a distinct marker so the reader is never misled.

## Problem 2 — vLLM does not run on Windows

**What happened.** `vllm` (the real PagedAttention implementation)
is Linux-only. Our test machine is Windows 11. We could not run
the actual PagedAttention benchmark.

**The fix.** We wrote a **pure-PyTorch simulation** of PagedAttention:

- Allocate a pool of fixed-size pages.
- Maintain a per-request page table.
- Use the page table to look up the correct pages for each
  attention step.

The simulation has the **same qualitative** memory profile as
the real vLLM (no per-sequence over-allocation), but it is **much
slower** (no CUDA kernel fusion). We noted this in the
Limitations and recommended WSL2 or Linux for real PagedAttention
benchmarks.

## Problem 3 — `torch.float8_e5m2` is not available on the MX230

**What happened.** Our GPU is an NVIDIA MX230 with compute capability
6.1. PyTorch's FP8 dtypes require SM 8.9+ (RTX 4090 or H100). The
FP8 code path fell back to a CPU approximation that was painfully
slow.

**The fix.** We added a **graceful fallback** in the FP8 technique:
if `torch.float8_e5m2` is not available, store K/V as 1-byte
approximations (just halving the element count) and dequantize
to FP16 on read. This is what the paper reports as the "FP8 (sim)"
row in the techniques table.

The authors of NVFP4 specifically designed their format for
Hopper-class GPUs. The lesson: **published numbers assume
hardware you may not have**.

## Problem 4 — Bitsandbytes is Windows-unfriendly

**What happened.** `bitsandbytes` (the de-facto INT8 quantization
library) ships pre-compiled CUDA wheels for Linux but not for
Windows. The Windows install path is fragile.

**The fix.** We wrote our **own** per-channel symmetric INT8
quantizer in pure PyTorch using `torch.round` and `torch.clamp`.
It is not as fast as bitsandbytes but it works on every platform
and is reproducible.

## Problem 5 — Memory measurement is noisy

**What happened.** `torch.cuda.max_memory_allocated()` includes
allocator overhead, which varies between runs. We saw 5–10%
variance in `peak_gpu_mb` between identical trials.

**The fix.** We averaged over **5 trials** per (technique, ctx,
batch) combination. The runner reports the **mean** and the
**standard deviation** for every cell. The tables in the paper
show the mean.

## Problem 6 — Cold-start vs warm cache

**What happened.** The first trial of every technique pays the
model-loading cost. Subsequent trials are faster. This made the
`tokens_per_s` numbers look slower than they really are for
production servers that pre-warm.

**The fix.** We added a **warm-up trial** that is not counted. The
runner runs the technique once without measuring, then counts the
next $N$ trials.

## Problem 7 — Offloading stuck on small contexts

**What happened.** When the context length is small (e.g. 128
tokens), the offloading "window" is larger than the entire cache,
so the offloading path degenerates to the baseline. The numbers
are wasted.

**The fix.** We made the offloading window **proportional to
context length** (e.g. min(256, ctx / 4)). When the context is
very small, offloading is skipped. This is honest: offloading
is for long contexts.

## Problem 8 — Long-context prompts don't fit in 8 GB GPU

**What happened.** Our largest configuration (ctx=4096, batch=4)
with the FP16 baseline tried to allocate more memory than the
MX230's 2 GB. The kernel crashed.

**The fix.** We added an **adaptive batch sizing** fallback: if
the configuration cannot fit, the runner reduces the batch size
and records the largest batch that **did** fit. The paper
acknowledges this as a hardware limitation.

## Problem 9 — Random projections for low-rank gave bad accuracy

**What happened.** Our low-rank compression uses random Kaiming
projection matrices. The reconstruction error is much higher
than what a calibrated SVD would give. So the numbers in
Table V are a **pessimistic** estimate of low-rank's accuracy.

**The fix.** We documented this in the Limitations section. The
paper says the low-rank results demonstrate the **memory-latency
trade-off** but not the accuracy claims of the original SVD-based
methods.

## Problem 10 — Reproducibility across machines

**What happened.** The framework ran on the development machine
(Windows 11, MX230, 8 GB RAM) but the absolute numbers will
change on any other machine. We did not have a CI cluster.

**The fix.** We seeded **every** random number generator (Python
`random`, NumPy, PyTorch) before every trial. The framework
ensures the same machine produces the same numbers. For
**cross-machine** comparisons, we recommend the relative numbers
(the ordering of techniques) rather than absolute latency.

## How we documented problems

- `coding/LIMITATIONS.md` — the full list of limitations.
- The paper's "Limitations" section — a condensed version.
- Inline comments in the code where the workaround lives.

## What we learned from these problems

1. **Reproducibility is a discipline**, not a feature. You cannot
   add it at the end.
2. **Hardware matters.** The published numbers in NVFP4 and MiniKV
   assume Hopper-class GPUs. We have a laptop GPU. Plan accordingly.
3. **Cross-platform code is hard.** vLLM and bitsandbytes are
   Linux-friendly; Windows is a second-class citizen.
4. **Honesty is the best policy.** When we could not run a
   technique, we **simulated** it and **marked** it. We never
   presented a simulation as a measurement.

## Common viva questions

1. **Q: What was the biggest problem you faced?**
   A: The three proprietary techniques cannot be run on our
   hardware, so we had to simulate them. We solved this by
   being explicit about it in every table and plot.

2. **Q: Why didn't you use vLLM?**
   A: Windows incompatibility. We wrote a pure-PyTorch
   simulation with the same qualitative behavior.

3. **Q: How did you ensure reproducibility?**
   A: Seeded every RNG before every trial, ran multiple trials
   per cell, and averaged.

4. **Q: What would you do differently if you had more time?**
   A: Use a Linux machine with an H100 GPU, run real NVFP4 /
   MiniKV / xKV kernels, and add accuracy benchmarks
   (perplexity, MMLU).
