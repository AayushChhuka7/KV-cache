# 04 — Quantization Techniques

## What is quantization?

Quantization means **storing numbers in fewer bits**.

A weight stored in FP16 uses 16 bits = 2 bytes.
A weight stored in FP8 uses 8 bits = 1 byte.
A weight stored in FP4 uses 4 bits = 0.5 bytes.

Halving the bit count halves the memory and (often) halves the
bandwidth. The cost is a small loss in numerical precision.

This is the **$B$ factor** in Equation 1.

## Why quantization works for the KV cache

The KV cache is just a big tensor of floating-point numbers. It can be
quantized exactly like model weights. The model has to dequantize the
K and V back to FP16 just before the attention computation, then
re-quantize them after the computation if it wants to keep them stored
in low precision.

## The five quantization methods in our paper

### 1. INT8

- Each K/V element stored as an 8-bit integer.
- A scale factor per channel converts between int8 and float16.
- Used in our `kvbench` via per-channel symmetric quantization.
- **Memory:** 50% of FP16. **Accuracy:** small loss.
- **Hardware:** supported on basically every modern GPU.
- **Why we used it:** it's the safest baseline; well understood.

### 2. FP8 (E5M2)

- Each K/V element stored as an 8-bit float with 5-bit exponent and
  2-bit mantissa.
- We used `torch.float8_e5m2` when the GPU supports it; otherwise we
  approximated by storing 1 byte per element.
- **Memory:** 50% of FP16. **Accuracy:** slightly less than INT8 for
  outlier-heavy tensors, comparable in our runs.
- **Hardware:** needs an H100 or RTX 4090-class GPU for native FP8
  tensor cores. On older GPUs, dequantization becomes the bottleneck.

### 3. NVFP4 (simulated — we cannot run it)

- NVIDIA's 4-bit floating-point format for the KV cache.
- Halves the FP8 cache size (75% reduction vs FP16).
- Originally published in a December 2025 NVIDIA developer blog.
- We **simulate** it analytically: $KV_{\text{NVFP4}} = KV_{\text{FP16}} / 4$.
- **Why simulated:** the kernel is proprietary and not pip-installable.
- **Why we include it anyway:** it's the strongest 4-bit contender in
  the literature and excluding it would weaken the taxonomy.

### 4. MixQuant

- Mixed-precision quantization: **different precisions for different
  tensors**.
- Important K tensors (e.g. for system prompts) are kept at FP8;
  less important ones are dropped to INT4.
- The paper is *Mix-Quant: Quantized Prefilling, Precise Decoding for
  Agentic LLMs* (Lu et al., 2026).
- **Memory:** variable, tensor-dependent. **Accuracy:** better than
  uniform low precision because critical tensors keep higher precision.
- We did not implement MixQuant directly; it sits in the same
  quantization column and is discussed in the survey tables.

### 5. MiniKV (simulated — we cannot run it)

- Combines 2-bit storage with **adaptive token eviction**.
- The eviction policy keeps only the most "important" tokens based on
  attention scores.
- Memory: more than 80% reduction vs FP16 in the published numbers.
- We **simulate** analytically: $KV_{\text{MiniKV}} = 0.2 \times KV_{\text{FP16}}$.
- **Why simulated:** the kernel is proprietary.
- This is the single biggest memory reducer in the entire paper.

## How quantization affects the math

For FP8, $B$ drops from 2 bytes to 1 byte, so $KV_{\text{per token}}$
halves. For NVFP4, $B$ drops to 0.5 bytes, so it quarters. Asymptotic
complexity does not change, but the constant factor shrinks.

For MiniKV, **two things change at once**: $B$ drops (2-bit) **and**
the effective number of cached tokens drops (eviction), so
$KV \propto B' \cdot N'$ where $N' < N$.

## Accuracy vs memory trade-off (the central tension)

| Method | Bits | Memory vs FP16 | Accuracy |
|--------|------|----------------|----------|
| FP16 | 16 | 100% | baseline |
| INT8 | 8 | 50% | ~0.1–0.3% drop on most tasks |
| FP8 | 8 | 50% | similar to INT8 |
| NVFP4 | 4 | 25% | ~0.5–1% drop |
| MixQuant | mixed | 30–60% | better than uniform FP4 |
| MiniKV | 2 + eviction | <20% | retained by adaptive eviction |

The "best" method depends on the **accuracy budget** of the
application. A creative-writing chatbot might tolerate 1% drop. A
medical Q&A system might not.

## Why quantization is its own category

Quantization does not change the **structure** of the cache (no
head-sharing, no SVD, no pages). It only changes the **number of bits
per element**. That's a clean, single-axis change, which is why it
makes sense as one category.

## Limitations common to all quantization methods

- **Dequantization overhead** — every step, you have to convert back
  to FP16 to do the attention math. On GPUs without FP8/FP4 tensor
  cores, this overhead eats the savings.
- **Outliers** — a few very large K/V values can dominate the
  dynamic range and hurt accuracy. INT8/FP8 papers spend a lot of
  effort on outlier handling (per-channel scaling, smoothquant, etc.).
- **Training vs post-training** — most KV cache quantization is
  post-training (no retraining), but accuracy is best when the
  model is trained or fine-tuned to tolerate the lower precision.

## Common viva questions

1. **Q: What is the simplest way to halve KV cache memory?**
   A: Quantize to FP8 or INT8.

2. **Q: Why doesn't quantization change asymptotic complexity?**
   A: Because $B$ is a constant factor in Equation 1. Halving it halves
   the constant; it does not change the $O(\cdot)$ growth.

3. **Q: What is the difference between INT8 and FP8?**
   A: INT8 is integer (uniform spacing of values), FP8 is float
   (non-uniform spacing with an exponent). FP8 is better when there
   are outliers.

4. **Q: Why did you simulate NVFP4 and MiniKV instead of running them?**
   A: Their GPU kernels are proprietary and not pip-installable. We
   derived analytical estimates from the papers' own equations and
   flagged them as simulated in all tables and plots.

5. **Q: Does quantization always hurt accuracy?**
   A: Yes, but the loss can be made very small (under 0.5% on most
   tasks) with the right scaling and outlier handling. The MiniKV
   paper shows that 2-bit + adaptive eviction is a special case where
   accuracy is preserved by the eviction policy.
