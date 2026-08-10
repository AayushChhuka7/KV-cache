# 02 — The Taxonomy (the paper's main contribution)

## Why a taxonomy is needed

Before our paper, the literature was a mess:

- Each KV cache paper published in isolation.
- Different papers used different categorizations.
- Some papers focused on one technique and ignored others.
- Comparing two techniques from two different papers was almost impossible
  because the experiments, hardware, and metrics all differed.

So the **first contribution** of our paper is to put all 10 techniques on
the same shelf.

## What makes our taxonomy different

**Most prior work** categorized techniques by:
- publication year (2023 papers vs 2025 papers),
- research group (NVIDIA papers vs academic papers),
- the subfield (compression vs quantization).

**We categorize by bottleneck.** We look at Equation 1:

$$
KV_{\text{per token}} = 2 \times H \times L \times B \times D
$$

and we ask: **which factor does each technique change, or does it leave
the math alone and just store things better?**

This is a **bottleneck-oriented taxonomy**.

## The 4 categories

| # | Category | What it does | Factor in Eq. 1 |
|---|----------|--------------|-----------------|
| 1 | Attention-level redesign | Changes the architecture so fewer K/V heads are needed | $H$ |
| 2 | Quantization | Stores each number in fewer bits | $B$ |
| 3 | Compression | Removes redundancy in the K/V vectors | $D$, $L$ |
| 4 | Memory management | Leaves the math alone; stores the cache more efficiently | (none — storage layout) |

## The 10 techniques, mapped to categories

| Category | Technique | What it does in one line |
|----------|-----------|--------------------------|
| Attention-level | GQA (Grouped-Query Attention) | Multiple query heads share one K/V head |
| Quantization | FP8 (E5M2) | Store K/V in 8-bit floats |
| Quantization | INT8 | Store K/V in 8-bit integers |
| Quantization | NVFP4 (simulated) | Store K/V in 4-bit floats |
| Quantization | MixQuant | Different precisions for different tensors |
| Quantization | MiniKV (simulated) | 2-bit storage + evict unimportant tokens |
| Compression | Low-rank | Project K/V into a smaller rank-$r$ space |
| Compression | xKV (simulated) | Share low-rank K/V across layers via SVD |
| Memory management | PagedAttention (vLLM) | Page-based allocation like OS virtual memory |
| Memory management | CPU Offloading | Move old cache to CPU RAM, keep only the window in GPU |

## Why this taxonomy is useful for a decision framework

The decision framework in the paper says:

> "If your **bottleneck is GPU memory capacity**, look in the
> **Quantization** and **Compression** columns."
>
> "If your **bottleneck is per-user latency**, look in the
> **Attention-level** column — GQA gives architectural speedup with no
> per-step overhead."
>
> "If your **bottleneck is multi-user throughput**, look in the
> **Memory management** column — PagedAttention and Shared Attention let
> more requests share the GPU."
>
> "If your **bottleneck is accuracy**, look at MixQuant and multi-level
> compression — they balance memory and quality."

The taxonomy is the **index**, the decision framework is the **glossary**.

## How we built the taxonomy (the procedure)

1. Read each of the 10 papers / blog posts.
2. For each technique, find the equation or claim that says
   "this reduces memory by X" or "this reduces per-token cost by Y".
3. Look at which factor in Eq. 1 that reduction comes from.
4. Put the technique in the matching category.

That is why our taxonomy is **reproducible** — anyone who follows the same
procedure from the same papers will arrive at the same grouping.

## What we are NOT claiming

- We are **not** saying the four categories are airtight. Some techniques
  (MiniKV, xKV) genuinely live in two categories. We say so explicitly.
- We are **not** saying one technique is always best. The decision
  framework depends on the bottleneck.
- We are **not** saying the categories are novel. Earlier surveys
  sometimes use "quantization" and "compression" as one category; the
  four-way split is what we contribute.

## Common viva questions

1. **Q: Why did you build a taxonomy?**
   A: Because the existing literature is fragmented. We needed a single
   framework to compare 10 techniques that were published independently.

2. **Q: How is your taxonomy different from others?**
   A: Others categorized by year, paper, or sub-field. We categorized by
   **which factor of the KV-per-token equation each technique changes**.

3. **Q: Is your taxonomy perfect?**
   A: No. Some techniques (MiniKV, xKV) span two categories; we list them
   in both. But it's the most useful lens we found.

4. **Q: Give one example of why the taxonomy helps a practitioner.**
   A: A practitioner with a 24 GB GPU running Llama-3-8B and a 100k
   context can look at our Table V (synthesis) and immediately see that
   NVFP4 + xKV (4-bit) gives a ~25× reduction — that directly tells them
   which combination to try.
