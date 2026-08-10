# 01 — What is the KV Cache?

## Plain-English definition

A **KV cache** is a memory buffer inside an LLM that stores the **Key** and
**Value** vectors of every token the model has already processed, so that
when the model generates the next token, it does **not have to re-compute
those vectors from scratch**.

It's the standard textbook trick: **trade memory for compute**.

## Why it is needed

### Step 1: How a transformer generates text (autoregressive)

LLMs generate text one token at a time. To produce token $i$:

1. Take all previous tokens $t_1, t_2, \dots, t_{i-1}$.
2. Project them into **Query (Q), Key (K), Value (V)** matrices.
3. Compute attention: $\text{softmax}(Q K^\top / \sqrt{d}) \cdot V$.
4. Sample / argmax to pick the next token $t_i$.
5. Append $t_i$ to the input and repeat.

This is called **autoregressive generation** — each step depends on the
previous step's output.

### Step 2: The wastefulness problem

For every new token $i$, the model re-runs the **same K and V projections
for all $i-1$ previous tokens**. The K and V of token $t_1$ are the same
whether we are generating token $i=5$ or $i=5000$. So we are re-doing the
same matrix multiplications over and over.

Total work without a cache:
$$
\sum_{i=1}^{N} O(i) = O(N^2)
$$

That quadratic cost is fine for short prompts but explodes for long ones
(32k, 128k context windows).

### Step 3: The cache

Store the K and V of each past token in memory. When generating token $i$,
just **read** the cached K and V instead of **recomputing** them.

**Asymptotic complexity stays $O(N^2)$** because attention still has to
look at all previous tokens. But the **constant factor drops dramatically**
because we skip the K and V projections on every step.

## The two key equations (the entire paper depends on these)

### Equation 1 — KV per token

$$
KV_{\text{per token}} = 2 \times H \times L \times B \times D
$$

| Symbol | Meaning |
|--------|---------|
| $H$ | number of attention heads |
| $L$ | number of transformer layers |
| $B$ | bytes per element (e.g. 2 for FP16, 1 for FP8, 0.5 for FP4) |
| $D$ | head dimension |

The factor 2 is because we store both K **and** V.

For a given model, all four of $H, L, B, D$ are fixed, so the **KV per
token is a constant**. This is what makes the next equation important.

### Equation 2 — Total KV cache size

$$
KV_{\text{cache size}} = KV_{\text{per token}} \times \text{Context Length}
$$

This is **linear** in context length. **This linear growth is the root
problem the paper is about.**

## What breaks when the cache grows

1. **GPU memory capacity** — the cache has to fit in VRAM.
2. **Memory bandwidth** — every decode step reads the entire cache.
3. **Latency** — reading more bytes takes more time.
4. **Throughput** — fewer requests fit in a GPU.
5. **Cost** — bigger GPUs cost more.

## The paper's core idea

> Each optimization technique targets **one of the four factors** in
> Equation 1, or it targets **how the cache is stored** (not its size).

| Factor | Reduced by |
|--------|-----------|
| $H$ | GQA, MQA (share K/V heads across query heads) |
| $B$ | FP8, NVFP4, INT8, MixQuant, MiniKV (use lower precision) |
| $D$ | Low-rank compression, xKV (SVD across layers) |
| $L$ | xKV (share a low-rank representation across every $k$ layers) |
| storage layout | PagedAttention, Offloading, Shared Attention |

## A useful analogy

Think of an LLM as a student writing a paragraph.
- **Without KV cache:** the student re-reads every previous sentence before
  writing the next word.
- **With KV cache:** the student writes each word on a sticky note and
  glances at the sticky notes instead of re-reading.
- The problem is that **the sticky-note pile grows** as the paragraph
  gets longer, and the student has to glance at more and more sticky
  notes for every new word.
- The optimizations are different strategies to make the pile **smaller,
  lighter, or easier to organize**.

## Common viva questions (practice answering)

1. **Q: What is a KV cache?**
   A: A memory buffer that stores the Key and Value vectors of past tokens
   so the model doesn't have to recompute them during autoregressive generation.

2. **Q: Why do we need it?**
   A: Without it, every new token requires redoing the K and V projections
   for all previous tokens, which makes the cost $O(N^2)$ in time and gives
   poor latency for long contexts.

3. **Q: Does the KV cache change the asymptotic complexity?**
   A: No. Decoding is still $O(N^2)$ because attention reads all cached
   tokens. The cache reduces the constant factor, not the exponent.

4. **Q: Why is the KV cache a problem?**
   A: Because it grows linearly with context length. For a 128k context
   LLM, the cache can take many GB of GPU memory, and reading it every
   decode step saturates memory bandwidth.

5. **Q: What are K and V?**
   A: K (Key) is a learned representation that identifies a token, V (Value)
   carries the token's content. Together they let the attention mechanism
   decide which past tokens matter for the next token.
