# 03 — Attention-Level Techniques: GQA

## What is attention?

In a transformer, the attention layer takes a sequence of token vectors
and computes, for each token, a weighted sum of all other tokens:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V
$$

Each token produces three vectors: a **Query (Q)**, a **Key (K)**, and a
**Value (V)**.

## The "head" concept

A single attention "head" learns one pattern of "which tokens should
attend to which". In practice, transformers use **multiple heads in
parallel** so they can learn many patterns at once.

- **Multi-Head Attention (MHA):** $H$ independent heads, each with its
  own $Q, K, V$ projections. Standard in the original Transformer.
- **Multi-Query Attention (MQA):** all $H$ query heads share **one**
  $K$ and one $V$ projection. Fast, but quality drops.
- **Grouped-Query Attention (GQA):** a middle ground. Query heads are
  divided into $g$ groups, and each group shares one $K$ and one $V$.

## Why does this affect the KV cache?

The KV cache stores $K$ and $V$ for every head of every past token. So:

- **MHA:** cache has $H$ K-heads and $H$ V-heads per layer per token.
- **MQA:** cache has $1$ K-head and $1$ V-head per layer per token.
- **GQA:** cache has $H/g$ K-heads and $H/g$ V-heads per layer per token.

Replace $H$ with $H/g$ in Equation 1:

$$
KV_{\text{per token}}^{\text{GQA}} = 2 \times \frac{H}{g} \times L \times B \times D
$$

**Memory drops by a factor of $g$ with no per-step overhead.**

## Where GQA came from (Ainslie et al., 2023)

- The paper is called *"GQA: Training Generalized Multi-Query Transformer
  Models from Multi-Head Checkpoints"*.
- The authors show that an MHA model can be **converted to a GQA model
  cheaply** by averaging the K/V projections of every $g$ query heads.
- Training a GQA model from scratch also works, and gives nearly the
  same quality as MHA at much less memory cost.
- The famous **Llama-2-70B** uses GQA with $g=8$. Mistral and Llama-3
  also use GQA.

## Why GQA is in our taxonomy under "Attention-level"

GQA changes the **architecture** of the model. To use GQA, you either
need a model that was trained with GQA, or you apply the averaging trick
once. Either way, the model's structure is different from a vanilla
MHA model. So GQA is **not a runtime optimization** that you can flip
on and off — it's a design choice that affects everything downstream.

## How we tested GQA in `kvbench`

We used `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, which has 32 query heads
and 4 KV heads. That means $g = 32 / 4 = 8$. The analytic KV cache is
8× smaller than an equivalent MHA model with 32 KV heads. We confirmed
this by also loading `gpt2` (full MHA) and comparing `analytic_kv_mb`
under the same context length and batch size.

## Complexity impact

| Aspect | MHA | GQA |
|--------|-----|-----|
| KV per token | $2 \cdot H \cdot L \cdot B \cdot D$ | $2 \cdot (H/g) \cdot L \cdot B \cdot D$ |
| Decoding complexity | $O(N^2)$ | $O(N^2)$ (unchanged) |
| Quality | baseline | within ~1% on most benchmarks |
| Memory bandwidth | baseline | $1/g$ of MHA |

## GQA vs. other categories — why it shines for latency

GQA's main advantage is **no per-step overhead**. Quantization has to
quantize and dequantize on every step. Compression has to reconstruct on
every step. PagedAttention has to manage page tables. But GQA is just a
different layout — the hardware reads fewer bytes per step, and that's
it.

So for **single-user, latency-sensitive** inference (chat, code
completion), GQA is the lowest-hanging fruit.

## Limitations of GQA

- It requires a model trained or converted for GQA. You cannot turn it
  on for an arbitrary MHA model at runtime.
- Grouping too aggressively (large $g$) starts to lose quality. The
  paper's sweet spot is $g = 4$ to $g = 8$.
- It does not address the **multi-user** problem at all.

## Common viva questions

1. **Q: What is GQA in one line?**
   A: Multiple query heads share a single Key/Value head.

2. **Q: How much memory does GQA save?**
   A: By a factor of $g$ (the number of query heads per group). For
   Llama-2-70B, $g=8$, so 8× less KV cache.

3. **Q: Is GQA the same as MQA?**
   A: No. MQA is the extreme case of GQA with $g = H$. GQA is a
   parameterized family in between MHA ($g=1$) and MQA ($g=H$).

4. **Q: Does GQA change the decoding complexity?**
   A: No. The decode is still $O(N^2)$ attention over cached tokens. The
   memory savings come from fewer K/V vectors per token.

5. **Q: Can you apply GQA at runtime to an existing model?**
   A: Partially. Ainslie et al. describe a mean-pooling conversion of
   an MHA checkpoint to a GQA checkpoint, which is cheap. But you
   cannot do it on the fly during inference.
