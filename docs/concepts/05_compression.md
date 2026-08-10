# 05 — Compression Techniques

## What is KV cache compression?

Compression reduces the **amount of information** stored in the cache
by removing redundancy. Unlike quantization, which keeps the same
information in fewer bits, compression **changes the representation
itself** to a smaller one.

This affects the $D$ factor (head dimension) and the $L$ factor
(number of layers) in Equation 1.

## The four compression methods in our paper

### 1. Low-rank compression

**Idea:** every head's K and V vectors live in a $D$-dimensional
space. Most of the variance sits in a much smaller $r$-dimensional
subspace. We project K and V through a learned matrix
$W \in \mathbb{R}^{r \times D}$ and store only the $r$-dimensional
projections. To use them, we multiply by $W^\top$ to reconstruct.

**Memory:** $KV \propto r$ instead of $D$. With $r = D/2$ we cut memory
in half.

**Why we tested it:** it is the simplest form of compression and a
natural baseline.

**How we tested it:** we used **random Kaiming-uniform projection
matrices**, not calibrated SVD. This is a limitation we acknowledge in
the paper — the reconstruction error is worse than a calibrated
projection, but the **memory-latency trade-off is the same**.

### 2. KV merging

**Idea:** combine similar K/V pairs. If two past tokens have nearly
identical K vectors, store one merged vector instead of two.

**Memory:** depends on how much merging happens. In the best case,
substantial.

**Why we did not implement it deeply:** the literature has many
proposals with no single dominant method. We discuss it in the
survey section but do not benchmark it in `kvbench` because we did not
have time to settle on a single, reproducible implementation.

### 3. Multi-level compression

**Idea:** keep important tokens at full precision, compress
less-important ones more aggressively. Importance is decided by
attention scores or by recency.

**Memory:** adaptive.

**Why we did not implement it deeply:** same reason as KV merging.

### 4. xKV (simulated — we cannot run it)

This is the most interesting compression method in the paper.

**Observation (Chang et al., 2025):** the K/V caches of **adjacent
transformer layers are very similar**. The cosine similarity between
layer $i$ and layer $i+1$ is high. So storing 32 separate K/V caches
(one per layer) is wasteful.

**Method:** group every $k$ consecutive layers. Compute a
**low-rank Singular Value Decomposition (SVD)** of the K/V matrices
across the group. Store only the top singular vectors — a shared
"skeleton" representation. The full K/V for each layer can be
reconstructed from the skeleton plus a small per-layer offset.

**SVD refresher:** any matrix $M$ can be written as
$M = U \Sigma V^\top$ where $U, V$ are orthogonal and $\Sigma$ is
diagonal with non-negative singular values. Truncating to the top
$r$ singular values gives the best rank-$r$ approximation in the
Frobenius norm.

**Memory:** $L$ drops to $L/k$ in Equation 1. So if $k = 4$, the
cache is 4× smaller. The paper reports up to **8× reduction** and
**25.6× when combined with 4-bit quantization**.

**Cost:** a one-time offline SVD computation per layer group. Cost:
$O(D^2 \cdot k)$. After that, every layer reads the same shared
skeleton and only a small per-layer delta.

**Why we simulated:** the SVD preprocessing is a one-shot per
model, not pip-installable. We computed the analytical
$KV_{\text{xKV}} = KV_{\text{FP16}} / 8$ based on the paper's
reported compression ratio.

## How compression affects the math

For low-rank:
$$
KV \propto r \quad \text{instead of} \quad KV \propto D
$$

For xKV:
$$
KV \propto \frac{L}{k} \quad \text{instead of} \quad KV \propto L
$$

Asymptotic complexity is unchanged. The compression factor is a
constant multiplier on memory.

## Accuracy vs memory trade-off

| Method | Compression | Accuracy |
|--------|-------------|----------|
| Low-rank (rank 32, $D=64$) | 2× | moderate (depends on projection quality) |
| Low-rank calibrated SVD | 2–4× | much better |
| KV merging | 1.5–3× | depends on merge threshold |
| Multi-level | 2–4× | balanced |
| xKV | 8× | minimal loss |
| xKV + 4-bit | 25.6× | small loss |

The key insight from the xKV paper is that **cross-layer redundancy
is real and large**, so removing it is almost "free" in accuracy.

## Why compression is its own category

Compression changes the **representation** of the K/V vectors, not
just their bit width (quantization) or the architectural count of
heads (attention-level). It needs a **projection or decomposition
step**, which quantization does not. That's why it gets its own
column in the taxonomy.

## Common viva questions

1. **Q: What is low-rank compression?**
   A: Project K/V vectors into a smaller $r$-dimensional subspace
   using a learned matrix, and store the smaller projection.

2. **Q: What is SVD?**
   A: Singular Value Decomposition. Any matrix $M$ can be written as
   $U \Sigma V^\top$ where $\Sigma$ has the singular values on the
   diagonal. Truncating $\Sigma$ to the top $r$ values gives the
   best rank-$r$ approximation.

3. **Q: Why does xKV work?**
   A: Because adjacent transformer layers learn similar attention
   patterns, so their K/V caches are highly redundant.

4. **Q: How much can xKV compress?**
   A: Up to 8× by itself, and 25.6× when combined with 4-bit
   quantization, with minimal accuracy loss.

5. **Q: What's the catch with low-rank?**
   A: Reconstruction error. Random projections are worse than
   calibrated SVD. The cost of the projection and reconstruction
   on every step.
