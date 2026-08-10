# 07 — Complexity Analysis

This is the section that math-heavy teachers like to dig into. We
derive the complexity of every category from Equation 1.

## The two costs we care about

1. **Time complexity** — how many FLOPs (or operations) per decode step.
2. **Memory complexity** — how many bytes of KV cache.

The KV cache is fundamentally a **memory** optimization, so the
memory complexity is where the action is.

## Baseline complexity (no optimization)

### Without KV cache

For every new token $i$, we re-run K and V projections for all $i-1$
previous tokens. So:

$$
\text{Total FLOPs} = \sum_{i=1}^{N} O(i) = O(N^2)
$$

### With KV cache

We store K and V. We still have to do attention over all $N$ tokens,
but we skip the K/V projections for the past tokens. The asymptotic
complexity stays $O(N^2)$ but the constant factor is much smaller.

$$
\text{Time complexity with cache} = O(N^2)
$$

### Memory complexity

$$
KV_{\text{per token}} = 2 \cdot H \cdot L \cdot B \cdot D
$$
$$
KV_{\text{cache}} = N \cdot KV_{\text{per token}} = O(L \cdot H \cdot D \cdot B \cdot N)
$$

So baseline memory complexity is **linear in $N$** but has the
constant $L \cdot H \cdot D \cdot B$. That's the constant we attack.

## GQA complexity

GQA groups $g$ query heads behind one K/V head. So the number of cached
K/V heads is $H/g$ instead of $H$. Substituting into Eq. 1:

$$
KV_{\text{GQA}} = N \cdot 2 \cdot \frac{H}{g} \cdot L \cdot B \cdot D
= O\!\left(L \cdot \frac{H}{g} \cdot D \cdot B \cdot N\right)
$$

**Memory reduced by factor $g$. Time complexity unchanged.**

## Quantization complexity

Quantization changes $B$ to a smaller $B'$. For FP8, $B' = B/2$. For
NVFP4, $B' = B/4$. For MiniKV, $B' \approx B/8$ **and** $N$ is reduced
to $N' < N$ by eviction.

$$
KV_{\text{FP8}} = N \cdot 2 \cdot H \cdot L \cdot B' \cdot D
= O(L \cdot H \cdot D \cdot B' \cdot N)
$$

$$
KV_{\text{MiniKV}} = N' \cdot 2 \cdot H \cdot L \cdot B' \cdot D
= O(L \cdot H \cdot D \cdot B' \cdot N')
$$

**Memory reduced by a constant factor. Asymptotic complexity unchanged.**

There is a **constant per-step cost** for dequantization, which is
captured in the latency overhead but not in the $O(\cdot)$ form.

## Compression complexity

### Low-rank

Replace head dimension $D$ with rank $r < D$:

$$
KV_{\text{low-rank}} = N \cdot 2 \cdot H \cdot L \cdot B \cdot r
= O(L \cdot H \cdot r \cdot B \cdot N)
$$

### xKV

Share a low-rank K/V across every $k$ consecutive layers. The SVD
preprocessing cost is $O(D^2 \cdot k)$ per group, done once offline.
After SVD, the effective number of layers we store is $L/k$:

$$
KV_{\text{xKV}} = N \cdot 2 \cdot H \cdot \frac{L}{k} \cdot B \cdot D
= O\!\left(\frac{L}{k} \cdot H \cdot D \cdot B \cdot N\right)
$$

**Memory reduced by factor $k$. One-time preprocessing cost $O(D^2 \cdot k)$.**

## Memory-management complexity

These techniques do **not** change Eq. 1. The asymptotic memory
complexity is the same:

$$
KV_{\text{PagedAttention}} = O(L \cdot H \cdot D \cdot B \cdot N)
$$

What they change is **peak reserved memory**:

- **Without PagedAttention:** reserved memory is $O(\text{max context
  per request})$. This is the worst-case over-allocation.
- **With PagedAttention:** reserved memory is bounded by the
  **actually used** memory plus the page size. Wasted memory per
  active sequence is $O(\text{page size})$, not $O(\text{max context})$.

- **Offloading:** the GPU memory is bounded by $O(\text{window size})$
  instead of $O(N)$. The total memory (GPU + CPU) is still $O(N)$ but
  split across two devices.

## Summary table

| Technique | Recurrence | Memory saved by | Time complexity |
|-----------|------------|-----------------|------------------|
| Baseline | $O(L \cdot H \cdot D \cdot B \cdot N)$ | 1× | $O(N^2)$ |
| GQA | $O(L \cdot \frac{H}{g} \cdot D \cdot B \cdot N)$ | $g$ | $O(N^2)$ |
| FP8 | $O(L \cdot H \cdot D \cdot B' \cdot N)$, $B' = B/2$ | 2× | $O(N^2)$ + dequant |
| NVFP4 | $O(L \cdot H \cdot D \cdot B' \cdot N)$, $B' = B/4$ | 4× | $O(N^2)$ + dequant |
| MiniKV | $O(L \cdot H \cdot D \cdot B' \cdot N')$, $B' \approx B/8$, $N' < N$ | 5–10× | $O(N^2)$ + eviction |
| Low-rank | $O(L \cdot H \cdot r \cdot B \cdot N)$, $r < D$ | $D/r$ | $O(N^2)$ + reconstruct |
| xKV | $O(\frac{L}{k} \cdot H \cdot D \cdot B \cdot N)$ | $k$ | $O(N^2)$ + one-shot SVD |
| PagedAttention | $O(L \cdot H \cdot D \cdot B \cdot N)$ (same) | 1× (peak reduced) | $O(N^2)$ |
| Offloading | $O(L \cdot H \cdot D \cdot B \cdot N)$ (GPU window only) | partial | $O(N^2)$ + transfer |
| Shared Attention | $O(L \cdot H \cdot D \cdot B \cdot N)$ (shared prefix) | dup factor | $O(N^2)$ |

## What teachers like to ask

**Q: Why does the decoding complexity stay $O(N^2)$ after the KV cache?**

A: Because attention reads all cached tokens. The cache removes the
**recomputation** of K/V, but the attention dot product is still over
all $N$ tokens. The $O(N^2)$ is from attention, not from K/V projection.

**Q: Why is quantization a constant-factor optimization, not an
asymptotic one?**

A: Because $B$ is a constant in Big-O notation. Replacing $B$ with $B/2$
halves the constant. The growth rate in $N$ is unchanged.

**Q: Could any technique reduce the asymptotic complexity?**

A: Yes — anything that bounds the **effective** $N$ (e.g. MiniKV's
eviction keeps only $N' < N$ tokens). In that case, the complexity
becomes $O(N'^2)$ for attention, which is asymptotically smaller than
$O(N^2)$. But the constant factor for the eviction logic must be paid
each step.

**Q: Why is xKV's preprocessing $O(D^2 \cdot k)$?**

A: SVD on a $D \times kD$ matrix (concatenating $k$ layers' K/V) costs
$O(D^2 k)$ in the matrix-multiplications sense. It's done once per
group, so it's not in the per-step cost.

**Q: Why is PagedAttention's asymptotic complexity the same as the
baseline?**

A: Because PagedAttention doesn't change $L$, $H$, $D$, $B$, or $N$. It
changes the **allocation policy**. The total bytes allocated is still
$L \cdot H \cdot D \cdot B \cdot N$ — the peak is just closer to this
number instead of being inflated by worst-case over-reservation.
