# 06 — Memory Management Techniques

## What is memory management?

Memory management techniques do **not** change the size of the KV
cache. They change **how the cache is stored, allocated, and
accessed** on the hardware.

The cache itself still obeys Equation 1 — same number of bytes per
token. But the *peak* memory, the *fragmentation*, and the
*throughput* all change.

## The three memory-management methods in our paper

### 1. CPU Offloading

**Idea:** keep the **most recent** K/V vectors on the GPU (the
"GPU window") and move the **older** vectors to CPU RAM (pinned
memory, for fast DMA transfer). When a new token needs to attend to
an old vector that has been offloaded, transfer it back to the GPU.

**Why it helps:** the GPU is small (e.g. 8 GB on our test
machine) but the host RAM is large (e.g. 16 GB+). By using both,
the model can handle contexts that would not fit on the GPU
alone.

**Why it hurts:** every transfer across the PCIe bus takes time
($\sim$ microseconds per MB). For each generated token, we may
have to pull the entire old cache back into GPU.

**How we tested it:** we split the cache into a GPU window of
size $W$ (e.g. 256 tokens) and a CPU pinned-memory buffer for the
rest. On each step, the oldest GPU window slice is copied to CPU
and the latest slice is copied back.

**Measured impact:** in our runs, the GPU memory footprint stays
close to $W \times KV_{\text{per token}}$ instead of growing with
context length, but tokens/s drops because of the per-step
transfer.

### 2. PagedAttention (vLLM)

**Idea:** borrow the **virtual memory** idea from operating
systems. In a normal LLM serving system, each request reserves
a contiguous block of GPU memory for the maximum possible
context length (e.g. 128k tokens), even if the request only uses
a few thousand. That is **internal fragmentation** — wasted
memory inside the reserved block.

PagedAttention divides the KV cache into **fixed-size pages**
(e.g. 16 tokens per page). Each request is allocated a **page
table** that points to whichever pages it currently needs.
Memory is allocated **on demand** as new pages are needed.

**Why it helps:** it almost eliminates internal fragmentation.
The total reserved memory is close to the actually-used memory.
This lets the GPU serve **many more requests in parallel**.

**How it works (vLLM, Kwon et al., 2023):**
- Each request has a logical KV cache of arbitrary length.
- The physical storage is a pool of fixed-size pages.
- A per-request page table maps logical positions to physical
  pages.
- When a new token is generated, a new page is allocated if
  needed.
- Beam search and parallel sampling reuse pages across
  branches of the same request.

**Why we struggled to test it:** `vllm` is **Linux only**. Our
test machine is Windows. We had to fall back to a pure-PyTorch
**simulation** of PagedAttention that has the same qualitative
behavior (no over-allocation per sequence) but is much slower
than the real vLLM kernel. We document this in
`coding/LIMITATIONS.md`.

**Memory bound:** instead of $O(\text{max reserved context})$,
waste is bounded by $O(\text{page size})$ per active sequence.

### 3. Shared Attention

**Idea:** when many users send similar prompts (e.g. a system
prompt + user-specific tail), they can **share the K/V cache**
for the shared prefix. Only the user-specific tail needs
per-request storage.

**Why it helps:** in workloads like chatbots, retrieval-augmented
generation (RAG), and shared-document Q&A, the prefix is huge
and identical across requests. Sharing saves a lot of memory and
compute.

**How it works (Fawaz et al., 2024):**
- The serving system detects that multiple requests share a
  prefix.
- It computes the K/V for the prefix once and stores it in a
  shared cache.
- Each request's per-request cache is appended to the shared
  prefix cache.
- The shared prefix is not recomputed for every request.

**Why we did not benchmark it heavily:** it depends on the
**request pattern**, not just the model. We mention it in the
survey and synthesis table but did not run end-to-end benchmarks.

## Why these are in their own category

None of them change Equation 1. The bytes per token, heads, layers,
and head dimension are all the same. What changes is the **physical
layout** of those bytes on the hardware. So we group them as
"memory management" — orthogonal to quantization and compression.

## Complexity impact

| Method | Effect on $KV_{\text{per token}}$ | Effect on peak memory | Effect on latency |
|--------|-----------------------------------|------------------------|--------------------|
| Offloading | unchanged | reduced (window only on GPU) | increased (transfer) |
| PagedAttention | unchanged | reduced (no over-allocation) | small (page table lookup) |
| Shared Attention | unchanged | reduced (shared prefix) | reduced (no recompute) |

## What we measured in `kvbench`

For Offloading, we measured the GPU window size and the per-step
transfer time. For PagedAttention (simulated), we measured the
per-request reserved memory vs the actual cache size. Shared
Attention is discussed in the survey only.

## Limitations common to all three

- **Offloading:** PCIe transfer becomes the bottleneck for
  long contexts.
- **PagedAttention:** requires vLLM, which is Linux-only and
  needs an NVIDIA GPU for best performance.
- **Shared Attention:** only useful when requests really do
  share prefixes. For a chat with a long unique conversation,
  it does nothing.

## Common viva questions

1. **Q: What is PagedAttention?**
   A: A memory-management scheme that divides the KV cache into
   fixed-size pages and allocates them on demand, like OS
   virtual memory.

2. **Q: What problem does it solve?**
   A: Internal fragmentation. Without it, each request reserves
   memory for the maximum possible context, wasting GPU memory.

3. **Q: What is CPU offloading?**
   A: Storing old K/V vectors on CPU RAM and pulling them back
   to GPU when needed. It extends the effective context but
   adds transfer latency.

4. **Q: What is shared attention?**
   A: Reusing the K/V cache for prompts that share a common
   prefix across many requests. Saves both memory and compute.

5. **Q: Why did you not run real vLLM PagedAttention?**
   A: Our machine is Windows. vLLM is Linux-only. We used a
   PyTorch simulation that captures the qualitative behavior
   (no over-allocation) but not the full performance. This is
   documented in our limitations.
