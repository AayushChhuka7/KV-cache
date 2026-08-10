# 10 — What We Learned

This is the section that summarizes the **insights** from the whole
study. Teachers usually ask at least one of these questions.

## Insight 1 — The bottleneck drives the choice

The 10 techniques do not solve the same problem. They solve
**different** problems. The right one depends on which bottleneck
your system has.

| Bottleneck | Best-fit category | Best-fit techniques |
|------------|-------------------|---------------------|
| GPU memory capacity | Quantization, Compression | NVFP4, MiniKV, xKV |
| Single-user latency | Attention-level | GQA |
| Multi-user throughput | Memory management | PagedAttention, Shared Attention |
| Accuracy preservation | Quantization, Compression | MixQuant, multi-level |

This is the **central decision framework** of the paper.

## Insight 2 — Techniques compose, not compete

The techniques are not mutually exclusive. They can (and should) be
**stacked**. The paper's headline example:

> xKV (8× cross-layer compression) **combined with** 4-bit
> quantization gives a **25.6× total reduction** with only a small
> accuracy loss.

Another common stack in production:

> PagedAttention (memory management) **on top of** FP8 quantization
> — used in vLLM serving stacks.

So the taxonomy is a **set of building blocks**, not a one-from-each
question.

## Insight 3 — The trade-off is multi-dimensional

There is no single "best" axis. Each technique has its own
trade-off curve:

- **Quantization** trades **accuracy** for **memory**.
- **Compression** trades **reconstruction quality** for **memory**.
- **Attention-level redesign** trades **architectural flexibility**
  for **memory and speed**.
- **Memory management** trades **complexity** for **serving throughput**.

A good deployment picks a combination that respects **all four** of
the constraints: memory, latency, accuracy, complexity.

## Insight 4 — The analytic KV cache size is a lower bound

`analytic_kv_mb` (from Eq. 1) is the **theoretical minimum** for the
cache size given the model's architecture and the technique's
parameters. The actual `peak_gpu_mb` is always larger because:

- Model weights are loaded (often the largest fraction).
- The allocator has overhead.
- Intermediate activations live on the GPU.
- Driver and CUDA context reserve memory.

The analytic number is the **headroom** — how much memory is
strictly the cache. Everything else is fixed for a given model.

In our experiments, the model weights dominated, so the relative
ordering of techniques was the same in `analytic_kv_mb` and
`peak_gpu_mb`, but the absolute ratio was small.

## Insight 5 — Architecture beats runtime (GQA)

GQA is the only technique that **saves memory without any per-step
overhead**. Quantization has to dequantize; compression has to
reconstruct; PagedAttention has to walk the page table; offloading
has to DMA. GQA is just a different layout — the hardware reads
fewer bytes because there are fewer bytes.

So for **latency-critical single-user** inference, GQA is the
lowest-overhead choice.

## Insight 6 — Cross-layer redundancy is real (xKV)

The xKV paper's central claim — that adjacent transformer layers
have similar K/V caches — was surprising to us but the
**8× + 4-bit = 25.6×** number is hard to argue with. It also
matches the intuition that transformers have "redundant
intermediate computation" that has been observed in other
compression work (e.g. layer pruning).

The cost is a one-time SVD step per layer group, which is
acceptable for an offline model preparation.

## Insight 7 — Memory management is orthogonal and lossless

PagedAttention, offloading, and shared attention **do not lose any
information**. They just move the cache around. So they are the
**safest** techniques to deploy — you cannot degrade accuracy.

The cost is **complexity** (orchestrating the page table, the
DMA, the shared-prefix detection) and possibly **latency** (PCIe
transfers for offloading).

## Insight 8 — Quantization and compression need hardware

A hidden assumption behind MiniKV and NVFP4 is **GPU tensor cores**
that can do FP4/FP8/INT8 arithmetic at full speed. On our test
hardware (NVIDIA MX230, compute capability 6.1), many of these
formats are not natively supported. Dequantization to FP16 becomes
the bottleneck, eating the savings.

This is why we **simulated** those techniques — we could not do
them fairly on our hardware.

## Insight 9 — The relative ordering is hardware-independent

The **absolute** numbers (latency, throughput) depend on the GPU.
But the **relative ordering** of techniques is stable. INT8 is
always faster than FP16, FP8 is always faster than FP16, NVFP4 is
always the smallest cache, etc. As long as you compare on the same
machine, the framework's conclusions hold.

## Insight 10 — A good benchmark is honest about its limits

We made the framework distinguish **measured** results from
**analytical simulations**. Every table has a footnote. Every
plot has a marker. The reader can never confuse a real number
with a formula. This is the part of the paper we are most proud
of — we never oversold the data.

## Common viva questions

1. **Q: What is the single most important takeaway of the paper?**
   A: The choice of KV cache optimization technique depends on the
   bottleneck. There is no universal best. The taxonomy is a
   decision framework, not a ranking.

2. **Q: Can you give one example of stacking techniques?**
   A: xKV (8×) + 4-bit quantization (4×) = 25.6× total reduction
   with small accuracy loss. This is from the xKV paper itself.

3. **Q: What's the difference between a technique and a framework?**
   A: A technique changes how the cache is stored or computed
   (quantization, compression, GQA). A framework is the tool that
   compares them. `kvbench` is the framework; GQA, FP8, etc.
   are the techniques.

4. **Q: Why is the GQA-style architectural change the cleanest?**
   A: Because it has no per-step overhead. The hardware just reads
   fewer bytes.

5. **Q: What is the most surprising thing you found?**
   A: That cross-layer redundancy is so large (8×). The xKV
   paper's claim held up to our analytic reproduction.
