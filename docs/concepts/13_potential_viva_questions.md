# 13 — Potential Viva Questions and Model Answers

Practice these out loud. ~50 questions total, organized by topic.

## A. Background (KV cache basics)

1. **Q: What is the KV cache?**
   A: A memory buffer that stores the Key and Value vectors of
   past tokens to avoid re-computing them during autoregressive
   generation.

2. **Q: Why do we need it?**
   A: Without it, each new token recomputes K/V for all previous
   tokens, making the cost $O(N^2)$ and giving poor latency.

3. **Q: Does the KV cache change asymptotic complexity?**
   A: No. Decoding is still $O(N^2)$ because attention reads all
   cached tokens. The cache reduces the constant factor.

4. **Q: Why is the KV cache a problem?**
   A: It grows linearly with context length, so it can exhaust
   GPU memory, saturate bandwidth, and slow down inference.

5. **Q: What is the formula for KV cache size?**
   A: $KV_{\text{per token}} = 2 \cdot H \cdot L \cdot B \cdot D$
   and $KV_{\text{cache}} = N \cdot KV_{\text{per token}}$.

6. **Q: What is autoregressive generation?**
   A: Generating tokens one at a time, where each token depends
   on all previous tokens.

7. **Q: What is attention?**
   A: A weighted sum of value vectors weighted by the
   softmax-normalized dot product of queries and keys.

## B. Taxonomy

8. **Q: Why did you build a taxonomy?**
   A: The literature was fragmented. We needed a unified
   framework to compare 10 techniques fairly.

9. **Q: What is your taxonomy based on?**
   A: The bottleneck each technique addresses — which factor of
   the KV-per-token equation it changes.

10. **Q: What are the four categories?**
    A: Attention-level redesign, quantization, compression,
    memory management.

11. **Q: Why is GQA attention-level and not quantization?**
    A: Because GQA changes the architecture (number of K/V
    heads), not the bit width.

12. **Q: Why is low-rank compression and not quantization?**
    A: Because it changes the representation (the dimension
    $D$), not the bit width.

13. **Q: Why is PagedAttention memory management and not
    compression?**
    A: Because it does not change the cache size at all — only
    how it is allocated.

14. **Q: Can a technique belong to two categories?**
    A: Yes. MiniKV is quantization + eviction. xKV is
    compression + cross-layer. We list them in both.

## C. Techniques

15. **Q: What is GQA?**
    A: Grouped-Query Attention. Multiple query heads share one
    K/V head.

16. **Q: How much memory does GQA save?**
    A: By a factor of $g$ (the number of query heads per
    group). Llama-2-70B has $g=8$, so 8× less KV cache.

17. **Q: What is FP8?**
    A: 8-bit floating-point, with 5-bit exponent and 2-bit
    mantissa. Halves memory vs FP16.

18. **Q: What is NVFP4?**
    A: NVIDIA's 4-bit floating-point format. Quarters memory
    vs FP16.

19. **Q: What is MiniKV?**
    A: 2-bit quantization + adaptive token eviction. Reduces
    KV cache by >80%.

20. **Q: What is low-rank compression?**
    A: Project K/V into a smaller rank-$r$ space. Saves memory
    by $D/r$.

21. **Q: What is xKV?**
    A: Cross-layer KV cache compression using SVD. Shares a
    low-rank representation across every $k$ layers.

22. **Q: What is PagedAttention?**
    A: Page-based KV cache allocation, like OS virtual memory.
    Eliminates internal fragmentation.

23. **Q: What is CPU offloading?**
    A: Move old K/V vectors to CPU RAM, keep a GPU window. Adds
    transfer latency but extends effective context.

24. **Q: What is shared attention?**
    A: Reuse K/V cache for shared prefixes across requests.
    Saves memory and recompute in multi-user serving.

## D. Complexity

25. **Q: Why is decoding $O(N^2)$ even with the KV cache?**
    A: Because attention reads all cached tokens. The cache
    removes recomputation of K/V, not the attention dot product.

26. **Q: Why does quantization not change asymptotic complexity?**
    A: Because $B$ is a constant in Big-O. Halving it halves
    the constant, not the growth rate.

27. **Q: What is the asymptotic memory complexity of the
    baseline?**
    A: $O(L \cdot H \cdot D \cdot B \cdot N)$.

28. **Q: What is the asymptotic memory complexity of GQA?**
    A: $O(L \cdot (H/g) \cdot D \cdot B \cdot N)$.

29. **Q: What is the asymptotic memory complexity of xKV?**
    A: $O((L/k) \cdot H \cdot D \cdot B \cdot N)$.

30. **Q: What is the SVD preprocessing cost for xKV?**
    A: $O(D^2 \cdot k)$ per layer group, done once offline.

## E. Methodology

31. **Q: Why did you build `kvbench`?**
    A: No existing framework covered all 10 techniques in a
    single interface.

32. **Q: Why simulate NVFP4, MiniKV, xKV?**
    A: Their kernels are proprietary. We compute analytical
    estimates from the papers' formulas.

33. **Q: How do you ensure fair comparison?**
    A: Same prompts, same seeds, same hardware, same metrics,
    averaged over multiple trials.

34. **Q: What is the role of `seed_everything()`?**
    A: It seeds Python `random`, NumPy, and PyTorch before each
    trial so results are reproducible.

35. **Q: What are the 5 metrics?**
    A: peak GPU memory, CPU RSS, wall-clock latency, tokens per
    second, analytical KV cache size.

36. **Q: Why did you not measure accuracy?**
    A: We focused on memory and latency. Accuracy is
    dataset-dependent and would require a separate benchmark.

## F. Results

37. **Q: Which technique gives the smallest cache?**
    A: MiniKV (98.1% reduction vs FP16) followed by xKV
    (96.9%).

38. **Q: Which technique gives the fastest tokens/s?**
    A: GQA and PagedAttention lead in our runs.

39. **Q: Why does FP8 give 50% reduction but peak GPU memory
    drops by less?**
    A: Because model weights dominate peak GPU memory. The KV
    cache savings show up clearly only when the cache is large
    relative to the model (long contexts, large batches).

40. **Q: Does the cache scale linearly with context?**
    A: Yes. Eq. 2 predicts linear scaling, and our scaling
    table confirms it.

## G. Discussion

41. **Q: What is the central decision framework?**
    A: Pick the category that matches your bottleneck:
    quantization/compression for memory, attention-level for
    latency, memory management for throughput.

42. **Q: Can techniques stack?**
    A: Yes. xKV + 4-bit quantization gives 25.6× reduction.
    PagedAttention on top of FP8 is common in production.

43. **Q: Which technique has no per-step overhead?**
    A: GQA. It just changes the layout.

44. **Q: Which technique is lossless?**
    A: Memory management (PagedAttention, offloading, shared
    attention). They move the cache around without changing it.

## H. Limitations

45. **Q: What is the biggest limitation?**
    A: No end-task accuracy benchmark.

46. **Q: Why small models?**
    A: Our GPU has 2 GB. Frontier models need 24+ GB.

47. **Q: Why simulated kernels?**
    A: NVFP4, MiniKV, xKV kernels are proprietary.

48. **Q: Why no real vLLM?**
    A: vLLM is Linux-only. We wrote a PyTorch fallback.

49. **Q: What's the absolute numbers' portability?**
    A: They depend on hardware. The relative ordering is
    stable.

## I. Future work

50. **Q: What would you do next?**
    A: Run on a Linux + H100 machine with real kernels, add
    accuracy benchmarks (perplexity, MMLU, long-context
    retrieval), extend to 32k / 128k contexts, and benchmark
    multi-tenant prefix-sharing workloads.

---

## Tips for the viva

- **Speak slowly.** Don't rush.
- **Define the term first** before answering. "Q: What is the KV
  cache?" → "The KV cache is a memory buffer that stores..."
- **Use the equations.** Teachers love to see you write
  $KV_{\text{per token}} = 2 \cdot H \cdot L \cdot B \cdot D$.
- **Be honest about limitations.** If you don't know, say
  "I don't know, but here's how I would find out."
- **Connect back to the paper.** "This is in our Table V, which
  shows..."
- **Have the paper open** in case you need to point at a table.
