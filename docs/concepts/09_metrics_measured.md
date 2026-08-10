# 09 — Metrics We Measured

## Why we picked these metrics

A fair comparison needs the **same set of metrics** for every
technique. We picked five that cover the full picture: memory
(GPU, CPU, KV-specific), time (latency, throughput), and one
implicit metric for analytical reasoning (cache size from Eq. 1).

## The five metrics

### 1. `peak_gpu_mb` — peak GPU memory in MB

- **What:** the maximum GPU memory allocated during the trial.
- **How:** `torch.cuda.max_memory_allocated()` (CUDA) or
  `psutil.Process().memory_info().rss` (CPU fallback).
- **Units:** megabytes.
- **Why it matters:** the dominant constraint on most GPUs.

### 2. `cpu_rss_mb` — CPU resident set size in MB

- **What:** how much host RAM the Python process is using.
- **How:** `psutil.Process().memory_info().rss`.
- **Units:** megabytes.
- **Why it matters:** for offloading techniques, the cache lives
  on CPU. For others, it tells you how much pre-processing is
  using.

### 3. `wall_clock_s` — wall-clock latency in seconds

- **What:** total time from `setup()` to `run_trial()` completion.
- **How:** `time.perf_counter()` deltas.
- **Units:** seconds.
- **Why it matters:** what the user actually feels.

### 4. `tokens_per_s` — throughput in tokens per second

- **What:** number of generated tokens / total decode time.
- **How:** derived from `wall_clock_s` and the number of tokens
  the model emitted.
- **Units:** tokens / second.
- **Why it matters:** how many requests per second a serving
  system can handle.

### 5. `analytic_kv_mb` — analytical KV cache size in MB

- **What:** the size of the KV cache derived from Eq. 1 and the
  technique's parameters.
- **How:** a direct formula, not a measurement.
- **Units:** megabytes.
- **Why it matters:** this is the **theoretical** number, not
  polluted by the model weights, the allocator, or the kernel
  overhead. It is what the taxonomy predicts.

## How we put them together

Every `BenchmarkResult` is a dict with these five values for one
(technique, ctx, batch, trial) combination. The runner writes
one CSV per combination and one aggregated CSV per technique.

## Important: TTFT vs average latency

We compute `tokens_per_s` as **total tokens / total wall-clock**,
which is closer to **steady-state throughput** than to **TTFT
(Time To First Token)**.

In the paper we acknowledge that `wall_clock_s / n_tokens` is a
**pessimistic** estimate of TTFT when:
- the context is large (most of the time is in prefill, not in
  the per-token decode), and
- the model is freshly loaded (no warm cache).

Production servers pre-warm KV caches and use speculative decoding
to amortize TTFT. Our framework reports the cold-start number,
which is conservative.

## What we did NOT measure

- **End-task accuracy** — perplexity, MMLU, long-context retrieval
  accuracy. We deliberately did not include these because the
  framework needs ground-truth labels and calibrated datasets, and
  none of the techniques are benchmarked on accuracy in the same
  way in the literature. We acknowledge this as a limitation.
- **Energy consumption** — could be derived from GPU power draw
  but depends on the specific GPU's TDP and is not a focus of the
  paper.
- **Multi-tenant fairness** — we serve one request at a time (or a
  fixed batch). We did not model request-arrival patterns.

## Common viva questions

1. **Q: Why did you measure five metrics and not just one?**
   A: Because "KV cache optimization" means different things
   depending on the bottleneck. A technique that wins on memory
   might lose on latency. We need all five to tell the full story.

2. **Q: Why is `analytic_kv_mb` different from `peak_gpu_mb`?**
   A: `analytic_kv_mb` is the KV cache size from the formula.
   `peak_gpu_mb` is the actual GPU memory **including the model
   weights, the activations, and the allocator overhead**. In
   our experiments, the model weights dominate, so the ratio
   between the two metrics is small.

3. **Q: What is a "good" tokens/s?**
   A: That depends on the model size and the hardware. For our
   `tiny-gpt2` reference, we get tens of tokens per second on
   CPU and hundreds on GPU. The relative ordering between
   techniques is what matters.

4. **Q: Did you measure accuracy?**
   A: No. We acknowledge this as a limitation. Accuracy is
   dataset-dependent and would require a separate benchmark.

5. **Q: Why use `psutil` for CPU memory?**
   A: Because `torch.cuda.max_memory_allocated()` only works on
   CUDA. For CPU runs and for the offloading path, we need a
   cross-platform way to read the process's RSS.
