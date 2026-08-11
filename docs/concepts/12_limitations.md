# 12 — Limitations

This is the section where the paper is honest about what it
**did not** do. Every study has limits. Teachers love to ask
about them — it shows you understand the boundary of your work.

## 1. Dataset limitations

`kvbench` uses a **fixed set of eight short prompts** to fill the
context window. These prompts are repeated and combined to reach
the desired context length. The framework measures **memory and
latency** with controlled inputs, not **accuracy** on a downstream
benchmark.

- **What we did not measure:** perplexity, MMLU, long-context
  retrieval accuracy, BLEU, ROUGE.
- **Why:** accuracy is dataset-dependent and requires a separate
  benchmark with calibrated ground-truth labels. None of the
  techniques in the literature report accuracy in the same
  experiments, so direct comparison would be unfair.
- **Impact:** the paper cannot say "technique X is best for
  accuracy". We can only say "technique X preserves the most
  accuracy in the paper that introduced it".

## 2. Computational / hardware constraints

The absolute numbers in Table V depend on the **specific
hardware**. Different GPUs (A100, RTX 4090, MX230, CPU) will
produce different absolute latency and throughput.

- **What we did not control for:** GPU model, CPU model, RAM
  speed, PCIe bus speed.
- **What we did:** seeded the RNG, averaged over multiple trials,
  used the same hardware for every technique.
- **Impact:** the **relative ordering** of techniques is stable
  across hardware. The **absolute numbers** are not.

The CPU path is much slower than the CUDA path. The paper
compares within each platform only.

## 3. Model limitations

We used **small models**: `sshleifer/tiny-gpt2` for the baseline,
`TinyLlama-1.1B-Chat-v1.0` for GQA, `gpt2` for the MHA comparison.

- **Why small:** the test machine has only 2 GB of GPU memory.
  A 7B model at FP16 takes ~14 GB.
- **Impact:** the framework cannot replicate the exact throughput
  performance of frontier models like Llama-3-70B or Mistral at
  100k context. The **ordering** of techniques is preserved, but
  the absolute tokens/s is not.
- **Why this also bounds the accuracy benchmark:** the GPU on the
  development machine is a 2 GB NVIDIA MX230. A 7B model at FP16
  needs ~14 GB just for the weights, so we cannot host the
  checkpoints on which accuracy numbers (perplexity, MMLU, long-
  context retrieval) are reported in the original KV-cache
  papers. This is why end-task accuracy is **reported from the
  literature**, not measured head-to-head by `kvbench`.

## 4. Evaluation limitations

- **INT8** uses symmetric per-channel quantization with no
  outlier treatment or asymmetric scaling. Production-grade INT8
  (e.g. SmoothQuant, ZeroQuant) has more elaborate schemes.
- **FP8** uses `torch.float8_e5m2` when available; otherwise a
  1-byte approximation. On devices without FP8 tensor cores,
  dequantization cost becomes the bottleneck.
- **Low-rank compression** uses random Kaiming-uniform projections
  rather than calibrated SVD. This produces worse reconstruction
  than the literature, but the memory-latency trade-off is the
  same.

## 5. Simulated techniques

NVFP4, MiniKV, and xKV are **simulated** because their GPU
kernels are proprietary or one-shot preprocessing is not
pip-installable.

- **What this means:** the simulators give **analytical estimates**
  of memory savings and latency overhead based on the published
  numbers — not measured values.
- **What we did:** every simulated row is marked "(simulated)" in
  the tables and uses a distinct marker in the plots.
- **Impact:** the latency entries for these three rows are
  estimated, not measured. The memory entries are the formula's
  prediction, not a measurement.

## 6. PagedAttention requires vLLM

vLLM is the real PagedAttention implementation. It is **Linux
only** and works best on NVIDIA GPUs. On Windows, we fall back
to a pure-PyTorch simulation that has the same qualitative
behavior (no over-allocation per sequence) but is much slower.

- **Impact:** the PagedAttention row in our tables is
  representative of the **memory** behavior but not the
  **performance** behavior of real vLLM.
- **Recommendation:** run the framework under WSL2 or Linux for
  full PagedAttention results.

## 7. Inference speed estimation

We compute `tokens_per_s` as `tokens / wall_clock_s`. This is a
**pessimistic estimator** of TTFT when:

- the context is large (most of the time is in prefill, not in
  per-token decode),
- the cache is cold (freshly loaded model, freshly allocated
  cache).

Production servers use:
- **Speculative decoding** to amortize TTFT.
- **KV cache pre-warming** between requests.
- **Continuous batching** to overlap requests.

None of these are in our framework.

## 8. Generalization and scalability

The default sweep is:

- context lengths: $[128, 512, 2048, 4096]$
- batch sizes: $[1, 2, 4]$

These are **modest** sizes chosen to fit on commodity hardware.
We did not sweep to 32k, 64k, or 128k contexts. We did not
benchmark multi-tenant prefix-sharing workloads.

- **Impact:** the framework's findings are **indicative** for
  small and medium contexts. For very long contexts, the
  ordering should be the same but the absolute numbers will
  differ.

## 9. Determinism

Non-deterministic CUDA kernels (notably FlashAttention's
reduction) are not made deterministic in the default mode. To
get fully deterministic numbers, run with
`KVBENCH_LOG_LEVEL=DEBUG` and `deterministic=True` inside
`seed_everything()`. This costs a small amount of throughput.

## 10. End-task accuracy

The framework does **not** measure end-task accuracy. This is
the **biggest** omission. Future work should add:

- **Perplexity** on WikiText-103 or C4.
- **MMLU** for multiple-choice reasoning.
- **Long-context retrieval** (e.g. Needle-in-a-Haystack) to
  measure how compression affects long-context recall.
- **Human evaluation** for chat-quality tasks.

## 11. What we did not do

- We did not **retrain** any model (everything is
  post-training or architecture-pretrained).
- We did not **optimize** the kernels (used PyTorch /
  HuggingFace defaults).
- We did not **benchmark on a cluster** (single machine only).
- We did not **measure energy consumption**.

## Common viva questions

1. **Q: What is the biggest limitation of your paper?**
   A: No end-task accuracy benchmark. We measure memory and
   latency, not quality.

2. **Q: Why did you not use a bigger model?**
   A: Hardware. Our GPU has 2 GB; frontier models need 24+ GB.

3. **Q: If you had more time, what would you add?**
   A: Real NVFP4 / MiniKV / xKV kernels on a Linux + H100
   machine, plus an accuracy benchmark.

4. **Q: Could your taxonomy be wrong?**
   A: Possibly. Some techniques (MiniKV, xKV) span two
   categories. We acknowledge this. The taxonomy is a heuristic,
   not a theorem.

5. **Q: What would change if you ran on a bigger GPU?**
   A: The absolute numbers would change. The relative ordering
   of techniques should not.
