# 14 — Glossary

One-line definitions for every term used in the paper. Keep this
file open while you revise.

## A

- **Attention** — a weighted sum of value vectors weighted by
  softmax-normalized dot products of queries and keys.
- **Attention-level redesign** — a category of KV cache
  optimization that changes the transformer's structure (e.g. GQA).
- **Autoregressive** — generating tokens one at a time, each
  depending on all previous tokens.

## B

- **bitsandbytes** — a Python library for INT8 quantization
  (Linux-friendly).
- **Bottleneck** — the slowest or most-constrained part of the
  inference pipeline (GPU memory, memory bandwidth, latency,
  throughput).

## C

- **Cache** — a memory buffer that stores results for reuse.
- **Cold start** — the first request after a model is loaded.
- **Compression** — a category of KV cache optimization that
  changes the representation (e.g. low-rank, xKV).
- **CUDA** — NVIDIA's API for programming GPUs.
- **CPU offloading** — moving K/V vectors from GPU to CPU RAM.

## D

- **Dequantization** — converting a low-precision number back to
  a higher precision (e.g. INT8 to FP16).
- **Determinism** — the property that the same inputs always
  produce the same outputs.

## E

- **E5M2** — a 5-bit-exponent, 2-bit-mantissa FP8 format.
- **Embedding** — a learned vector representation of a token.
- **End-task accuracy** — quality on a downstream task (MMLU,
  perplexity, etc.).

## F

- **FlashAttention** — a fast, memory-efficient attention
  algorithm that fuses multiple operations.
- **FP4** — 4-bit floating-point.
- **FP8** — 8-bit floating-point.
- **FP16** — 16-bit floating-point (standard).
- **Fragmentation** — wasted memory due to non-contiguous
  allocation.

## G

- **GPU** — Graphics Processing Unit. The hardware that runs
  LLM inference.
- **GQA** — Grouped-Query Attention. Multiple query heads share
  one K/V head.

## H

- **HuggingFace `transformers`** — a Python library for using
  pretrained models.
- **Head** — one parallel attention computation in a
  multi-head transformer.

## I

- **INT8** — 8-bit integer.
- **Internal fragmentation** — wasted memory inside a reserved
  block.

## K

- **KV cache** — Key-Value cache. Stores K and V vectors of past
  tokens to avoid recomputation.
- **kvbench** — our benchmarking framework for KV cache
  optimization techniques.
- **Key (K)** — a learned vector that identifies a token in
  attention.

## L

- **Latency** — time per request.
- **Low-rank compression** — projecting K/V into a smaller
  rank-$r$ subspace.

## M

- **MHA** — Multi-Head Attention. Each query head has its own
  K/V head.
- **MQA** — Multi-Query Attention. All query heads share one K/V
  head.
- **MixQuant** — mixed-precision quantization.
- **Memory bandwidth** — the rate at which the GPU can read or
  write memory.
- **Memory management** — a category of KV cache optimization
  that changes how the cache is stored (e.g. PagedAttention).
- **MiniKV** — 2-bit quantization + adaptive token eviction.
- **Mixed precision** — using different precisions for different
  tensors.

## N

- **NVFP4** — NVIDIA's 4-bit floating-point format for KV cache.
- **NumPy** — Python numerical library.

## O

- **Outlier** — a value far from the bulk of the distribution,
  which can hurt quantization accuracy.

## P

- **PagedAttention** — page-based KV cache allocation, like OS
  virtual memory.
- **PCIe** — the bus that connects the GPU to the CPU.
- **Perplexity** — a measure of how well a model predicts text.
- **Pre-fill** — the initial processing of the input prompt.
- **Pinned memory** — CPU memory that the GPU can DMA to/from
  efficiently.
- **Precision** — the number of bits used to represent a number.
- **Projection** — a matrix multiplication that maps a vector to
  a new space (e.g. Q, K, V projections).
- **PyTorch** — Python deep-learning library.

## Q

- **Quantization** — a category of KV cache optimization that
  reduces the bits per element (e.g. FP8, NVFP4).
- **Query (Q)** — the vector that asks "which tokens should I
  attend to?".

## R

- **RSS** — Resident Set Size. The amount of RAM a process is
  using.
- **Reconstruction** — converting a compressed representation back
  to the original.

## S

- **Shared Attention** — reusing K/V cache for shared prefixes
  across requests.
- **Simulation** — an analytical estimate, not a measurement.
- **SVD** — Singular Value Decomposition. Any matrix $M$ can be
  written as $U \Sigma V^\top$.
- **SM** — Streaming Multiprocessor. NVIDIA's GPU compute unit.

## T

- **Tensor cores** — NVIDIA GPU units specialized for
  low-precision matrix multiplication.
- **Throughput** — tokens per second, or requests per second.
- **TTFT** — Time To First Token. Latency until the first
  generated token appears.
- **Tokenizer** — converts text to token IDs.

## V

- **vLLM** — a high-performance LLM serving system with
  PagedAttention. Linux only.
- **Value (V)** — the vector that carries the token's content in
  attention.
- **VRAM** — video RAM on the GPU.

## W

- **Wall-clock time** — real elapsed time, as measured by a
  clock.
- **Warm cache** — a cache that has been pre-filled, so the
  first request is fast.

## X

- **xKV** — Cross-Layer KV Cache Compression via SVD.

## Z

- **ZeroQuant** — a quantization technique with outlier
  handling.
