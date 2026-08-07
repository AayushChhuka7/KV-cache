# Limitations

This document describes what `kvbench` does and does *not* measure, and
why. It is intentionally honest so that the resulting tables are not
over-interpreted.

## 1. Hardware differences

Different hardware (A100 vs RTX 4090 vs CPU) will produce very different
absolute latency and throughput numbers. The framework reports *all*
metrics so the relative ranking of techniques is meaningful even when
absolute numbers are platform-specific.

* **CUDA** — all GPU techniques run on CUDA when available.
* **CPU** — without CUDA the same code paths run on CPU but at much
  lower throughput. Memory measurements remain meaningful.

## 2. PagedAttention requires vLLM

`vLLM < 0.4.3` is not officially supported on Windows or on
non-NVIDIA GPUs. When vLLM is unavailable:

* `PagedAttentionVLLM` falls back to a pure-PyTorch implementation
  that uses page-sized `list[tensor]` storage instead of vLLM's
  block manager. The qualitative behaviour (no per-sequence
  over-allocation, lower peak memory) is preserved, but absolute
  throughput is much lower.
* On Windows, install WSL2 and run the framework inside Linux to get
  full PagedAttention.

## 3. NVFP4, MiniKV, xKV are simulated

These techniques depend on:

| Technique | What's missing                              | How it's simulated                                                                |
|-----------|---------------------------------------------|-----------------------------------------------------------------------------------|
| NVFP4     | TensorRT-LLM / Transformer Engine kernels   | 0.5-byte analytic memory formula + ≈ 5% latency overhead                          |
| MiniKV    | Custom CUDA kernels + 2-bit quant + eviction| 0.25-byte × 15% token keep × ≈ 50% per-token decode overhead                     |
| xKV       | Offline SVD pre-processing of weights       | cache memory divided by `layer_share_k × compression_ratio` ≈ 8× + ≈ 2% overhead   |

The simulators are **deliberately conservative**: they report the
*analytical* memory savings and an *estimated* latency overhead, not a
measured number. The latency columns are zero for these rows. They are
clearly tagged `[analytical simulation]` in the output.

To turn a simulator into a real benchmark, you would replace its
`run_trial()` body with a call into the proprietary kernel stack and
update the `analytic_kv_mb()` formula accordingly.

## 4. Time-to-first-token (TTFT)

`TTFT` is approximated as `wall_time_s / n_tokens`. This is an
*over-estimate* on long contexts (the first token is emitted near the
start of the prefill, not after all decoding finishes) and a
*reasonable estimate* for the average case. If you need a precise
TTFT, replace the wall-clock measurement in `_hf_generate()` with a
`TextStreamer` callback.

## 5. Model availability

`load_model_and_tokenizer()` falls back to `gpt2` if the requested
model cannot be downloaded. This is intentional for offline test
environments, but if you want reproducible results make sure
`KVBENCH_MODEL` points to a model that is downloadable in your
environment.

## 6. Quantization fidelity

* **INT8** uses per-channel symmetric quantization, which is close to
  the scheme used in production KV-cache quantization papers but does
  not include outlier handling or asymmetric ranges.
* **FP8** uses `torch.float8_e5m2` when supported, else the bytes are
  stored as `uint8` and only upcast during attention. On hardware that
  lacks FP8 tensor cores (most consumer GPUs), the dequantization
  overhead is the dominant cost.

## 7. Low-rank reconstruction

`LowRankCompression` initializes random projectors (Kaiming-uniform).
It does *not* learn them via a calibration pass, so the resulting
accuracy is poor — but the memory and latency trade-off is exactly
the one described in the paper. To compare reconstruction quality,
swap the random init for an SVD-based fit on a calibration set.

## 8. Determinism

PyTorch operations that are not deterministic on CUDA (e.g. flash
attention reductions) are not enforced here. Setting
`KVBENCH_LOG_LEVEL=DEBUG` and re-running with `deterministic=True`
inside `seed_everything()` will make numbers more reproducible at a
small throughput cost. The unit tests do not exercise CUDA, so they
are deterministic on every machine.

## 9. Shared-prefix savings not measured

vLLM's prefix-sharing is a major source of its throughput advantage
in multi-tenant serving. The PagedAttention simulation here does not
implement prefix-sharing; it only eliminates over-allocation. If you
need that, install vLLM and run on a multi-tenant workload.

## 10. Cold-start vs warm-cache

Each trial starts from a freshly loaded model and a freshly reset
allocator. Real serving systems often see warm KV caches that are
already populated from prior requests. The framework does not model
warm-cache effects; it measures the cold path, which is the
*worst-case* scenario for memory and a *typical* scenario for latency.

---

If you discover a new limitation, please open an issue and document
it here.
