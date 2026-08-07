# KV Cache Optimization Benchmark Report

Generated on 2026-08-07 19:44:08

## Configuration

```yaml
seed: 42
trials: 1
warmup_steps: 2
max_new_tokens: 32
model: sshleifer/tiny-gpt2
mha_model: gpt2
gqa_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
use_cuda: true
dtype: float16
context_lengths:
- 256
- 1024
batch_sizes:
- 1
- 2
prompts:
- The history of computing began with analog machines and mechanical calculators.
- Modern transformer architectures rely on self-attention to model long-range dependencies.
- Key-value caching avoids the redundant recomputation of past token projections.
- Memory bandwidth often becomes the primary bottleneck during autoregressive decoding.
- Quantization reduces the precision of stored tensors to save memory at a small accuracy
  cost.
- PagedAttention partitions the KV cache into fixed-size blocks to reduce fragmentation.
- Grouped-query attention shares key and value heads across multiple query heads.
- Cross-layer compression exploits redundancy between consecutive transformer layers.
output_dir: results
save_raw: true
generate_plots: true
run_simulations: true

```


## Per-Technique Summary

| Technique          | Category              | Avg Peak GPU MB | Avg CPU RSS MB | Avg KV MB | Avg Latency (s) | Avg Tokens/s | Avg TTFT (s) | Notes                                                                                                                |
|--------------------|-----------------------|-----------------|----------------|-----------|-----------------|--------------|--------------|----------------------------------------------------------------------------------------------------------------------|
| NVFP4 (simulated)  | quantization          | 0.0             | 0.0            | 2.06      | 0.0             | 0.0          | 0.0          | NVFP4 (4-bit fp) KV cache simulation; halving FP8 memory. [analytical simulation]                                    |
| MiniKV (simulated) | quantization_eviction | 0.0             | 0.0            | 0.15      | 0.0             | 0.0          | 0.0          | MiniKV 2-bit + adaptive token eviction; reports >80% memory reduction in the original paper. [analytical simulation] |
| xKV (simulated)    | compression           | 0.0             | 0.0            | 0.26      | 0.0             | 0.0          | 0.0          | Cross-layer SVD KV cache compression, ~8× memory reduction in the original paper. [analytical simulation]            |


## Per-Cell Results

| Technique          | Category              | Context | Batch | Peak GPU MB | CPU RSS MB | KV Cache MB | Latency (s) | Tokens/s | TTFT (s) | Memory Saving (%) | Throughput Δ (%) | Notes                                                                                                                |
|--------------------|-----------------------|---------|-------|-------------|------------|-------------|-------------|----------|----------|-------------------|------------------|----------------------------------------------------------------------------------------------------------------------|
| NVFP4 (simulated)  | quantization          | 256     | 1     | 0.0         | 0.0        | 1.38        | 0.0         | 0.0      | 0.0      |                   |                  | NVFP4 (4-bit fp) KV cache simulation; halving FP8 memory. [analytical simulation]                                    |
| NVFP4 (simulated)  | quantization          | 256     | 2     | 0.0         | 0.0        | 2.75        | 0.0         | 0.0      | 0.0      |                   |                  | NVFP4 (4-bit fp) KV cache simulation; halving FP8 memory. [analytical simulation]                                    |
| NVFP4 (simulated)  | quantization          | 256     | 1     | 0.0         | 0.0        | 1.38        | 0.0         | 0.0      | 0.0      |                   |                  | NVFP4 (4-bit fp) KV cache simulation; halving FP8 memory. [analytical simulation]                                    |
| NVFP4 (simulated)  | quantization          | 256     | 2     | 0.0         | 0.0        | 2.75        | 0.0         | 0.0      | 0.0      |                   |                  | NVFP4 (4-bit fp) KV cache simulation; halving FP8 memory. [analytical simulation]                                    |
| MiniKV (simulated) | quantization_eviction | 256     | 1     | 0.0         | 0.0        | 0.1         | 0.0         | 0.0      | 0.0      |                   |                  | MiniKV 2-bit + adaptive token eviction; reports >80% memory reduction in the original paper. [analytical simulation] |
| MiniKV (simulated) | quantization_eviction | 256     | 2     | 0.0         | 0.0        | 0.21        | 0.0         | 0.0      | 0.0      |                   |                  | MiniKV 2-bit + adaptive token eviction; reports >80% memory reduction in the original paper. [analytical simulation] |
| MiniKV (simulated) | quantization_eviction | 256     | 1     | 0.0         | 0.0        | 0.1         | 0.0         | 0.0      | 0.0      |                   |                  | MiniKV 2-bit + adaptive token eviction; reports >80% memory reduction in the original paper. [analytical simulation] |
| MiniKV (simulated) | quantization_eviction | 256     | 2     | 0.0         | 0.0        | 0.21        | 0.0         | 0.0      | 0.0      |                   |                  | MiniKV 2-bit + adaptive token eviction; reports >80% memory reduction in the original paper. [analytical simulation] |
| xKV (simulated)    | compression           | 256     | 1     | 0.0         | 0.0        | 0.17        | 0.0         | 0.0      | 0.0      |                   |                  | Cross-layer SVD KV cache compression, ~8× memory reduction in the original paper. [analytical simulation]            |
| xKV (simulated)    | compression           | 256     | 2     | 0.0         | 0.0        | 0.34        | 0.0         | 0.0      | 0.0      |                   |                  | Cross-layer SVD KV cache compression, ~8× memory reduction in the original paper. [analytical simulation]            |
| xKV (simulated)    | compression           | 256     | 1     | 0.0         | 0.0        | 0.17        | 0.0         | 0.0      | 0.0      |                   |                  | Cross-layer SVD KV cache compression, ~8× memory reduction in the original paper. [analytical simulation]            |
| xKV (simulated)    | compression           | 256     | 2     | 0.0         | 0.0        | 0.34        | 0.0         | 0.0      | 0.0      |                   |                  | Cross-layer SVD KV cache compression, ~8× memory reduction in the original paper. [analytical simulation]            |


## Plots

- ![Memory vs Context](memory_vs_context.png)
- ![Throughput vs Context](throughput_vs_context.png)
- ![Latency vs Context](latency_vs_context.png)
- ![Memory vs Batch](memory_vs_batch.png)
- ![Summary Bar](summary_tokens_per_s.png)
