# Paper Figures — Complete Data & Reproduction Spec

This file contains **all measured data** behind Figures 2-4 of the paper
(`main.tex`) and the exact specification needed to regenerate the six figures
correctly. Everything below is generated from the real sweep at
`coding/results/paper_sweep/` (per-cell `results.csv`, raw JSON `all.json`).
No synthetic numbers.

- **Hardware**: NVIDIA GeForce GTX 1650, 4 GB (sm_75), Intel Core i5-11300H,
  16 GB RAM, Windows 11, PyTorch 2.11.0+cu128 (CUDA 12.8), transformers 5.15.0.
- **Sweep**: 10 techniques x ctx [128, 512, 2048, 4096] x batch [1, 2, 4] x 3
  trials (median not used; cells below are the 3-trial averages).
- **GQA cells NOT measured** (CUDA OOM on 4 GB): ctx=4096, batch=2 and
  ctx=4096, batch=4. They are absent from the CSV — do NOT invent them.
- **Simulated techniques** (NVFP4, MiniKV, xKV): analytic-only. Their
  `Peak GPU MB`, `Latency (s)`, `Tokens/s`, `TTFT (s)` are 0 because no GPU
  kernel runs. They appear ONLY in the KV-size figure/panels.
- **PagedAttention (vLLM)** ran its pure-PyTorch CPU fallback (note contains
  `[CPU simulation]`) — treat it as simulated in styling (x marker).

---

## 1. Canonical data — `results.csv` (verbatim, 118 rows)

Columns: Technique,Category,Context,Batch,Peak GPU MB,CPU RSS MB,KV Cache MB,Latency (s),Tokens/s,TTFT (s),Memory Saving (%),Throughput Δ (%),Notes

```csv
Baseline FP16,baseline,128,1,9.8,1314.3,0.0,0.364,148.95,0.011,0.0,0.0,HuggingFace default KV cache in FP16/FP32. No optimization.
Baseline FP16,baseline,128,2,10.3,1320.6,0.0,0.161,199.74,0.005,0.0,0.0,HuggingFace default KV cache in FP16/FP32. No optimization.
Baseline FP16,baseline,128,4,11.3,1325.9,0.01,0.165,198.97,0.005,0.0,0.0,HuggingFace default KV cache in FP16/FP32. No optimization.
Baseline FP16,baseline,512,1,14.9,1326.9,0.01,0.15,213.45,0.005,0.0,0.0,HuggingFace default KV cache in FP16/FP32. No optimization.
Baseline FP16,baseline,512,2,19.5,1327.4,0.02,0.164,195.42,0.005,0.0,0.0,HuggingFace default KV cache in FP16/FP32. No optimization.
Baseline FP16,baseline,512,4,28.6,1328.6,0.03,0.171,187.65,0.005,0.0,0.0,HuggingFace default KV cache in FP16/FP32. No optimization.
Baseline FP16,baseline,2048,1,97.6,1329.1,0.03,0.157,203.75,0.005,0.0,0.0,HuggingFace default KV cache in FP16/FP32. No optimization.
Baseline FP16,baseline,2048,2,169.8,1330.7,0.06,0.169,189.55,0.005,0.0,0.0,HuggingFace default KV cache in FP16/FP32. No optimization.
Baseline FP16,baseline,2048,4,314.2,1333.7,0.12,0.195,165.02,0.006,0.0,0.0,HuggingFace default KV cache in FP16/FP32. No optimization.
Baseline FP16,baseline,4096,1,361.8,1333.8,0.06,0.201,159.56,0.006,0.0,0.0,HuggingFace default KV cache in FP16/FP32. No optimization.
Baseline FP16,baseline,4096,2,650.2,1335.2,0.12,0.227,142.07,0.007,0.0,0.0,HuggingFace default KV cache in FP16/FP32. No optimization.
Baseline FP16,baseline,4096,4,1227.1,1340.2,0.25,0.283,113.19,0.008,0.0,0.0,HuggingFace default KV cache in FP16/FP32. No optimization.
Quantized INT8,quantization,128,1,21.7,1370.0,0.0,0.241,134.77,0.008,50.0,-9.5,KV cache quantized to INT8 with per-channel symmetric quantization. (round-trip int8 hooks: k_proj/v_proj or fused c_attn)
Quantized INT8,quantization,128,2,34.1,1370.3,0.0,0.219,146.42,0.007,50.0,-26.7,KV cache quantized to INT8 with per-channel symmetric quantization. (round-trip int8 hooks: k_proj/v_proj or fused c_attn)
Quantized INT8,quantization,128,4,59.7,1370.7,0.0,0.212,151.2,0.007,50.0,-24.0,KV cache quantized to INT8 with per-channel symmetric quantization. (round-trip int8 hooks: k_proj/v_proj or fused c_attn)
Quantized INT8,quantization,512,1,59.5,1370.7,0.0,0.212,151.55,0.007,50.0,-29.0,KV cache quantized to INT8 with per-channel symmetric quantization. (round-trip int8 hooks: k_proj/v_proj or fused c_attn)
Quantized INT8,quantization,512,2,107.7,1370.7,0.01,0.22,145.65,0.007,50.0,-25.5,KV cache quantized to INT8 with per-channel symmetric quantization. (round-trip int8 hooks: k_proj/v_proj or fused c_attn)
Quantized INT8,quantization,512,4,206.1,1371.2,0.02,0.24,134.5,0.007,50.0,-28.3,KV cache quantized to INT8 with per-channel symmetric quantization. (round-trip int8 hooks: k_proj/v_proj or fused c_attn)
Quantized INT8,quantization,2048,1,205.8,1371.6,0.02,0.217,148.03,0.007,50.0,-27.3,KV cache quantized to INT8 with per-channel symmetric quantization. (round-trip int8 hooks: k_proj/v_proj or fused c_attn)
Quantized INT8,quantization,2048,2,402.3,1372.3,0.03,0.226,141.86,0.007,50.0,-25.2,KV cache quantized to INT8 with per-channel symmetric quantization. (round-trip int8 hooks: k_proj/v_proj or fused c_attn)
Quantized INT8,quantization,2048,4,796.0,1373.6,0.06,0.267,120.34,0.008,50.0,-27.1,KV cache quantized to INT8 with per-channel symmetric quantization. (round-trip int8 hooks: k_proj/v_proj or fused c_attn)
Quantized INT8,quantization,4096,1,402.2,1373.6,0.03,0.238,134.78,0.007,50.0,-15.5,KV cache quantized to INT8 with per-channel symmetric quantization. (round-trip int8 hooks: k_proj/v_proj or fused c_attn)
Quantized INT8,quantization,4096,2,795.8,1374.8,0.06,0.272,118.03,0.009,50.0,-16.9,KV cache quantized to INT8 with per-channel symmetric quantization. (round-trip int8 hooks: k_proj/v_proj or fused c_attn)
Quantized INT8,quantization,4096,4,1580.8,1378.7,0.12,0.345,92.91,0.011,50.0,-17.9,KV cache quantized to INT8 with per-channel symmetric quantization. (round-trip int8 hooks: k_proj/v_proj or fused c_attn)
Quantized FP8 (E5M2),quantization,128,1,21.7,1401.9,0.0,0.285,112.84,0.009,50.0,-24.2,KV cache quantized to FP8 E5M2 (1 byte) using PyTorch fp8 dtype where supported, else 1-byte approximation. (round-trip fp8 hooks: k_proj/v_proj or fused c_attn)
Quantized FP8 (E5M2),quantization,128,2,34.1,1402.4,0.0,0.23,138.85,0.007,50.0,-30.5,KV cache quantized to FP8 E5M2 (1 byte) using PyTorch fp8 dtype where supported, else 1-byte approximation. (round-trip fp8 hooks: k_proj/v_proj or fused c_attn)
Quantized FP8 (E5M2),quantization,128,4,59.7,1402.6,0.0,0.268,122.91,0.008,50.0,-38.2,KV cache quantized to FP8 E5M2 (1 byte) using PyTorch fp8 dtype where supported, else 1-byte approximation. (round-trip fp8 hooks: k_proj/v_proj or fused c_attn)
Quantized FP8 (E5M2),quantization,512,1,59.5,1402.6,0.0,0.26,125.98,0.008,50.0,-41.0,KV cache quantized to FP8 E5M2 (1 byte) using PyTorch fp8 dtype where supported, else 1-byte approximation. (round-trip fp8 hooks: k_proj/v_proj or fused c_attn)
Quantized FP8 (E5M2),quantization,512,2,107.7,1402.7,0.01,0.254,126.5,0.008,50.0,-35.3,KV cache quantized to FP8 E5M2 (1 byte) using PyTorch fp8 dtype where supported, else 1-byte approximation. (round-trip fp8 hooks: k_proj/v_proj or fused c_attn)
Quantized FP8 (E5M2),quantization,512,4,206.1,1402.7,0.02,0.259,124.2,0.008,50.0,-33.8,KV cache quantized to FP8 E5M2 (1 byte) using PyTorch fp8 dtype where supported, else 1-byte approximation. (round-trip fp8 hooks: k_proj/v_proj or fused c_attn)
Quantized FP8 (E5M2),quantization,2048,1,205.8,1403.0,0.02,0.259,125.2,0.008,50.0,-38.6,KV cache quantized to FP8 E5M2 (1 byte) using PyTorch fp8 dtype where supported, else 1-byte approximation. (round-trip fp8 hooks: k_proj/v_proj or fused c_attn)
Quantized FP8 (E5M2),quantization,2048,2,402.3,1403.8,0.03,0.28,114.6,0.009,50.0,-39.5,KV cache quantized to FP8 E5M2 (1 byte) using PyTorch fp8 dtype where supported, else 1-byte approximation. (round-trip fp8 hooks: k_proj/v_proj or fused c_attn)
Quantized FP8 (E5M2),quantization,2048,4,796.0,1405.4,0.06,0.31,103.3,0.01,50.0,-37.4,KV cache quantized to FP8 E5M2 (1 byte) using PyTorch fp8 dtype where supported, else 1-byte approximation. (round-trip fp8 hooks: k_proj/v_proj or fused c_attn)
Quantized FP8 (E5M2),quantization,4096,1,402.2,1405.6,0.03,0.272,117.76,0.008,50.0,-26.2,KV cache quantized to FP8 E5M2 (1 byte) using PyTorch fp8 dtype where supported, else 1-byte approximation. (round-trip fp8 hooks: k_proj/v_proj or fused c_attn)
Quantized FP8 (E5M2),quantization,4096,2,795.8,1406.7,0.06,0.313,102.28,0.01,50.0,-28.0,KV cache quantized to FP8 E5M2 (1 byte) using PyTorch fp8 dtype where supported, else 1-byte approximation. (round-trip fp8 hooks: k_proj/v_proj or fused c_attn)
Quantized FP8 (E5M2),quantization,4096,4,1580.8,1409.7,0.12,0.361,89.06,0.011,50.0,-21.3,KV cache quantized to FP8 E5M2 (1 byte) using PyTorch fp8 dtype where supported, else 1-byte approximation. (round-trip fp8 hooks: k_proj/v_proj or fused c_attn)
GQA,attention,128,1,2124.9,3575.2,2.75,4.664,6.86,0.146,-140700.0,-95.4,GQA: 32 query heads sharing 4 KV heads (group size 8); native HuggingFace model.
GQA,attention,128,2,2139.5,3575.5,5.5,5.872,5.45,0.183,-140700.0,-97.3,GQA: 32 query heads sharing 4 KV heads (group size 8); native HuggingFace model.
GQA,attention,128,4,2164.1,3581.7,11.0,7.475,4.28,0.233,-140700.0,-97.8,GQA: 32 query heads sharing 4 KV heads (group size 8); native HuggingFace model.
GQA,attention,512,1,2219.7,3582.5,11.0,7.441,4.3,0.232,-140700.0,-98.0,GQA: 32 query heads sharing 4 KV heads (group size 8); native HuggingFace model.
GQA,attention,512,2,2329.2,3582.1,22.0,10.643,3.01,0.332,-140700.0,-98.5,GQA: 32 query heads sharing 4 KV heads (group size 8); native HuggingFace model.
GQA,attention,512,4,2562.0,3463.3,44.0,16.976,1.88,0.53,-140700.0,-99.0,GQA: 32 query heads sharing 4 KV heads (group size 8); native HuggingFace model.
GQA,attention,2048,1,3440.0,3464.1,44.0,18.03,1.77,0.563,-140700.0,-99.1,GQA: 32 query heads sharing 4 KV heads (group size 8); native HuggingFace model.
GQA,attention,2048,2,4723.0,4841.3,88.0,126.668,0.25,3.958,-140700.0,-99.9,GQA: 32 query heads sharing 4 KV heads (group size 8); native HuggingFace model.
GQA,attention,2048,4,7318.5,6493.2,176.0,345.707,0.09,10.803,-140700.0,-99.9,GQA: 32 query heads sharing 4 KV heads (group size 8); native HuggingFace model.
GQA,attention,4096,1,7074.9,4262.0,88.0,237.732,0.13,7.429,-140700.0,-99.9,GQA: 32 query heads sharing 4 KV heads (group size 8); native HuggingFace model.
Low-Rank Compression,compression,128,1,21.7,393.4,0.0,0.212,151.24,0.007,0.0,1.5,KV cache projected to a low-rank subspace of dimension `rank`, reconstructed at attention time. (rank=64, head_dim=1)
Low-Rank Compression,compression,128,2,34.1,393.9,0.0,0.176,182.41,0.006,0.0,-8.7,KV cache projected to a low-rank subspace of dimension `rank`, reconstructed at attention time. (rank=64, head_dim=1)
Low-Rank Compression,compression,128,4,59.8,395.1,0.01,0.174,184.84,0.005,0.0,-7.1,KV cache projected to a low-rank subspace of dimension `rank`, reconstructed at attention time. (rank=64, head_dim=1)
Low-Rank Compression,compression,512,1,59.5,395.4,0.01,0.181,176.74,0.006,0.0,-17.2,KV cache projected to a low-rank subspace of dimension `rank`, reconstructed at attention time. (rank=64, head_dim=1)
Low-Rank Compression,compression,512,2,107.7,395.6,0.02,0.228,141.92,0.007,0.0,-27.4,KV cache projected to a low-rank subspace of dimension `rank`, reconstructed at attention time. (rank=64, head_dim=1)
Low-Rank Compression,compression,512,4,206.1,396.0,0.03,0.225,142.32,0.007,0.0,-24.2,KV cache projected to a low-rank subspace of dimension `rank`, reconstructed at attention time. (rank=64, head_dim=1)
Low-Rank Compression,compression,2048,1,205.8,397.0,0.03,0.226,157.02,0.007,0.0,-22.9,KV cache projected to a low-rank subspace of dimension `rank`, reconstructed at attention time. (rank=64, head_dim=1)
Low-Rank Compression,compression,2048,2,402.3,396.4,0.06,0.184,175.45,0.006,0.0,-7.4,KV cache projected to a low-rank subspace of dimension `rank`, reconstructed at attention time. (rank=64, head_dim=1)
Low-Rank Compression,compression,2048,4,796.0,397.7,0.12,0.176,182.28,0.006,0.0,10.5,KV cache projected to a low-rank subspace of dimension `rank`, reconstructed at attention time. (rank=64, head_dim=1)
Low-Rank Compression,compression,4096,1,402.2,398.0,0.06,0.185,174.54,0.006,0.0,9.4,KV cache projected to a low-rank subspace of dimension `rank`, reconstructed at attention time. (rank=64, head_dim=1)
Low-Rank Compression,compression,4096,2,795.8,398.3,0.12,0.233,138.96,0.007,0.0,-2.2,KV cache projected to a low-rank subspace of dimension `rank`, reconstructed at attention time. (rank=64, head_dim=1)
Low-Rank Compression,compression,4096,4,1580.8,401.2,0.25,0.284,113.1,0.009,0.0,-0.1,KV cache projected to a low-rank subspace of dimension `rank`, reconstructed at attention time. (rank=64, head_dim=1)
PagedAttention (vLLM),memory_management,128,1,21.7,403.3,0.0,0.17,191.57,0.005,65.2,28.6,PagedAttention via vLLM with on-demand block allocation and prefix sharing. [CPU simulation]
PagedAttention (vLLM),memory_management,128,2,34.1,403.2,0.0,0.143,224.9,0.004,65.2,12.6,PagedAttention via vLLM with on-demand block allocation and prefix sharing. [CPU simulation]
PagedAttention (vLLM),memory_management,128,4,59.7,403.3,0.0,0.14,230.37,0.004,65.2,15.8,PagedAttention via vLLM with on-demand block allocation and prefix sharing. [CPU simulation]
PagedAttention (vLLM),memory_management,512,1,59.5,403.3,0.01,0.145,221.39,0.005,31.9,3.7,PagedAttention via vLLM with on-demand block allocation and prefix sharing. [CPU simulation]
PagedAttention (vLLM),memory_management,512,2,107.7,403.3,0.01,0.158,202.49,0.005,31.9,3.6,PagedAttention via vLLM with on-demand block allocation and prefix sharing. [CPU simulation]
PagedAttention (vLLM),memory_management,512,4,206.1,403.4,0.02,0.189,169.2,0.006,31.9,-9.8,PagedAttention via vLLM with on-demand block allocation and prefix sharing. [CPU simulation]
PagedAttention (vLLM),memory_management,2048,1,205.8,403.4,0.03,0.157,205.29,0.005,10.5,0.8,PagedAttention via vLLM with on-demand block allocation and prefix sharing. [CPU simulation]
PagedAttention (vLLM),memory_management,2048,2,402.3,403.4,0.06,0.158,203.22,0.005,10.5,7.2,PagedAttention via vLLM with on-demand block allocation and prefix sharing. [CPU simulation]
PagedAttention (vLLM),memory_management,2048,4,796.0,403.4,0.11,0.194,166.6,0.006,10.5,1.0,PagedAttention via vLLM with on-demand block allocation and prefix sharing. [CPU simulation]
PagedAttention (vLLM),memory_management,4096,1,402.2,403.4,0.06,0.18,178.21,0.006,5.5,11.7,PagedAttention via vLLM with on-demand block allocation and prefix sharing. [CPU simulation]
PagedAttention (vLLM),memory_management,4096,2,795.8,404.0,0.12,0.2,160.86,0.006,5.5,13.2,PagedAttention via vLLM with on-demand block allocation and prefix sharing. [CPU simulation]
PagedAttention (vLLM),memory_management,4096,4,1580.8,404.5,0.24,0.246,130.5,0.008,5.5,15.3,PagedAttention via vLLM with on-demand block allocation and prefix sharing. [CPU simulation]
CPU Offloading,memory_management,128,1,21.7,404.8,0.0,0.112,285.02,0.004,0.0,91.3,KV cache split: most recent tokens on GPU, older tokens pinned in CPU memory. (gpu_window=256; cpu_offload_mb≈0.0)
CPU Offloading,memory_management,128,2,34.1,404.8,0.0,0.144,229.35,0.005,0.0,14.8,KV cache split: most recent tokens on GPU, older tokens pinned in CPU memory. (gpu_window=256; cpu_offload_mb≈0.0)
CPU Offloading,memory_management,128,4,59.7,404.8,0.01,0.137,236.08,0.004,0.0,18.6,KV cache split: most recent tokens on GPU, older tokens pinned in CPU memory. (gpu_window=256; cpu_offload_mb≈0.0)
CPU Offloading,memory_management,512,1,59.5,404.8,0.0,0.123,260.93,0.004,50.0,22.2,KV cache split: most recent tokens on GPU, older tokens pinned in CPU memory. (gpu_window=256; cpu_offload_mb≈0.0)
CPU Offloading,memory_management,512,2,107.7,404.8,0.01,0.131,244.39,0.004,50.0,25.1,KV cache split: most recent tokens on GPU, older tokens pinned in CPU memory. (gpu_window=256; cpu_offload_mb≈0.0)
CPU Offloading,memory_management,512,4,206.1,404.8,0.02,0.17,188.1,0.005,50.0,0.2,KV cache split: most recent tokens on GPU, older tokens pinned in CPU memory. (gpu_window=256; cpu_offload_mb≈0.0)
CPU Offloading,memory_management,2048,1,205.8,404.8,0.0,0.181,176.61,0.006,87.5,-13.3,KV cache split: most recent tokens on GPU, older tokens pinned in CPU memory. (gpu_window=256; cpu_offload_mb≈0.0)
CPU Offloading,memory_management,2048,2,402.3,404.8,0.01,0.183,182.42,0.006,87.5,-3.8,KV cache split: most recent tokens on GPU, older tokens pinned in CPU memory. (gpu_window=256; cpu_offload_mb≈0.1)
CPU Offloading,memory_management,2048,4,796.0,404.8,0.02,0.183,174.9,0.006,87.5,6.0,KV cache split: most recent tokens on GPU, older tokens pinned in CPU memory. (gpu_window=256; cpu_offload_mb≈0.1)
CPU Offloading,memory_management,4096,1,402.2,404.8,0.0,0.174,186.23,0.005,93.8,16.7,KV cache split: most recent tokens on GPU, older tokens pinned in CPU memory. (gpu_window=256; cpu_offload_mb≈0.1)
CPU Offloading,memory_management,4096,2,795.8,404.8,0.01,0.212,151.5,0.007,93.8,6.6,KV cache split: most recent tokens on GPU, older tokens pinned in CPU memory. (gpu_window=256; cpu_offload_mb≈0.1)
CPU Offloading,memory_management,4096,4,1580.8,406.0,0.02,0.246,130.27,0.008,93.8,15.1,KV cache split: most recent tokens on GPU, older tokens pinned in CPU memory. (gpu_window=256; cpu_offload_mb≈0.2)
NVFP4 (simulated),quantization,128,1,0.0,0.0,0.0,0.0,0.0,0.0,75.0,-100.0,NVFP4 (4-bit fp) KV cache simulation; halving FP8 memory. [analytical simulation]
NVFP4 (simulated),quantization,128,2,0.0,0.0,0.0,0.0,0.0,0.0,75.0,-100.0,NVFP4 (4-bit fp) KV cache simulation; halving FP8 memory. [analytical simulation]
NVFP4 (simulated),quantization,128,4,0.0,0.0,0.0,0.0,0.0,0.0,75.0,-100.0,NVFP4 (4-bit fp) KV cache simulation; halving FP8 memory. [analytical simulation]
NVFP4 (simulated),quantization,512,1,0.0,0.0,0.0,0.0,0.0,0.0,75.0,-100.0,NVFP4 (4-bit fp) KV cache simulation; halving FP8 memory. [analytical simulation]
NVFP4 (simulated),quantization,512,2,0.0,0.0,0.0,0.0,0.0,0.0,75.0,-100.0,NVFP4 (4-bit fp) KV cache simulation; halving FP8 memory. [analytical simulation]
NVFP4 (simulated),quantization,512,4,0.0,0.0,0.01,0.0,0.0,0.0,75.0,-100.0,NVFP4 (4-bit fp) KV cache simulation; halving FP8 memory. [analytical simulation]
NVFP4 (simulated),quantization,2048,1,0.0,0.0,0.01,0.0,0.0,0.0,75.0,-100.0,NVFP4 (4-bit fp) KV cache simulation; halving FP8 memory. [analytical simulation]
NVFP4 (simulated),quantization,2048,2,0.0,0.0,0.02,0.0,0.0,0.0,75.0,-100.0,NVFP4 (4-bit fp) KV cache simulation; halving FP8 memory. [analytical simulation]
NVFP4 (simulated),quantization,2048,4,0.0,0.0,0.03,0.0,0.0,0.0,75.0,-100.0,NVFP4 (4-bit fp) KV cache simulation; halving FP8 memory. [analytical simulation]
NVFP4 (simulated),quantization,4096,1,0.0,0.0,0.02,0.0,0.0,0.0,75.0,-100.0,NVFP4 (4-bit fp) KV cache simulation; halving FP8 memory. [analytical simulation]
NVFP4 (simulated),quantization,4096,2,0.0,0.0,0.03,0.0,0.0,0.0,75.0,-100.0,NVFP4 (4-bit fp) KV cache simulation; halving FP8 memory. [analytical simulation]
NVFP4 (simulated),quantization,4096,4,0.0,0.0,0.06,0.0,0.0,0.0,75.0,-100.0,NVFP4 (4-bit fp) KV cache simulation; halving FP8 memory. [analytical simulation]
MiniKV (simulated),quantization_eviction,128,1,0.0,0.0,0.0,0.0,0.0,0.0,98.1,-100.0,MiniKV 2-bit + adaptive token eviction; reports >80% memory reduction in the original paper. [analytical simulation]
MiniKV (simulated),quantization_eviction,128,2,0.0,0.0,0.0,0.0,0.0,0.0,98.1,-100.0,MiniKV 2-bit + adaptive token eviction; reports >80% memory reduction in the original paper. [analytical simulation]
MiniKV (simulated),quantization_eviction,128,4,0.0,0.0,0.0,0.0,0.0,0.0,98.1,-100.0,MiniKV 2-bit + adaptive token eviction; reports >80% memory reduction in the original paper. [analytical simulation]
MiniKV (simulated),quantization_eviction,512,1,0.0,0.0,0.0,0.0,0.0,0.0,98.1,-100.0,MiniKV 2-bit + adaptive token eviction; reports >80% memory reduction in the original paper. [analytical simulation]
MiniKV (simulated),quantization_eviction,512,2,0.0,0.0,0.0,0.0,0.0,0.0,98.1,-100.0,MiniKV 2-bit + adaptive token eviction; reports >80% memory reduction in the original paper. [analytical simulation]
MiniKV (simulated),quantization_eviction,512,4,0.0,0.0,0.0,0.0,0.0,0.0,98.1,-100.0,MiniKV 2-bit + adaptive token eviction; reports >80% memory reduction in the original paper. [analytical simulation]
MiniKV (simulated),quantization_eviction,2048,1,0.0,0.0,0.0,0.0,0.0,0.0,98.1,-100.0,MiniKV 2-bit + adaptive token eviction; reports >80% memory reduction in the original paper. [analytical simulation]
MiniKV (simulated),quantization_eviction,2048,2,0.0,0.0,0.0,0.0,0.0,0.0,98.1,-100.0,MiniKV 2-bit + adaptive token eviction; reports >80% memory reduction in the original paper. [analytical simulation]
MiniKV (simulated),quantization_eviction,2048,4,0.0,0.0,0.0,0.0,0.0,0.0,98.1,-100.0,MiniKV 2-bit + adaptive token eviction; reports >80% memory reduction in the original paper. [analytical simulation]
MiniKV (simulated),quantization_eviction,4096,1,0.0,0.0,0.0,0.0,0.0,0.0,98.1,-100.0,MiniKV 2-bit + adaptive token eviction; reports >80% memory reduction in the original paper. [analytical simulation]
MiniKV (simulated),quantization_eviction,4096,2,0.0,0.0,0.0,0.0,0.0,0.0,98.1,-100.0,MiniKV 2-bit + adaptive token eviction; reports >80% memory reduction in the original paper. [analytical simulation]
MiniKV (simulated),quantization_eviction,4096,4,0.0,0.0,0.0,0.0,0.0,0.0,98.1,-100.0,MiniKV 2-bit + adaptive token eviction; reports >80% memory reduction in the original paper. [analytical simulation]
xKV (simulated),compression,128,1,0.0,0.0,0.0,0.0,0.0,0.0,96.9,-100.0,Cross-layer SVD KV cache compression, ~8× memory reduction in the original paper. [analytical simulation]
xKV (simulated),compression,128,2,0.0,0.0,0.0,0.0,0.0,0.0,96.9,-100.0,Cross-layer SVD KV cache compression, ~8× memory reduction in the original paper. [analytical simulation]
xKV (simulated),compression,128,4,0.0,0.0,0.0,0.0,0.0,0.0,96.9,-100.0,Cross-layer SVD KV cache compression, ~8× memory reduction in the original paper. [analytical simulation]
xKV (simulated),compression,512,1,0.0,0.0,0.0,0.0,0.0,0.0,96.9,-100.0,Cross-layer SVD KV cache compression, ~8× memory reduction in the original paper. [analytical simulation]
xKV (simulated),compression,512,2,0.0,0.0,0.0,0.0,0.0,0.0,96.9,-100.0,Cross-layer SVD KV cache compression, ~8× memory reduction in the original paper. [analytical simulation]
xKV (simulated),compression,512,4,0.0,0.0,0.0,0.0,0.0,0.0,96.9,-100.0,Cross-layer SVD KV cache compression, ~8× memory reduction in the original paper. [analytical simulation]
xKV (simulated),compression,2048,1,0.0,0.0,0.0,0.0,0.0,0.0,96.9,-100.0,Cross-layer SVD KV cache compression, ~8× memory reduction in the original paper. [analytical simulation]
xKV (simulated),compression,2048,2,0.0,0.0,0.0,0.0,0.0,0.0,96.9,-100.0,Cross-layer SVD KV cache compression, ~8× memory reduction in the original paper. [analytical simulation]
xKV (simulated),compression,2048,4,0.0,0.0,0.0,0.0,0.0,0.0,96.9,-100.0,Cross-layer SVD KV cache compression, ~8× memory reduction in the original paper. [analytical simulation]
xKV (simulated),compression,4096,1,0.0,0.0,0.0,0.0,0.0,0.0,96.9,-100.0,Cross-layer SVD KV cache compression, ~8× memory reduction in the original paper. [analytical simulation]
xKV (simulated),compression,4096,2,0.0,0.0,0.0,0.0,0.0,0.0,96.9,-100.0,Cross-layer SVD KV cache compression, ~8× memory reduction in the original paper. [analytical simulation]
xKV (simulated),compression,4096,4,0.0,0.0,0.01,0.0,0.0,0.0,96.9,-100.0,Cross-layer SVD KV cache compression, ~8× memory reduction in the original paper. [analytical simulation]
```

---

## 2. Canonical technique order (also = legend order and color order)

1. `Baseline FP16` (baseline, real)
2. `Quantized INT8` (quantization, real)
3. `Quantized FP8 (E5M2)` (quantization, real)
4. `GQA` (attention, real; model = TinyLlama-1.1B-Chat-v1.0, NOT tiny-gpt2)
5. `Low-Rank Compression` (compression, real)
6. `PagedAttention (vLLM)` (memory management, real but CPU-fallback -> treat as simulated)
7. `CPU Offloading` (memory management, real)
8. `NVFP4 (simulated)` (quantization, simulated)
9. `MiniKV (simulated)` (quantization+eviction, simulated)
10. `xKV (simulated)` (compression, simulated)

Palette (cycle, in order): `#1F3A5F #C8553D #588B8B #F28F3B #7C6F9C #3A6B35 #A66C29 #586F7D` (repeat).

Markers: real techniques = `o` (solid alpha 1.0); simulated techniques (incl.
PagedAttention CPU fallback) = `x` with alpha 0.55. Simulated bars get `//`
hatch + alpha 0.55.

---

## 3. Figure-by-figure specifications

### Figure A — `memory_vs_context.png`

- **Panel**: peak GPU memory (y, MB) vs context length (x).
- **Data**: `results.csv` rows with `Batch == 1` ONLY. **Exclude** NVFP4,
  MiniKV, xKV (they report Peak GPU MB = 0). Include the other 7 techniques.
- **X axis**: log scale base 2. Y linear (0 - 7.1 GB = 0 - 7300 MB).
- **Title**: `Peak GPU memory vs context length (batch=1)`
- **X label**: `Context length (tokens)`; **Y label**: `Peak GPU memory (MB)`.
- **Legend**: upper left, 2 columns, fontsize 8.
- Expected curve shapes: tiny-gpt2 techniques 9.8 -> 361.8 MB (baseline) and
  21.7 -> 402.2 MB (INT8/FP8/LowRank/Paged/Offloading; they overlap exactly —
  plot them, the overlap is real and expected). GQA: 2124.9 -> 7074.9 MB.

### Figure B — `throughput_vs_context.png`

- **Panel**: Tokens/s (y) vs context length (x).
- **Data**: `Batch == 1`, exclude the 3 simulators. Include PagedAttention
  (its CPU-fallback numbers are real outputs; mark with `x`).
- **X axis**: log scale base 2. Y linear (0 - 300).
- **Title**: `Throughput vs context length (batch=1)`
- **X label**: `Context length (tokens)`; **Y label**: `Tokens / second`.
- **Legend**: upper right, 2 columns, fontsize 8.
- Ranges (batch=1): CPU Offloading 176.6-285.0; PagedAttention 178.2-221.4;
  Baseline 148.9-213.4; Low-Rank 151.2-176.7; INT8 134.8-151.6; FP8
  112.8-126.0; GQA 0.1-6.9.

### Figure C — `latency_vs_context.png`

- **Panel**: Latency (s, y) vs context length (x).
- **Data**: `Batch == 1`, exclude the 3 simulators.
- **X axis**: log scale base 2. Y axis: **log scale** (GQA reaches ~238 s).
- **Title**: `End-to-end latency vs context length (batch=1)`
- **X label**: `Context length (tokens)`; **Y label**: `Wall time (s)`.
- **Legend**: upper left, 2 columns, fontsize 8.
- Ranges (batch=1): GQA 4.66-237.73 s; all tiny-gpt2 techniques <= 0.36 s.

### Figure D — `memory_vs_batch.png`

- **Panel**: peak GPU memory (y, MB) vs batch size (x).
- **Data**: rows with `Context == 512` ONLY; exclude the 3 simulators.
  Batches on x: 1, 2, 4 (linear axis, could use 2^(x) spacing or plain 1/2/4).
- **Title**: `Peak GPU memory vs batch size (ctx=512)`
- **X label**: `Batch size`; **Y label**: `Peak GPU memory (MB)`.
- **Legend**: upper left, 2 columns, fontsize 8.
- Curves at ctx=512: Baseline 14.9/19.5/28.6; INT8/FP8/LowRank/Paged/
  Offloading identical 59.5/107.7/206.1; GQA 2219.7/2329.2/2562.0.

### Figure E — `summary_tokens_per_s.png`

- **Panel**: horizontal bars, one per technique (7 real ones only — exclude
  the 3 simulators, whose value would be 0).
- **Value**: mean of `Tokens/s` over ALL cells of that technique
  (all contexts x batches, computed live from the CSV above):
  - Baseline FP16: 176.4
  - Quantized INT8: 135.0
  - Quantized FP8 (E5M2): 117.0
  - GQA: 2.8
  - Low-Rank Compression: 160.1
  - PagedAttention (vLLM): 190.4
  - CPU Offloading: 203.8
  - NVFP4 (simulated): 0.0
  - MiniKV (simulated): 0.0
  - xKV (simulated): 0.0
- **Title**: `Average tokens per second across all configurations`
- **X label**: `Tokens / second`; y = technique names, largest on top
  (invert y-axis). Bars with edgecolor black; PagedAttention hatched (CPU
  fallback).

### Figure F — `summary_kv_mb.png`

- **Panel**: horizontal bars, ALL 10 techniques (simulators included; this is
  their genuine metric).
- **Value**: mean of `KV Cache MB` over ALL cells (computed live from the
  CSV above; the 4-decimal form is exact, the 2-decimal CSV rounds):
  - Baseline FP16: 0.0592
  - Quantized INT8: 0.0292
  - Quantized FP8 (E5M2): 0.0292
  - GQA: 49.2250
  - Low-Rank Compression: 0.0592
  - PagedAttention (vLLM): 0.0550
  - CPU Offloading: 0.0083
  - NVFP4 (simulated): 0.0150
  - MiniKV (simulated): 0.0000
  - xKV (simulated): 0.0008
- **X axis**: **log scale** (values span ~5 decades; set xlim ~ [1e-4, 1e2]).
- **Title**: `Average analytical KV cache size (MB) across all configurations`
- **X label**: `KV cache size (MB)`; y = technique names, largest on top.
  Simulated bars (NVFP4, MiniKV, xKV): `//` hatch + alpha 0.55.
- Note: MiniKV's mean rounds to 0.0000 in the 2-decimal CSV; use the true
  value or plot the rounded one — either is acceptable, caption treats it as
  "lowest footprint".

---

## 4. Exact reproduction

The code that produced the current PNGs (in the project root, next to
`main.tex`) is:

```text
cd coding
.venv/Scripts/python.exe scripts/make_paper_figures.py
```

`scripts/make_paper_figures.py` reads `coding/results/paper_sweep/results.csv`,
builds `BenchmarkResult` rows in the canonical order (Section 2) and calls the
plotting functions in `coding/kvbench/reporting/plots.py`:

```text
plot_memory_vs_context(real_rows, out/memory_vs_context.png, batch_size=1)
plot_throughput_vs_context(real_rows, out/throughput_vs_context.png, batch_size=1)
plot_latency_vs_context(real_rows, out/latency_vs_context.png, batch_size=1)
plot_memory_vs_batch(real_rows, out/memory_vs_batch.png, context_length=512)
plot_summary_bar(real_rows, out/summary_tokens_per_s.png, metric="tokens_per_s", ...)
plot_summary_bar(all_rows, out/summary_kv_mb.png, metric="analytic_kv_mb", log_scale=True, ...)
```

where `real_rows` = all techniques except the 3 simulators, and `all_rows` =
everything. Files are written to the project root (same directory as
`main.tex`, which includes them by bare filename). DPI 150.

If the plotting code is unavailable, implement Figures A-F from Section 3
using matplotlib with the Section 1 data; the result must match the above.

---
