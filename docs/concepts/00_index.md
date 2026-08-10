# Viva Preparation — Index

> **Paper:** *KV Cache Optimization in LLMs: A Taxonomy and Decision Framework for Efficient Inference*
> **Authors:** Aayush Chhuka, Aditya Brajacharya, Sudip Dhungana
> **Institution:** Department of CSE, National College of Engineering (Lalitpur, Nepal)
> **Viva:** Tomorrow (with AOA teacher)
> **Folder:** `docs/concepts/`

This folder contains a complete, exam-style explanation of every concept used in the
research paper. Read these files in order before the viva.

---

## How to use this folder

| File | Topic | Why it matters |
|------|-------|----------------|
| `01_what_is_kv_cache.md` | What the KV cache is, why it exists, the two equations | The single most fundamental question in the paper |
| `02_taxonomy_overview.md` | The 4-category bottleneck-oriented taxonomy | The paper's main contribution |
| `03_attention_level.md` | GQA, MHA, MQA | Category 1 of the taxonomy |
| `04_quantization.md` | FP8, NVFP4, MixQuant, MiniKV | Category 2 of the taxonomy |
| `05_compression.md` | Low-rank, KV merging, multi-level, xKV (SVD) | Category 3 of the taxonomy |
| `06_memory_management.md` | Offloading, PagedAttention, Shared Attention | Category 4 of the taxonomy |
| `07_complexity_analysis.md` | O(N²), O(L·H·D·B·N), how each technique affects it | The math section that teachers love |
| `08_kvbench_framework.md` | The Python benchmarking framework | "How did you actually run the experiments?" |
| `09_metrics_measured.md` | peak GPU MB, CPU RSS, wall-clock, throughput, tokens/s, analytical KV | "What did you measure?" |
| `10_what_we_learned.md` | Key insights, observations, findings | "What conclusions did you draw?" |
| `11_problems_and_solutions.md` | What went wrong, what we did to fix it | "Did you face any problems?" |
| `12_limitations.md` | Honest limitations, what's missing | Always asked in viva |
| `13_potential_viva_questions.md` | Likely questions with model answers | Last-minute revision |
| `14_glossary.md` | One-line definitions for every acronym | Quick lookup |
| `15_models_prompts_datastructures.md` | Which models, which data structures, which prompts | "What did you actually use?" |

---

## 30-second elevator pitch (memorize this)

> "KV cache lets LLMs generate text one token at a time without re-computing
> everything before, but the cache grows linearly with context and becomes
> a bottleneck for GPU memory, memory bandwidth, and inference latency. In
> this paper, we **organize 10 KV cache optimization techniques** into
> **4 categories** — attention-level redesign, quantization, compression,
> and memory management — based on **which part of the cache they optimize**.
> We **built `kvbench`**, a Python benchmarking framework, to compare them
> on the same prompts, seeds, hardware, and metrics. Three proprietary
> techniques (NVFP4, MiniKV, xKV) we couldn't run, so we **simulated them
> analytically using the paper's own formulas**. The result is a
> **decision framework** that tells you which technique to use when the
> bottleneck is memory, latency, or throughput."

---

## Reading order for the night before the viva

1. `00_index.md` (this file)
2. `01_what_is_kv_cache.md`
3. `02_taxonomy_overview.md`
4. Read the four category files (`03_` → `06_`)
5. `07_complexity_analysis.md`
6. `08_kvbench_framework.md`
7. `09_metrics_measured.md`
8. `10_what_we_learned.md`
9. `11_problems_and_solutions.md`
10. `12_limitations.md`
11. `13_potential_viva_questions.md` ← practice these out loud
12. `14_glossary.md` ← keep open while revising
13. `15_models_prompts_datastructures.md` ← if asked "what did you actually use?"

Good luck tomorrow.
