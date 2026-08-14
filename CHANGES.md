# Changes Log

Changes made to the KV-cache research project (paper: `main.tex`, framework: `coding/kvbench/`).

---

## Session: fused-QKV support — tiny-gpt2 now genuinely runs all five real techniques

The memory-side benchmark and the perplexity harness previously **silently skipped** the
Quantized INT8 / FP8 and Low-Rank Compression techniques on `sshleifer/tiny-gpt2` because its
attention uses a fused QKV projection (a single `c_attn` weight producing Q, K, V together),
which has no separable `k_proj`/`v_proj` to hook. Additionally, the quantized technique classes
were **uninstantiable**: `_QuantizedBase` never implemented the abstract `run_trial()` from
`base.py`. All of this is now fixed.

### `coding/kvbench/techniques/quantized.py`
- Added `install_kv_roundtrip_hooks(model, quant_dtype)` — a shared hook installer covering two layouts:
  - Llama-style: hooks `k_proj` / `v_proj` outputs (unchanged behaviour).
  - GPT-2-style: hooks the fused `c_attn` output, splits it into Q/K/V slices, round-trips the
    K/V slices through INT8 (per-channel symmetric) or FP8 E5M2, and re-concatenates.
- Added `_fp8_e5m2_roundtrip(x)` (value-level E5M2 round-trip, previously duplicated in
  `perplexity_eval.py`) and `_c_attn_width()` (HF `Conv1D` exposes `nf`, not `out_features`).
- Implemented `run_trial()` on `_QuantizedBase` (prefill + decode loop with the quant hooks
  active, mirroring `LowRankCompression.run_trial`). This fixes the abstract-class
  `TypeError` that made `Int8SimulatedKV` / `FP8SimulatedKV` impossible to instantiate.
- Removed the dead `_generate()` / `_capture_enabled` machinery (caches were never populated;
  `_capture_enabled` was set but never read).

### `coding/kvbench/techniques/compression.py`
- `_install_projectors()` gained a fused-QKV branch: when `k_proj`/`v_proj` are missing but
  `attn.c_attn` exists, the fused output is split into Q/K/V slices, each K/V slice is
  low-rank projected (down/up, Kaiming-init), and re-concatenated. Projectors now install on
  tiny-gpt2 (2 of 2 layers); previously every layer was silently skipped.

### `coding/kvbench/techniques/perplexity_eval.py`
- `_install_kv_quant_hooks()` now delegates to the shared `install_kv_roundtrip_hooks()`; the
  "fused QKV ... equals FP16 baseline" fallback notes were removed (they can no longer occur
  for GPT-2). The safety-net note now only fires if a model has neither separable projections
  nor a fused `c_attn`.

### Results (re-run on the GTX 1650 machine, `seed=42`)
- `coding/results/perplexity_tinygpt2_fixed/perplexity_results.csv` — new tiny-gpt2 perplexity
  (103 chunks each, chunk_len 512):

  | Technique      | Before      | After       |
  |----------------|-------------|-------------|
  | Baseline FP16  | 50310.82    | 50310.8204  |
  | Quantized INT8 | 50310.82    | 50310.8138  |
  | Quantized FP8  | 50310.82    | 50310.8287  |
  | GQA            | 50310.82    | 50310.8204 (lossless) |
  | Low-Rank       | 50310.82    | 50311.2642  |

  Differences are small because tiny-gpt2's `head_dim=1` makes the INT8/FP8 round-trip nearly
  lossless and the rank-64 projectors are effectively full rank on the 2-wide K/V slice — this
  is documented in the paper table notes.
- `coding/results/perplexity_verify/perplexity_results.csv` — canonical tiny-gpt2 CSV updated
  with the new values (the paper's Table IX was transcribed from this file).
- `coding/results/tinygpt2_memory_real/` — first real memory/latency benchmark of the five real
  techniques on tiny-gpt2 (ctx 256/512/2048, batch 1/2/4, 1 trial). INT8/FP8/Low-Rank run with
  genuine K/V modification; GQA loads TinyLlama as the equivalent GQA model (paper design).
  Note: `peak_gpu_mb` reads 0.0 on this GTX 1650 machine (MemoryTracker limitation; the paper's
  memory tables come from the MX230 machine).
- TinyLlama regression check: 44 quant hooks / 22 low-rank projectors install as before;
  TinyLlama perplexity untouched (11.2467 / 11.2491 / 11.3258 / 11.2467 / 2704.5701).

### `main.tex`
- **Table IX (tab:perplexity)**: tiny-gpt2 rows updated to the new differentiated values with
  new notes ("fused `c_attn` output split; K/V round-trip", low-rank "rank 2 = full rank at
  head_dim=1").
- **Observations (Section V-D)**: added a sentence on the tiny-gpt2 signal and why it is small;
  the meaningful accuracy effects are on TinyLlama.
- **Table V (techniques)**: INT8 / FP8 / Low-Rank notes changed from "requires separable K/V
  projections (fused-QKV models fall back to FP16)" to "hooks k_proj/v_proj or the fused
  c_attn output split".
- **Table VII (tab:results_summary) caption**: "analytical only on tiny-gpt2" replaced — the
  techniques are now genuinely executed at runtime on both models; the cache-size figures remain
  analytic estimates from `analytic_kv_mb()`.
- **Section VII (Limitations)**: the "tiny-gpt2 architecture limitation" bullet rewritten as
  "tiny-gpt2 fused-QKV handling" — describes the output-split interception implementation, the
  measured tiny-gpt2 differences, and the remaining caveats (memory figures analytic on both
  models; TinyLlama low-rank degradation from random uncalibrated projectors).
- PDF recompiled cleanly (13 pages).

---

## Session: data provenance — figures & tables regenerated from a real GTX 1650 sweep

The paper's figures were previously generated from hardcoded synthetic numbers
(`make_paper_figures.py` used "representative numbers derived from the literature"), and the
venv contained a **CPU-only PyTorch build**, so every earlier "GPU" measurement reported
`peak_gpu_mb = 0.0` and silently ran on the CPU. This session fixed the environment, ran the
full paper-config sweep on the real GPU, and rewired the figures/tables to the measured CSV.

### Environment
- Installed `torch==2.11.0+cu128` (was `2.13.0+cpu`) — the GTX 1650 (4 GB, sm_75, driver 581.80)
  is now used for all runs; `torch.cuda.max_memory_allocated()` reports real peaks.

### `coding/kvbench/utils/models.py`
- New `extend_position_embeddings(model, tokenizer, min_seq_len)`: extends learned absolute
  position embeddings (GPT-2 `wpe`, BERT-style tables) by cyclically repeating the rows and
  raises the tokenizer's `model_max_length`. Needed because tiny-gpt2's `n_positions=1024`
  otherwise makes the 2048/4096-token sweep cells crash (`IndexError: index out of range`) —
  or worse, the tokenizer silently truncated them to 1024 so the "2048" cells were actually
  measured at 1024. RoPE models (TinyLlama) get only the tokenizer bump.

### `coding/kvbench/experiments/runner.py`
- `_run_technique` now calls `extend_position_embeddings` after `setup()` for every technique
  that loads a real model (target = max context + `max_new_tokens` + headroom).

### `coding/kvbench/techniques/simulations.py`
- `_SimulatorBase.run_trial` no longer hardcodes `approx_ctx = 256`: it tokenizes the actual
  prompt (lazy-loaded tokenizer, also assigned in `setup()` so the runner builds properly
  scaled prompts). Simulated cells now land on the real 128/512/2048/4096 grid.

### `coding/scripts/sweep_checkpointed.py` (new)
- Checkpointed sweep runner: saves `all.json` + `results.csv` + `results_aggregated.csv` after
  every technique; `--only` / `--force` / resume semantics; merges prior results instead of
  overwriting (an early merge bug lost data once; fixed by tracking actually-re-run names and
  reading `all.json` — `runner.save("all")` writes `all.json`, not `results.json`).

### Results (`coding/results/paper_sweep/`)
- Full sweep on the GTX 1650: 10 techniques × ctx [128,512,2048,4096] × bs [1,2,4] × 3 trials
  = 118 measured cells. GQA (TinyLlama-1.1B) OOMs at ctx=4096, bs=2/4 (4 GB VRAM) — recorded as
  "not measured", not fabricated. Simulated techniques are analytic-only (peak/latency 0).
- Notable real findings: tiny-gpt2's KV cache is 16 bytes/token (≤0.06 MB at ctx=4096), so
  peak-GPU curves of the tiny-gpt2 techniques overlap (weights/activations dominate); GQA runs
  a much larger model (TinyLlama-1.1B) and is heaviest (2.1–7.1 GB) and slowest
  (0.1–6.9 tok/s at bs=1); quantization measurably reduces throughput at this scale
  (avg 176 tok/s baseline vs 135 INT8 / 117 FP8); PagedAttention used its pure-PyTorch CPU
  fallback (marked `[CPU simulation]` in notes → `×` marker in figures).

### `coding/scripts/make_paper_figures.py`
- Rewritten: loads the real per-cell CSV (`results.csv`), builds `BenchmarkResult` rows in
  canonical technique order, writes the six PNGs to the project root (next to `main.tex`).
  Simulators are excluded from the peak-GPU/throughput/latency panels (they measured nothing)
  and included in the KV-size summary. `plot_summary_bar` gained a `log_scale` option (KV
  sizes span 5 decades) and human-readable axis labels.

### `main.tex`
- **Table VI / Section IV-C**: hardware corrected to the real machine — NVIDIA GTX 1650 4 GB
  (sm_75), Intel Core i5-11300H, 16 GB RAM, Windows 11 Pro build 26220, PyTorch 2.11.0+cu128,
  CUDA 12.8. All measurements (memory/latency AND perplexity) now come from this one machine;
  the MX230 story is gone. Documents the GQA OOM cells as "not measured".
- **Table VII (tab:results_summary)**: real analytic values at ctx=256, bs=1 (tiny-gpt2
  shape: baseline 0.0039 MB; INT8/FP8 0.0020; low-rank 0.0039 = 0% since rank-64 projectors are
  full-rank at head_dim=1; PagedAttention 0.0020 = 48.4% block utilization; CPU Offloading
  0.0039 = 0% since ctx=256 fits the 256-token GPU window; NVFP4 0.0010, MiniKV 0.0001 (98.1%),
  xKV 0.0001). GQA row: 5.5 MB on TinyLlama-1.1B, 8× vs same-shape MHA — not comparable to the
  tiny-gpt2 rows (caption explains).
- **Table VIII (tab:scaling)**: simulated techniques at ctx=512/2048 from the real sweep
  (NVFP4 0.0020→0.0078; MiniKV 0.0001→0.0006; xKV 0.0002→0.0010; ≈4× growth).
- **Figure 2–4 captions**: rewritten for the measured data (sims omitted from GPU/throughput/
  latency panels with justification; PagedAttention CPU fallback marked; GQA's TinyLlama model
  size called out). Figures regenerated at project root.
- **Observations (iii)/(iv)**: low-rank shows 0% analytic reduction on tiny-gpt2 (full-rank
  projectors); PagedAttention 48.4% block utilization; Offloading savings only materialize
  above the 256-token GPU window.
- **Section VII**: new "Measured data provenance" bullet — figures/tables come from
  `coding/results/paper_sweep/`; discloses the position-embedding extension (1024→4136, cyclic
  repetition), the GQA OOM cells reported as "not measured", and the PagedAttention CPU
  fallback marker.

---

## Session: TinyLlama perplexity + scientific-consistency fixes (earlier)

### `coding/kvbench/techniques/compression.py`
- GQA projector sizing fix: `kv_dim = k_proj_weight.out_features` instead of `head_dim` —
  previously the low-rank projectors crashed with a shape error on GQA models like TinyLlama
  (narrow shared KV width), which is why the TinyLlama low-rank perplexity was OOM before.

### Results
- `coding/results/perplexity_tinyllama/perplexity_results.csv` — TinyLlama low-rank perplexity
  re-run after the fix: 2704.5701 (122 chunks, 0 OOM). Baseline 11.2467, INT8 11.2491,
  FP8 11.3258, GQA 11.2467.

### `main.tex`
- **TABLE IX (tab:perplexity)**: TinyLlama rows filled (Baseline 11.25, INT8 11.25, FP8 11.33,
  GQA 11.25, Low-Rank 2704.57) + caption fix (missing closing brace, `\code` → `\texttt`).
- **Observations**: updated % changes (+0.02% / +0.70% / 0.00% / +23947.6%) and added a
  "so what" sentence tying the accuracy results to the taxonomy.
- **Section IV-C / Table VI**: hardware provenance made explicit — main benchmark on the MX230
  2 GB machine; perplexity evaluation on the GTX 1650 4 GB machine (justification: TinyLlama
  does not fit on the MX230). Confirmed by the user.
- **Section VII**: added the "tiny-gpt2 architecture limitation" disclosure paragraph (later
  superseded by the fused-QKV handling bullet of the current session).
- **Table V notes / Table VII caption**: initial honesty pass (analytic-only labelling).

---

## Not changed (per scope constraints)

- Paper title, the category-boundaries paragraph in Section III-A, and the inline limitation
  pointers in Section V-A were left untouched.
- The TinyLlama GQA discussion in Section V-A(i) was not modified.

## Known outstanding issues (not addressed in this work)

- The perplexity CSVs and the `_smoke/` results predate the CUDA-enabled sweep; they remain
  valid measurements but were produced before the position-embedding extension and the CUDA
  build (perplexity is GPU-independent; the smoke run is superseded by `paper_sweep/`).
- The GQA (TinyLlama-1.1B) cells at ctx=4096, bs=2/4 cannot run on the 4 GB GTX 1650 and are
  reported as "not measured".
- The simulated techniques (NVFP4, MiniKV, xKV) remain analytical-only, as intended.