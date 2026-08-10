# 15 — Models, Data Structures, and Prompts Used

This file answers the three "what did you actually use" questions:
**which models, which data structures, which prompts**. Teachers
frequently ask these — they want to confirm you ran the experiments
and didn't just cite papers.

---

## A. Models Used

We used **three HuggingFace causal LM models** from the
`transformers` library. Each one is chosen for a specific reason.

### 1. `sshleifer/tiny-gpt2` — the default baseline

| Property | Value |
|----------|-------|
| HuggingFace ID | `sshleifer/tiny-gpt2` |
| Parameters | ~2 M (very small) |
| Layers | 2 |
| Attention | Multi-Head Attention (MHA) |
| Dtype | FP16 |
| Why we use it | Tiny enough to run on the 2 GB MX230 GPU. Used as the reference baseline for FP16, INT8, FP8, low-rank, PagedAttention, offloading, and the three analytical simulators. |
| Limitations | Too small to replicate frontier-model throughput. |

**Where it is configured:** `config.py` line 26:
```python
DEFAULT_MODEL = os.environ.get("KVBENCH_MODEL", "sshleifer/tiny-gpt2")
```

### 2. `TinyLlama/TinyLlama-1.1B-Chat-v1.0` — the GQA model

| Property | Value |
|----------|-------|
| HuggingFace ID | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Parameters | 1.1 B |
| Number of query heads | 32 |
| Number of KV heads | 4 |
| Grouping factor $g$ | 32 / 4 = **8** |
| Dtype | FP16 |
| Why we use it | It is a real GQA model, so we can directly measure the ~8× reduction in `analytic_kv_mb` vs. an equivalent MHA. |
| Limitations | Still smaller than frontier models. Mostly used for the GQA category. |

**Where it is configured:** `config.py` line 30:
```python
DEFAULT_GQA_MODEL = os.environ.get(
    "KVBENCH_GQA_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)
```

### 3. `gpt2` — the MHA comparison

| Property | Value |
|----------|-------|
| HuggingFace ID | `gpt2` |
| Parameters | 124 M |
| Layers | 12 |
| Attention | MHA (no Grouping) |
| Dtype | FP16 |
| Why we use it | Used as the MHA "counterpart" to TinyLlama for comparing the analytic KV cache size: same kind of model, full MHA, shows the baseline before GQA. |
| Limitations | Slower than tiny-gpt2 but still small. |

**Where it is configured:** `config.py` line 34:
```python
DEFAULT_MHA_MODEL = os.environ.get("KVBENCH_MHA_MODEL", "gpt2")
```

### Why we used three different models

We needed three models because:

- **Reference baseline (tiny-gpt2):** tiny, fast, fits on commodity GPU.
- **GQA model (TinyLlama):** validates the GQA category on a real
  grouped-query architecture.
- **MHA model (gpt2):** gives a side-by-side MHA comparison so the
  reader sees the architectural delta directly.

The model loader has a **fallback path** — if the requested model
fails to download (e.g. offline test environment), it loads `gpt2`
instead. See `utils/models.py` lines 162–192.

### Tokenizer

For each model, we load the matching HuggingFace tokenizer via
`AutoTokenizer.from_pretrained(...)`. If the tokenizer has no
`pad_token`, we set it to `eos_token` (a standard HF pattern).

---

## B. Data Structures Used

This is the section teachers often skip but is essential to know for
the viva. Our framework uses a small, well-defined set of Python data
structures.

### 1. `BenchmarkResult` (dataclass) — the central record

Defined in `kvbench/techniques/base.py`.

```python
@dataclass
class BenchmarkResult:
    technique: str
    category: str
    context_length: int
    batch_size: int

    # Memory
    peak_gpu_mb: float
    gpu_reserved_mb: float
    cpu_rss_mb: float
    analytic_kv_mb: float

    # Latency / throughput
    wall_time_s: float
    prefill_time_s: float
    decode_time_s: float
    tokens_generated: int
    tokens_per_s: float
    time_to_first_token_s: float

    # Resources
    avg_gpu_util_pct: float
    peak_gpu_util_pct: float
    avg_cpu_pct: float
    peak_rss_mb: float

    notes: str
    raw: Dict[str, List[float]]   # per-trial values
```

**Why a dataclass:** it gives us immutability, type hints, and easy
serialization (`asdict()`) for the JSON dump. Every technique
returns one `BenchmarkResult` per trial.

### 2. `KVShape` (dataclass) — the cache shape descriptor

Defined in `kvbench/utils/models.py`.

```python
@dataclass
class KVShape:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    bytes_per_element: int

    @property
    def kv_per_token(self) -> int:
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * self.bytes_per_element

    def cache_size_bytes(self, batch: int, seq_len: int) -> int:
        return self.kv_per_token * batch * seq_len

    def cache_size_mb(self, batch: int, seq_len: int) -> float:
        return self.cache_size_bytes(batch, seq_len) / (1024 ** 2)
```

**Why:** this is the **mechanical implementation** of Equation 1 from
the paper. Every technique's `analytic_kv_mb(batch, seq_len)` calls
this and applies the technique's specific reduction (e.g. divide by 2
for INT8, by 8 for xKV).

### 3. `BenchmarkConfig` (dataclass) — the experiment config

Defined in `kvbench/config.py`. Holds every knob:

```python
@dataclass
class BenchmarkConfig:
    seed: int = 42
    trials: int = 3
    warmup_steps: int = 2
    max_new_tokens: int = 32

    model: str = "sshleifer/tiny-gpt2"
    mha_model: str = "gpt2"
    gqa_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    use_cuda: bool = True
    dtype: str = "float16"

    context_lengths: List[int] = [128, 512, 2048, 4096]
    batch_sizes: List[int] = [1, 2, 4]

    prompts: List[str] = [...]    # 8 base prompts (see below)

    output_dir: str = "results"
    save_raw: bool = True
    generate_plots: bool = True
    run_simulations: bool = True
```

**Why:** a single source of truth. The config can be saved to YAML
and reloaded, so every experiment is reproducible.

### 4. `TechniqueSpec` (dataclass) — static metadata

Defined in `kvbench/techniques/base.py`.

```python
@dataclass
class TechniqueSpec:
    name: str
    category: str          # "attention-level" / "quantization" / ...
    description: str
    is_simulation: bool    # True for NVFP4, MiniKV, xKV
    requires_gpu: bool
    requires_vllm: bool
    extra: Dict[str, Any]
```

**Why:** it lets the runner group / filter techniques by category
without knowing each subclass's internals. The whole framework
becomes a closed loop over `TechniqueSpec` instances.

### 5. `KVTechnique` (abstract base class) — the technique interface

```python
class KVTechnique(abc.ABC):
    spec: TechniqueSpec

    @abc.abstractmethod
    def setup(self) -> None: ...

    @abc.abstractmethod
    def run_trial(self, prompts, max_new_tokens, batch_size) -> BenchmarkResult: ...

    @abc.abstractmethod
    def analytic_kv_mb(self, batch: int, seq_len: int) -> float: ...

    def teardown(self) -> None: ...
    def average(self, results: List[BenchmarkResult]) -> BenchmarkResult: ...
```

**Why:** a uniform interface. The runner doesn't know whether a
technique is a real CUDA kernel or an analytical simulator — it
just calls the same three methods.

### 6. `ExperimentRunner` — the orchestrator

Lives in `kvbench/experiments/runner.py`. The "outer loop" data
structure:

```python
class ExperimentRunner:
    def __init__(self, config: BenchmarkConfig, output_dir: str | Path):
        self.results: List[BenchmarkResult] = []
        self.raw: List[Dict] = []
        self.technique_meta: Dict[str, Dict] = {}

    def run_full_sweep(self, techniques, batch_sizes, context_lengths):
        for tech in techniques:
            self._run_technique(tech, batch_sizes, context_lengths)
            tech.teardown()
```

The runner maintains a **flat list** of `BenchmarkResult` objects plus
a **dict** of metadata per technique. The same class has a
`save(name)` method that dumps everything to JSON.

### 7. Internal containers used inside techniques

| Container | Where | Purpose |
|-----------|-------|---------|
| `List[str]` of prompts | `utils/prompts.py` | The 8 base prompts (see Section C). |
| `List[torch.Tensor]` | pages of KV | PagedAttention uses a **list of fixed-size tensors** as the page pool. |
| `Dict[page_id, List[token_id]]` | PagedAttention | A per-request **page table** maps logical position → physical page. |
| `torch.Tensor` (FP16 GPU buffer) | Offloading | The GPU window of the most recent tokens. |
| `torch.Tensor` (CPU pinned memory) | Offloading | The older spillover buffer. |
| `nn.Linear` (Kaiming-uniform init) | Low-rank | The random projection matrix `W ∈ ℝ^{r×D}` and its transpose. |
| `torch.Tensor` (int8) | INT8 quantization | The quantized K/V cache plus a per-channel scale. |
| `random.Random(seed)` | Prompts | Deterministic prompt construction. |

### 8. Persistence formats

- **JSON** for the full per-trial dump (`results.json`).
- **CSV** for the per-cell aggregated tables.
- **YAML** for the `BenchmarkConfig` (so it's human-readable).
- **PNG** for the six publication-quality plots.

---

## C. Prompts Used

We use **8 base prompts** that are repeated and concatenated to
reach any target context length. The prompts are deliberately
**non-sensitive, factual sentences** about computing and ML.

The full list, verbatim from `config.py` lines 59–68:

```python
DEFAULT_PROMPTS: List[str] = [
    "The history of computing began with analog machines and mechanical calculators.",
    "Modern transformer architectures rely on self-attention to model long-range dependencies.",
    "Key-value caching avoids the redundant recomputation of past token projections.",
    "Memory bandwidth often becomes the primary bottleneck during autoregressive decoding.",
    "Quantization reduces the precision of stored tensors to save memory at a small accuracy cost.",
    "PagedAttention partitions the KV cache into fixed-size blocks to reduce fragmentation.",
    "Grouped-query attention shares key and value heads across multiple query heads.",
    "Cross-layer compression exploits redundancy between consecutive transformer layers.",
]
```

### Why these specific prompts

1. **Domain-appropriate** — they are about the same topic as the
   paper (LLMs, KV cache, attention). This means the model is not
   jarred by random topics.
2. **Roughly equal length** — each sentence is ~10–15 tokens, so
   round-robin repetition gives a near-uniform context length.
3. **Public-domain content** — no copyright concerns.
4. **Generic enough to be sent to any model** — no specific tokens
   that might cause the model to misbehave.

### How prompts are turned into a long context

The function `build_prompt` (in `utils/prompts.py`) does:

```python
def build_prompt(tokenizer, base_prompts, target_tokens, seed=0):
    """Build a deterministic prompt of approximately target_tokens tokens."""
    rng = random.Random(seed)
    pieces = []
    total = 0
    i = 0
    while total < target_tokens:
        s = base_prompts[i % len(base_prompts)]
        pieces.append(s)
        total += len(tokenizer.encode(s, add_special_tokens=False))
        i += 1
    text = " ".join(pieces)
    # Trim if we overshot.
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > target_tokens:
        text = tokenizer.decode(ids[:target_tokens], skip_special_tokens=True)
    return text
```

So for a `target_tokens=4096` cell, the function cycles through the
8 prompts ~256 times to reach ≈4096 tokens, then trims to exactly
4096 tokens by decoding only the first 4096 IDs.

### Why deterministic prompt construction

- The same `(target_tokens, seed)` pair always produces the same
  string.
- Different techniques see **the same input** — that's the
  definition of a fair comparison.
- The seed is `DEFAULT_SEED = 42` (from `config.py`) plus the trial
  index.

### Limits and what we did **not** include

- **No downstream task accuracy** — the prompts are not associated
  with any ground-truth answer. We measure memory and latency only.
- **No multi-turn chat** — the prompts are single-sentence fillers,
  not a conversation.
- **No multilingual content** — all English.
- **No injection attacks** — the prompts are factual, not adversarial.

---

## D. End-to-end data flow (for the viva)

If a teacher asks "***how does a single trial work end-to-end***",
here is the answer:

1. **Config loaded.** `BenchmarkConfig` is read from `config.py` or
   a YAML file.
2. **Runner created.** `ExperimentRunner(config, output_dir=...)`.
3. **For each technique** (from `techniques/__init__.py`):
   1. `technique.setup()` — loads the model (e.g. `tiny-gpt2`) and
      tokenizer.
   2. **For each (context_length, batch_size) cell:**
      1. **For each trial** (1 to `config.trials`):
         1. `seed_everything(seed + trial_idx)` — seed Python random,
            NumPy, PyTorch.
         2. `build_prompt(tokenizer, prompts, ctx, seed)` — produce
            a deterministic long prompt.
         3. `technique.run_trial(prompts, max_new_tokens, batch)` —
            runs the model, measures wall-clock, tokens/s, memory.
         4. Return a `BenchmarkResult`.
   3. `technique.average(results)` — average over trials.
   4. `technique.teardown()` — release GPU memory.
4. **Save results.** `runner.save("results")` writes:
   - `results.json` — full per-trial dump.
   - `results.csv` — per-cell aggregated.
   - Six PNG plots via `reporting/plots.py`.

---

## E. Common viva questions

1. **Q: Which model did you use for the baseline?**
   A: `sshleifer/tiny-gpt2`, a 2-layer FP16 model that fits on our
   2 GB GPU.

2. **Q: Which model did you use for GQA?**
   A: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, which has 32 query heads
   and 4 KV heads, giving a grouping factor $g = 8$.

3. **Q: Why three models?**
   A: tiny-gpt2 for the baseline and most techniques, TinyLlama for
   GQA, and gpt2 as the MHA counterpart for the GQA comparison.

4. **Q: What data structure holds the benchmark output?**
   A: A `BenchmarkResult` dataclass with fields for memory,
   latency, throughput, and notes.

5. **Q: How are prompts built?**
   A: 8 base sentences are repeated round-robin via
   `build_prompt()` until the target token count is reached, then
   trimmed to exactly that count.

6. **Q: How do you ensure every technique sees the same input?**
   A: The same `build_prompt(tokenizer, prompts, target_tokens, seed)`
   call is made for every cell. The seed is fixed.

7. **Q: How is the analytic KV size computed in code?**
   A: `KVShape.cache_size_mb(batch, seq_len)` which implements
   Eq. 1 directly.

8. **Q: Where is PagedAttention's page table stored?**
   A: As a `dict` mapping `page_id → list[token_id]` per request,
   kept in the `PagedAttentionVLLM` class.

9. **Q: How are the INT8 scales stored?**
   A: As a per-channel `torch.Tensor` of floats, kept alongside
   the int8 K/V cache.

10. **Q: What is the persistence format?**
    A: JSON for the full per-trial dump, CSV for the per-cell
    aggregated tables, YAML for the config, PNG for the plots.
