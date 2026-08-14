"""Perplexity evaluation for the directly-run KV cache techniques.

The main :mod:`kvbench` sweep measures *memory* and *latency*; it
deliberately does not touch task accuracy. Reviewers asked for at least
one measured accuracy signal, so this module adds a small, memory-safe
language-modelling perplexity evaluation on a held-out subset of
WikiText-2.

Design constraints (target hardware: NVIDIA GTX 1650, 4 GB VRAM, Intel
i5, no other accelerator):

* Only a *small* held-out subset is used -- the first
  :data:`PERPLEXITY_EVAL_LINES` non-empty lines of the ``test`` split --
  mirroring the framework's existing precedent of controlled, limited
  inputs (e.g. the eight fixed prompts in :mod:`kvbench.config`). This
  keeps runtime and memory bounded on a 4 GB card.
* The text is tokenized once and split into non-overlapping chunks of
  :data:`PERPLEXITY_CHUNK_LEN` tokens. 512 is a memory-safe default;
  going much higher risks OOM once a technique's own memory usage is
  added on top, especially for TinyLlama.
* Every chunk is scored with ``batch_size = 1`` under ``torch.no_grad()``.
  A per-chunk CUDA OOM is caught, the cache is cleared, the chunk is
  skipped (and counted), and evaluation continues instead of crashing.
* CUDA memory is fully released (``empty_cache`` + ``gc.collect``) after
  each technique so nothing leaks into the next -- critical on 4 GB.

Only the five *real* techniques from Table V are evaluated: Baseline
FP16, Quantized INT8, Quantized FP8 (E5M2), GQA and Low-Rank
Compression. NVFP4/MiniKV/xKV are analytical simulators with no real
model to run, and PagedAttention / CPU Offloading are lossless
memory-management techniques whose outputs -- and therefore perplexity
-- are identical to Baseline FP16 by construction, so they are not
re-computed here.
"""

from __future__ import annotations

import csv
import gc
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from ..config import BenchmarkConfig
from ..utils.logging import get_logger
from ..utils.models import get_torch_device

logger = get_logger("perplexity")

# ---------------------------------------------------------------------------
# Configuration constants.
# ---------------------------------------------------------------------------

# Number of non-empty WikiText-2 lines to evaluate. Kept small on purpose:
# the paper already uses controlled, limited inputs (eight fixed prompts),
# and a 4 GB GTX 1650 cannot afford the full test split alongside a
# technique's own memory. Raise on a larger card for a tighter estimate.
PERPLEXITY_EVAL_LINES = 500

# Non-overlapping context window (in tokens) used to score the text.
# 512 is the memory-safe default; the OOM fallback below drops to
# PERPLEXITY_CHUNK_LEN_FALLBACK when even batch_size=1 will not fit.
PERPLEXITY_CHUNK_LEN = 512
PERPLEXITY_CHUNK_LEN_FALLBACK = 256

# HuggingFace ``datasets`` coordinates for the evaluation corpus.
# ``Salesforce/wikitext`` is the namespaced, parquet-backed mirror of the
# original ``wikitext`` dataset; the bare ``wikitext`` id no longer loads on
# datasets>=3 / huggingface_hub>=1 (it required a removed loading script).
PERPLEXITY_DATASET = ("Salesforce/wikitext", "wikitext-2-raw-v1")
PERPLEXITY_SPLIT = "test"


def _oom_error_types():
    """CUDA out-of-memory exception types available on this PyTorch build."""
    specific = getattr(torch.cuda, "OutOfMemoryError", None)
    return (specific,) if specific is not None else tuple()


def _is_oom(exc: Exception) -> bool:
    """True if ``exc`` is a CUDA OOM (typed, or a RuntimeError by message)."""
    for t in _oom_error_types():
        if isinstance(exc, t):
            return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


# ---------------------------------------------------------------------------
# Data loading.
# ---------------------------------------------------------------------------

def load_wikitext_subset(n_lines: int = PERPLEXITY_EVAL_LINES) -> str:
    """Load and concatenate the first ``n_lines`` non-empty WikiText-2 lines.

    Uses the raw ``test`` split so the text is disjoint from training data.
    Blank / whitespace-only lines (WikiText separators) are skipped before
    counting, so ``n_lines`` counts real content lines.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "The 'datasets' package is required for perplexity evaluation. "
            "Install it with `pip install datasets`."
        ) from e

    name, subset = PERPLEXITY_DATASET
    ds = load_dataset(name, subset, split=PERPLEXITY_SPLIT)

    lines: List[str] = []
    for row in ds:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        lines.append(text)
        if len(lines) >= n_lines:
            break

    logger.info(
        "Loaded %d non-empty lines from %s/%s [%s].",
        len(lines), name, subset, PERPLEXITY_SPLIT,
    )
    # Join with blank lines, matching how WikiText documents are laid out.
    return "\n\n".join(lines)


def build_chunks(
    tokenizer,
    text: str,
    chunk_len: int = PERPLEXITY_CHUNK_LEN,
) -> List[List[int]]:
    """Tokenize ``text`` once and split it into non-overlapping chunks.

    A trailing chunk shorter than two tokens is dropped, since perplexity
    needs at least one (context, target) pair after the causal shift.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = [ids[i : i + chunk_len] for i in range(0, len(ids), chunk_len)]
    return [c for c in chunks if len(c) >= 2]

# ---------------------------------------------------------------------------
# Core perplexity computation.
# ---------------------------------------------------------------------------

@dataclass
class PerplexityResult:
    """Outcome of one (model, technique) perplexity evaluation."""

    perplexity: float
    num_chunks_evaluated: int
    num_chunks_skipped_oom: int
    chunk_len_used: int
    notes: str = ""


def compute_perplexity(
    model,
    tokenizer,
    technique_name: str,
    device,
    chunk_len: int = PERPLEXITY_CHUNK_LEN,
) -> PerplexityResult:
    """Token-level perplexity of ``model`` over the WikiText-2 subset.

    For each non-overlapping ``chunk_len``-token window we feed the
    ``input_ids`` as both input and labels; HuggingFace computes the
    shifted next-token cross-entropy internally and returns it as
    ``outputs.loss`` (a per-token mean over the ``T-1`` predicted
    positions). We accumulate the *summed* negative log-likelihood and
    the token count, so the final ``exp(total_nll / total_tokens)`` stays
    correct even when the last chunk is short or the fallback chunk length
    is used.

    Every window runs at ``batch_size = 1`` under ``torch.no_grad()``. A
    per-chunk CUDA OOM is caught: the cache is cleared, the chunk is
    skipped and counted, and evaluation continues.
    """
    model.eval()
    text = load_wikitext_subset(PERPLEXITY_EVAL_LINES)
    chunks = build_chunks(tokenizer, text, chunk_len=chunk_len)

    total_nll = 0.0
    total_tokens = 0
    evaluated = 0
    skipped_oom = 0

    for idx, chunk in enumerate(chunks):
        input_ids = None
        try:
            input_ids = torch.tensor([chunk], dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(input_ids=input_ids, labels=input_ids)
            # ``T-1`` predicted positions remain after the internal shift.
            n_tok = input_ids.shape[1] - 1
            if n_tok <= 0:
                continue
            total_nll += float(out.loss.item()) * n_tok
            total_tokens += n_tok
            evaluated += 1
        except Exception as exc:  # noqa: BLE001
            if _is_oom(exc):
                skipped_oom += 1
                logger.warning(
                    "[%s] CUDA OOM on chunk %d/%d (chunk_len=%d); "
                    "clearing cache and skipping.",
                    technique_name, idx, len(chunks), chunk_len,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise
        finally:
            input_ids = None

    # Fully release memory before the next technique -- critical on 4 GB.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    if total_tokens == 0:
        return PerplexityResult(
            perplexity=float("nan"),
            num_chunks_evaluated=0,
            num_chunks_skipped_oom=skipped_oom,
            chunk_len_used=chunk_len,
            notes="no chunks evaluated",
        )

    ppl = math.exp(total_nll / total_tokens)
    logger.info(
        "[%s] perplexity=%.4f over %d chunks (%d skipped, chunk_len=%d).",
        technique_name, ppl, evaluated, skipped_oom, chunk_len,
    )
    return PerplexityResult(
        perplexity=ppl,
        num_chunks_evaluated=evaluated,
        num_chunks_skipped_oom=skipped_oom,
        chunk_len_used=chunk_len,
    )


# ---------------------------------------------------------------------------
# Technique registry + orchestration.
# ---------------------------------------------------------------------------

# CSV column order (kept stable for the paper's Table IX).
PERPLEXITY_CSV_COLUMNS = [
    "model",
    "technique",
    "category",
    "num_chunks_evaluated",
    "num_chunks_skipped_oom",
    "perplexity",
    "notes",
]

# Default model sweep: tiny-gpt2 first (fast, near-zero VRAM risk; validates
# the harness), then TinyLlama (the real 4 GB stress case).
DEFAULT_PERPLEXITY_MODELS = [
    "sshleifer/tiny-gpt2",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]


def _real_technique_builders():
    """The five real (directly-run) techniques from Table V.

    Each entry is ``(display_name, category, kind, quant_dtype)``:

    * ``kind == "technique"`` -- built through its own class, whose
      ``setup()`` loads the model (and, for Low-Rank, installs the
      projector hooks that make its degradation observable in a plain
      forward pass). Used for Baseline, GQA and Low-Rank.
    * ``kind == "quant"`` -- the KV-cache-quantization techniques. Their
      classes measure a quantized *cache* inside a custom decode loop and
      do not alter a teacher-forced forward pass, so for perplexity we
      instead load the model directly and attach K/V quantization
      round-trip hooks (:func:`_install_kv_quant_hooks`). ``quant_dtype``
      selects INT8 or FP8 (E5M2).

    NVFP4 / MiniKV / xKV are intentionally absent (analytical simulators,
    no real model). PagedAttention and CPU Offloading are also absent:
    they are lossless, so their perplexity equals Baseline FP16 by
    construction and re-computing it would only waste VRAM.
    """
    return [
        ("Baseline FP16", "baseline", "technique", None),
        ("Quantized INT8", "quantization", "quant", "int8"),
        ("Quantized FP8 (E5M2)", "quantization", "quant", "fp8"),
        ("GQA", "attention", "technique", None),
        ("Low-Rank Compression", "compression", "technique", None),
    ]


def _build_technique(name: str, config: BenchmarkConfig):
    """Instantiate the technique class used for a ``kind == "technique"`` row."""
    from .baseline import BaselineKV
    from .gqa import GQAModel
    from .compression import LowRankCompression

    if name == "Baseline FP16":
        return BaselineKV(config)
    if name == "GQA":
        return GQAModel(config)
    if name == "Low-Rank Compression":
        return LowRankCompression(config, rank=64)
    raise ValueError(f"Unknown technique: {name}")


# ---------------------------------------------------------------------------
# Faithful accuracy simulation for the quantized techniques.
#
# The INT8 / FP8 technique classes measure the *memory and latency* of a
# quantized KV cache inside their own decode loop; a plain teacher-forced
# forward pass (which is what perplexity uses) does not exercise that path,
# so without help their perplexity would be byte-identical to the FP16
# baseline. To make the accuracy number reflect the technique, we install
# forward hooks on every K/V projection that round-trip the projection
# output through the same INT8 / FP8 representation the cache would store.
# This mirrors how KV-cache-quantization accuracy is evaluated in the
# literature (quantize the cached K/V, dequantize for attention) and reuses
# kvbench's own INT8 quantizer.
# ---------------------------------------------------------------------------

def _fp8_e5m2_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """Round ``x`` to FP8 E5M2 precision and back, staying in floating point.

    Uses the same 5-exponent / 2-mantissa rounding as
    :func:`kvbench.techniques.quantized._e5m2_from_fp32`, but returns the
    rounded *value* (not a reinterpreted byte), which is what an attention
    kernel reads after dequantizing an E5M2 cache. Pure tensor ops, so it
    behaves identically on CPU and CUDA.
    """
    sign = torch.sign(x)
    a = x.abs()
    finite_max = 57344.0  # largest normal E5M2 magnitude
    a = torch.where(torch.isfinite(a), a, torch.full_like(a, finite_max)).clamp(max=finite_max)
    exp = torch.floor(torch.log2(a.clamp(min=1e-30))).clamp(min=-14, max=15)
    mant = a / (2.0 ** exp)
    mant_q = 1.0 + torch.round((mant - 1.0) * 4.0).clamp(min=0, max=3) / 4.0
    return sign * mant_q * (2.0 ** exp)


def _install_kv_quant_hooks(model, quant_dtype: str) -> List:
    """Hook every ``k_proj`` / ``v_proj`` to quantize->dequantize its output.

    Returns the list of hook handles. The list is empty when the model has
    no separate K/V projections (e.g. GPT-2's fused ``c_attn``); in that
    case a quantized run legitimately equals the baseline and the caller
    notes it rather than reporting a spurious difference.
    """
    from .quantized import quantize_int8_per_channel, dequantize_int8_per_channel

    def _make_hook():
        def _hook(_mod, _inp, out):
            orig_dtype = out.dtype
            x = out.float()
            if quant_dtype == "int8":
                q, scale = quantize_int8_per_channel(x)
                deq = dequantize_int8_per_channel(q, scale)
            else:  # fp8 (E5M2)
                deq = _fp8_e5m2_roundtrip(x)
            return deq.to(orig_dtype).view_as(out)
        return _hook

    handles = []
    for name, module in model.named_modules():
        if name.split(".")[-1] in ("k_proj", "v_proj"):
            handles.append(module.register_forward_hook(_make_hook()))
    return handles


def _evaluate_once(
    name: str,
    kind: str,
    quant_dtype: Optional[str],
    model_name: str,
    device,
    use_cuda: bool,
    base_config: BenchmarkConfig,
    chunk_len: int,
) -> PerplexityResult:
    """Load a technique's model, score its perplexity, and always clean up.

    Two paths (see :func:`_real_technique_builders`):

    * ``kind == "technique"`` -- Baseline / GQA / Low-Rank are built through
      their own class so their ``setup()`` runs (Low-Rank installs its
      projector hooks). Both ``model`` and ``gqa_model`` are routed to
      ``model_name`` so GQA (which reads ``gqa_model``) loads the same model.
    * ``kind == "quant"`` -- INT8 / FP8 load the model directly and attach
      KV-quantization round-trip hooks, because their technique classes only
      quantize the *cache* in a custom decode loop that a teacher-forced
      forward pass never runs.
    """
    import copy

    cfg = copy.deepcopy(base_config)
    cfg.model = model_name
    cfg.gqa_model = model_name
    cfg.use_cuda = use_cuda

    technique = None
    quant_hooks: List = []
    note = ""
    try:
        if kind == "quant":
            from ..utils.models import (
                dtype_from_string, get_torch_device, load_model_and_tokenizer,
            )
            dev = device or get_torch_device(use_cuda=use_cuda)
            model, tokenizer = load_model_and_tokenizer(
                model_name,
                dtype=dtype_from_string(cfg.dtype),
                device=dev,
                use_cuda=use_cuda,
            )
            quant_hooks = _install_kv_quant_hooks(model, quant_dtype)
            if not quant_hooks:
                note = ("fused QKV projection (no separate k_proj/v_proj); "
                        "KV quantization not separable, equals FP16 baseline")
        else:
            technique = _build_technique(name, cfg)
            technique.setup()
            model, tokenizer = technique._model, technique._tokenizer
            if hasattr(technique, "_hooks") and not technique._hooks:
                # Low-Rank on a fused-QKV model installs no projectors, so its
                # forward pass -- and thus perplexity -- equals the baseline.
                note = ("no separable k_proj/v_proj; low-rank projectors not "
                        "installed, equals FP16 baseline")

        res = compute_perplexity(model, tokenizer, name, device, chunk_len=chunk_len)
        if note:
            res.notes = "; ".join(p for p in (res.notes, note) if p)
        return res
    finally:
        for h in quant_hooks:
            h.remove()
        if technique is not None:
            try:
                technique.teardown()
            except Exception:  # noqa: BLE001
                pass
        try:
            del model, tokenizer
        except Exception:  # noqa: BLE001
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def _needs_fallback(res: PerplexityResult) -> bool:
    """True if a 512-token run failed entirely to OOM (retry at 256)."""
    return res.num_chunks_evaluated == 0 and res.num_chunks_skipped_oom > 0


def run_perplexity_evaluation(
    config: BenchmarkConfig,
    output_csv: str | Path,
    models: Optional[Sequence[str]] = None,
    device=None,
    use_cuda: Optional[bool] = None,
) -> List[Dict[str, object]]:
    """Evaluate perplexity for the five real techniques across ``models``.

    For each (model, technique) pair we score at ``PERPLEXITY_CHUNK_LEN``.
    If a technique OOMs even at ``batch_size = 1`` and that chunk length,
    we retry once at ``PERPLEXITY_CHUNK_LEN_FALLBACK`` before recording the
    combination as ``"OOM -- not measured"`` rather than crashing the run.
    Which combinations succeeded, needed the fallback, or failed is logged.

    Results are written to ``output_csv`` and returned as a list of row
    dicts with :data:`PERPLEXITY_CSV_COLUMNS`.
    """
    models = list(models or DEFAULT_PERPLEXITY_MODELS)
    if use_cuda is None:
        use_cuda = config.use_cuda
    if device is None:
        device = get_torch_device(use_cuda=use_cuda)

    logger.info(
        "Perplexity evaluation on %s over %d model(s): %s",
        device, len(models), ", ".join(models),
    )
    logger.info(
        "PagedAttention and CPU Offloading are lossless -> perplexity "
        "equals Baseline FP16 by construction; not recomputed. "
        "NVFP4 / MiniKV / xKV are simulated -> no real model to run."
    )

    rows: List[Dict[str, object]] = []
    for model_name in models:
        is_tight = "tinyllama" in model_name.lower()
        smoke_note = (
            "harness check (untrained tiny model; absolute PPL not meaningful)"
            if "tiny-gpt2" in model_name.lower() else ""
        )
        for name, category, kind, quant_dtype in _real_technique_builders():
            note_parts: List[str] = []
            res: Optional[PerplexityResult] = None
            try:
                res = _evaluate_once(
                    name, kind, quant_dtype, model_name, device, use_cuda,
                    config, chunk_len=PERPLEXITY_CHUNK_LEN,
                )
                # Retry once at the reduced chunk length if 512 fully OOM'd.
                if _needs_fallback(res):
                    logger.warning(
                        "[%s @ %s] OOM at chunk_len=%d; retrying at %d.",
                        name, model_name, PERPLEXITY_CHUNK_LEN,
                        PERPLEXITY_CHUNK_LEN_FALLBACK,
                    )
                    res = _evaluate_once(
                        name, kind, quant_dtype, model_name, device, use_cuda,
                        config, chunk_len=PERPLEXITY_CHUNK_LEN_FALLBACK,
                    )
                    note_parts.append(
                        f"chunk_len reduced to {PERPLEXITY_CHUNK_LEN_FALLBACK} "
                        f"after OOM at {PERPLEXITY_CHUNK_LEN}"
                    )
            except Exception as exc:  # noqa: BLE001
                # Setup / first-forward OOM: retry once at the smaller size.
                if _is_oom(exc) and is_tight:
                    logger.warning(
                        "[%s @ %s] OOM during setup/forward at %d; retry %d.",
                        name, model_name, PERPLEXITY_CHUNK_LEN,
                        PERPLEXITY_CHUNK_LEN_FALLBACK,
                    )
                    try:
                        res = _evaluate_once(
                            name, kind, quant_dtype, model_name, device,
                            use_cuda, config,
                            chunk_len=PERPLEXITY_CHUNK_LEN_FALLBACK,
                        )
                        note_parts.append(
                            f"chunk_len reduced to "
                            f"{PERPLEXITY_CHUNK_LEN_FALLBACK} after OOM"
                        )
                    except Exception as exc2:  # noqa: BLE001
                        logger.error(
                            "[%s @ %s] failed at %d too: %s",
                            name, model_name, PERPLEXITY_CHUNK_LEN_FALLBACK,
                            exc2,
                        )
                else:
                    logger.exception(
                        "[%s @ %s] perplexity failed: %s",
                        name, model_name, exc,
                    )
                    note_parts.append(f"error: {type(exc).__name__}")

            # Assemble the CSV row (missing/failed runs -> "OOM -- not measured").
            if res is None or (
                res.num_chunks_evaluated == 0 and math.isnan(res.perplexity)
            ):
                skipped = res.num_chunks_skipped_oom if res else 0
                # Only genuine CUDA OOMs are labelled as such; other failures
                # (e.g. an incompatible model architecture) are labelled as
                # plain "not measured" so the CSV stays honest.
                note_parts.insert(
                    0, "OOM -- not measured" if skipped else "not measured"
                )
                rows.append({
                    "model": model_name,
                    "technique": name,
                    "category": category,
                    "num_chunks_evaluated": 0,
                    "num_chunks_skipped_oom": skipped,
                    "perplexity": "",
                    "notes": _join_notes(note_parts, smoke_note),
                })
                continue

            if res.notes:
                # Carry the per-technique explanation attached in
                # ``_evaluate_once`` (e.g. fused-QKV / equals-baseline) into the
                # CSV; without this it is silently dropped from Table IX.
                note_parts.insert(0, res.notes)
            if res.num_chunks_skipped_oom:
                note_parts.append(
                    f"{res.num_chunks_skipped_oom} chunk(s) skipped on OOM"
                )
            rows.append({
                "model": model_name,
                "technique": name,
                "category": category,
                "num_chunks_evaluated": res.num_chunks_evaluated,
                "num_chunks_skipped_oom": res.num_chunks_skipped_oom,
                "perplexity": round(res.perplexity, 4),
                "notes": _join_notes(note_parts, smoke_note),
            })

    write_perplexity_csv(rows, output_csv)
    return rows


def _join_notes(parts: Sequence[str], smoke_note: str = "") -> str:
    items = [p for p in parts if p]
    if smoke_note:
        items.append(smoke_note)
    return "; ".join(items)


def write_perplexity_csv(rows: Sequence[Dict[str, object]], path: str | Path) -> Path:
    """Write perplexity rows to ``path`` with the fixed column order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PERPLEXITY_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logger.info("Wrote %d perplexity rows to %s", len(rows), path)
    return path

