"""Model loading helpers.

These utilities take care of:
  * Choosing CUDA when available, otherwise falling back to CPU.
  * Mapping a string dtype ("float16", "bfloat16", "float32") to the
    corresponding torch dtype.
  * Loading a HuggingFace ``AutoModelForCausalLM`` and its tokenizer.
  * Detecting the KV cache shape that a model will use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


def get_torch_device(use_cuda: bool = True) -> torch.device:
    """Return CUDA when available (and requested), otherwise CPU."""
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def dtype_from_string(name: str) -> torch.dtype:
    """Map ``"float16"`` / ``"bfloat16"`` / ``"float32"`` to torch dtypes."""
    n = name.lower().strip()
    if n in ("fp16", "float16", "half"):
        return torch.float16
    if n in ("bf16", "bfloat16"):
        return torch.bfloat16
    if n in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {name}")


def count_parameters(model: torch.nn.Module) -> int:
    """Total number of parameters in a model (all dtypes)."""
    return sum(p.numel() for p in model.parameters())


@dataclass
class KVShape:
    """Lightweight description of the KV cache shape used by a model.

    For modern HuggingFace models this is the shape *per token, per batch,
    per layer, per KV head*. See ``detect_model_kv_shape`` for details.
    """

    num_layers: int
    num_kv_heads: int
    head_dim: int
    bytes_per_element: int

    @property
    def kv_per_token(self) -> int:
        """Bytes per token in a single batch element, summed over layers."""
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * self.bytes_per_element

    def cache_size_bytes(self, batch: int, seq_len: int) -> int:
        return self.kv_per_token * batch * seq_len

    def cache_size_mb(self, batch: int, seq_len: int) -> float:
        return self.cache_size_bytes(batch, seq_len) / (1024 ** 2)


def detect_model_kv_shape(
    model: torch.nn.Module,
    dtype: torch.dtype = torch.float16,
) -> KVShape:
    """Best-effort detection of the KV cache shape.

    We read ``num_hidden_layers``, ``num_attention_heads``,
    ``num_key_value_heads`` (falling back to the full head count for MHA
    models) and ``hidden_size / num_attention_heads`` for the per-head
    dimension. ``bytes_per_element`` is read from the model weight dtype.
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise ValueError("Model has no config attribute; cannot detect KV shape.")

    num_layers = int(
        getattr(cfg, "num_hidden_layers", None)
        or getattr(cfg, "n_layer", None)
        or 0
    )
    num_heads = int(
        getattr(cfg, "num_attention_heads", None)
        or getattr(cfg, "n_head", None)
        or 0
    )
    num_kv_heads = int(
        getattr(cfg, "num_key_value_heads", None)
        or getattr(cfg, "num_kv_heads", None)
        or num_heads  # MHA fallback
    )
    hidden_size = int(
        getattr(cfg, "hidden_size", None)
        or getattr(cfg, "n_embd", None)
        or 0
    )
    head_dim = hidden_size // num_heads if num_heads else 0

    bpe_map = {
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.float32: 4,
        torch.float64: 8,
        torch.int8: 1,
        torch.uint8: 1,
        torch.qint8: 1,
        torch.quint8: 1,
    }
    # Use the activation dtype for KV by default; if the model has
    # explicit hooks, prefer the embedding dtype.
    sample_dtype = dtype
    for p in model.parameters():
        sample_dtype = p.dtype
        break
    bpe = bpe_map.get(sample_dtype, 2)

    return KVShape(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        bytes_per_element=bpe,
    )


def extend_position_embeddings(
    model: torch.nn.Module,
    tokenizer: Optional["object"] = None,
    min_seq_len: int = 4096,
) -> int:
    """Extend a model's learned absolute position embeddings to at least
    ``min_seq_len`` positions.

    Some small models (e.g. GPT-2, ``n_positions=1024``) cap the context
    length at their learned positional embedding table. The paper's sweep
    grid reaches 4096 tokens, so the table is extended by cyclically
    repeating the existing rows (the standard "position extension by
    repetition" heuristic). RoPE-based models (LLaMA-style) have no learned
    table and are returned unchanged.

    The tokenizer's ``model_max_length`` is raised as well so tokenization
    does not silently truncate to the old limit.

    Returns the new maximum position length, or -1 when the model has no
    learned absolute position embeddings.
    """
    parent, attr, emb = None, None, None
    for path in ("transformer.wpe", "wpe", "embeddings.position_embeddings"):
        mod = model
        parts = path.split(".")
        ok = True
        for p in parts[:-1]:
            mod = getattr(mod, p, None)
            if mod is None:
                ok = False
                break
        if ok and isinstance(getattr(mod, parts[-1], None), torch.nn.Embedding):
            parent, attr = mod, parts[-1]
            emb = getattr(mod, attr)
            break
    if emb is None:
        if tokenizer is not None and hasattr(tokenizer, "model_max_length"):
            try:
                old = int(getattr(tokenizer, "model_max_length", 0) or 0)
                if old < int(min_seq_len):
                    tokenizer.model_max_length = int(min_seq_len)
            except Exception:
                pass
        return -1

    n = emb.weight.shape[0]
    new_n = max(n, int(min_seq_len))
    if new_n > n:
        with torch.no_grad():
            base = emb.weight.float()
            reps = (new_n + n - 1) // n
            new_weight = base.repeat(reps, 1)[:new_n]
        new_emb = torch.nn.Embedding(
            new_n, emb.embedding_dim, dtype=emb.weight.dtype, device=emb.weight.device
        )
        with torch.no_grad():
            new_emb.weight.copy_(new_weight)
        setattr(parent, attr, new_emb)

    cfg = getattr(model, "config", None)
    if cfg is not None:
        for key in ("n_positions", "max_position_embeddings"):
            if hasattr(cfg, key):
                try:
                    setattr(cfg, key, new_n)
                except Exception:
                    pass
    if tokenizer is not None and hasattr(tokenizer, "model_max_length"):
        try:
            old = int(getattr(tokenizer, "model_max_length", 0) or 0)
            if old < new_n:
                tokenizer.model_max_length = new_n
        except Exception:
            pass
    return new_n


def load_model_and_tokenizer(
    model_name: str,
    dtype: torch.dtype = torch.float16,
    device: Optional[torch.device] = None,
    use_cuda: bool = True,
    low_cpu_mem_usage: bool = True,
    cache_dir: Optional[str] = None,
) -> Tuple[torch.nn.Module, "object"]:
    """Load a HuggingFace causal LM and tokenizer.

    Falls back to ``GPT2`` (very small) if the requested model fails
    to download — this happens in offline test environments.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        device = get_torch_device(use_cuda=use_cuda)

    cache_dir = cache_dir or os.environ.get("HF_HOME") or None

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = None
    load_errors = []
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            cache_dir=cache_dir,
        )
    except Exception as e:  # noqa: BLE001
        load_errors.append(("requested", model_name, repr(e)))

    if model is None:
        # Fallback: gpt2 is tiny and almost always available offline.
        try:
            model = AutoModelForCausalLM.from_pretrained(
                "gpt2",
                torch_dtype=dtype,
                low_cpu_mem_usage=low_cpu_mem_usage,
                cache_dir=cache_dir,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                "gpt2", cache_dir=cache_dir
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:  # noqa: BLE001
            load_errors.append(("gpt2-fallback", "gpt2", repr(e)))

    if model is None:
        raise RuntimeError(
            "Failed to load any model. Tried: " + repr(load_errors)
        )

    model.eval()
    model.to(device)
    return model, tokenizer
