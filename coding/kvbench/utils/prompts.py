"""Prompt construction utilities.

The framework uses deterministic prompts so that every technique is
exercised on identical inputs. Long contexts are built by repeating the
base prompts in a round-robin fashion until the target token length is
reached.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, List, Sequence

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedTokenizerBase


def build_prompt(
    tokenizer: "PreTrainedTokenizerBase",
    base_prompts: Sequence[str],
    target_tokens: int,
    seed: int = 0,
) -> str:
    """Build a deterministic prompt of approximately ``target_tokens`` tokens.

    Parameters
    ----------
    tokenizer:
        Any HuggingFace tokenizer with ``__call__`` returning ``input_ids``.
    base_prompts:
        Source sentences that will be repeated.
    target_tokens:
        Approximate token count to reach.
    seed:
        Random seed for which base prompts to pick (if more than one).

    Returns
    -------
    str
        A single concatenated prompt string.
    """
    if target_tokens <= 0:
        return ""
    rng = random.Random(seed)

    chunks: List[str] = list(base_prompts)
    if not chunks:
        chunks = ["The quick brown fox jumps over the lazy dog. "]

    pieces: List[str] = []
    total = 0
    i = 0
    while total < target_tokens:
        s = chunks[i % len(chunks)]
        pieces.append(s)
        total += len(tokenizer.encode(s, add_special_tokens=False))
        i += 1
    text = " ".join(pieces)
    # Trim if we overshot.
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > target_tokens:
        text = tokenizer.decode(ids[:target_tokens], skip_special_tokens=True)
    return text


def repeat_to_length(text: str, tokenizer: "PreTrainedTokenizerBase", target_tokens: int) -> str:
    """Repeat ``text`` (with separators) until it reaches ``target_tokens`` tokens."""
    if target_tokens <= 0:
        return ""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return text
    pieces = [text]
    while True:
        joined = " ".join(pieces + [text])
        cur = len(tokenizer.encode(joined, add_special_tokens=False))
        if cur >= target_tokens:
            return joined
        pieces.append(text)
