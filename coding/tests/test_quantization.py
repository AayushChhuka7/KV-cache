"""Tests for the int8 / fp8 quantization helpers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from kvbench.techniques.quantized import (
    dequantize_int8_per_channel,
    quantize_int8_per_channel,
)


def test_int8_roundtrip_within_tolerance():
    torch.manual_seed(0)
    x = torch.randn(2, 16, 64) * 4.0
    q, scale = quantize_int8_per_channel(x)
    assert q.dtype == torch.uint8
    assert q.shape == x.shape
    rec = dequantize_int8_per_channel(q, scale)
    diff = (rec - x).abs().max().item()
    # Per-channel symmetric int8 with up to 127 levels: max error is
    # scale / 2. With scale ~ max(|x|)/127 ~ 4/127 ~= 0.031, the error
    # is bounded by ~ 0.031. We allow a slightly larger tolerance for
    # edge channels with very small scale.
    assert diff < 0.2


def test_int8_storage_is_one_byte():
    torch.manual_seed(0)
    x = torch.randn(1, 16, 64)
    q, _ = quantize_int8_per_channel(x)
    assert q.element_size() == 1