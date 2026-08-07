"""Tests for analytic KV cache size estimation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kvbench.profiling.memory import cache_size_from_shape
from kvbench.utils.models import KVShape


def make_shape(L=32, H=8, D=64, B=2):
    return KVShape(num_layers=L, num_kv_heads=H, head_dim=D, bytes_per_element=B)


def test_cache_size_zero_when_seq_zero():
    s = make_shape()
    assert cache_size_from_shape(s, batch=1, seq_len=0) == 0


def test_cache_size_linear_in_seq():
    s = make_shape(L=32, H=8, D=64, B=2)
    one = cache_size_from_shape(s, batch=1, seq_len=100)
    hundred = cache_size_from_shape(s, batch=1, seq_len=10_000)
    assert hundred > one
    # The KV cache should scale linearly with sequence length.
    ratio = hundred / one
    assert abs(ratio - 100) < 1e-6


def test_cache_size_linear_in_batch():
    s = make_shape()
    a = cache_size_from_shape(s, batch=1, seq_len=1024)
    b = cache_size_from_shape(s, batch=4, seq_len=1024)
    assert abs(b / a - 4) < 1e-6


def test_cache_size_factor_in_precision():
    s16 = make_shape(B=2)
    s8 = make_shape(B=1)
    a = cache_size_from_shape(s16, batch=1, seq_len=1024)
    b = cache_size_from_shape(s8, batch=1, seq_len=1024)
    assert abs(a / b - 2) < 1e-6