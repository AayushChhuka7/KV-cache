"""Tests for deterministic seeding."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kvbench.utils.seeding import seed_everything, get_global_seed


def test_seed_sets_python_random():
    seed_everything(7)
    import random
    v1 = [random.random() for _ in range(5)]
    seed_everything(7)
    v2 = [random.random() for _ in range(5)]
    assert v1 == v2


def test_seed_sets_numpy():
    seed_everything(13)
    import numpy as np
    a = np.random.rand(4).tolist()
    seed_everything(13)
    b = np.random.rand(4).tolist()
    assert a == b


def test_seed_records_global_seed():
    seed_everything(123)
    assert get_global_seed() == 123