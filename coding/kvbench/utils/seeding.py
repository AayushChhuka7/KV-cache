"""Random seeding helpers.

Ensures that ``random``, ``numpy``, and ``torch`` (CPU + CUDA) are all
seeded with the same value so that every trial of an experiment is
deterministic and reproducible.
"""

from __future__ import annotations

import os
import random
from typing import Optional

_GLOBAL_SEED: Optional[int] = None


def get_global_seed() -> Optional[int]:
    """Return the seed most recently passed to :func:`seed_everything`."""
    return _GLOBAL_SEED


def seed_everything(seed: int, deterministic: bool = False) -> int:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA).

    Parameters
    ----------
    seed:
        The integer seed to use everywhere.
    deterministic:
        If True, also enable PyTorch deterministic algorithms and disable
        the CUDA benchmark heuristic. This makes measurements perfectly
        reproducible at the cost of some performance.
    """
    global _GLOBAL_SEED
    _GLOBAL_SEED = int(seed)

    # Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
        else:
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    return seed
