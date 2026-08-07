"""KV cache optimization techniques.

Each technique implements :class:`KVTechnique`, a small interface that
captures everything a benchmark sweep needs:

  * a stable name and category,
  * a method that loads / configures the model,
  * a method that runs a single benchmark trial,
  * an analytic KV cache size for a given (batch, seq_len).

The framework runs all techniques against the same prompts, the same
seed, and the same number of trials, and aggregates the results.
"""

from .base import (
    BenchmarkResult,
    KVTechnique,
    TechniqueSpec,
)
from .baseline import BaselineKV
from .quantized import FP8SimulatedKV, Int8SimulatedKV
from .gqa import GQAModel
from .paged import PagedAttentionVLLM
from .offloading import CPUOffloading
from .compression import LowRankCompression
from .simulations import (
    NVFP4Simulator,
    MiniKVSimulator,
    XKVSimulator,
)

__all__ = [
    "BenchmarkResult",
    "KVTechnique",
    "TechniqueSpec",
    "BaselineKV",
    "FP8SimulatedKV",
    "Int8SimulatedKV",
    "GQAModel",
    "PagedAttentionVLLM",
    "CPUOffloading",
    "LowRankCompression",
    "NVFP4Simulator",
    "MiniKVSimulator",
    "XKVSimulator",
]
