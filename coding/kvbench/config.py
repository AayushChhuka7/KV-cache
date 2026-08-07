"""Global configuration for the kvbench framework.

All experiments read from this single source of truth. Override values
either by editing the file or by passing a YAML file through ``--config``.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

DEFAULT_SEED = 42

# Default model: a small, fast GQA model that fits on commodity hardware
# and exercises both MHA-style (early) and GQA-style (modern) behaviors.
# Replace with any HuggingFace causal LM. For example:
#   - "meta-llama/Llama-3.2-1B"            (GQA, small)
#   - "meta-llama/Llama-3.2-3B"            (GQA, larger)
#   - "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (GQA)
#   - "gpt2"                               (MHA, very small, fast)
DEFAULT_MODEL = os.environ.get("KVBENCH_MODEL", "sshleifer/tiny-gpt2")

# If set, use a larger model that benefits from GQA.
DEFAULT_GQA_MODEL = os.environ.get(
    "KVBENCH_GQA_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

# Default non-GQA (MHA) model used for baseline comparisons.
DEFAULT_MHA_MODEL = os.environ.get("KVBENCH_MHA_MODEL", "gpt2")

# Number of trials to average over for each experiment. Set to 1 for
# quick smoke tests, 3-5 for real research runs.
DEFAULT_TRIALS = int(os.environ.get("KVBENCH_TRIALS", "3"))

# Warmup steps are issued before timing begins, to load weights and to
# allow any JIT, cuDNN autotuner, or memory pool warm-up to settle.
DEFAULT_WARMUP_STEPS = int(os.environ.get("KVBENCH_WARMUP", "2"))

# Number of decode tokens to generate per request.
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("KVBENCH_MAX_NEW", "32"))

# Context lengths to sweep when scaling. These are intentionally modest
# to fit on consumer GPUs; adjust to your hardware.
DEFAULT_CONTEXT_LENGTHS: List[int] = [128, 512, 2048, 4096]

# Batch sizes to sweep.
DEFAULT_BATCH_SIZES: List[int] = [1, 2, 4]

# Whether to use CUDA when available.
DEFAULT_USE_CUDA = True

# Sample prompts used to populate the context window. The framework
# generates longer inputs by repeating and concatenating these.
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


@dataclass
class BenchmarkConfig:
    """All knobs controlling an experiment sweep."""

    seed: int = DEFAULT_SEED
    trials: int = DEFAULT_TRIALS
    warmup_steps: int = DEFAULT_WARMUP_STEPS
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS

    # Models
    model: str = DEFAULT_MODEL
    mha_model: str = DEFAULT_MHA_MODEL
    gqa_model: str = DEFAULT_GQA_MODEL

    # Hardware
    use_cuda: bool = DEFAULT_USE_CUDA
    dtype: str = "float16"  # one of float16, bfloat16, float32

    # Sweeps
    context_lengths: List[int] = field(
        default_factory=lambda: list(DEFAULT_CONTEXT_LENGTHS)
    )
    batch_sizes: List[int] = field(
        default_factory=lambda: list(DEFAULT_BATCH_SIZES)
    )

    # Prompts
    prompts: List[str] = field(default_factory=lambda: list(DEFAULT_PROMPTS))

    # Output
    output_dir: str = "results"
    save_raw: bool = True
    generate_plots: bool = True

    # Whether to run the simulated experiments (NVFP4, MiniKV, xKV)
    # whose kernels are proprietary. These still produce
    # research-quality numbers; the difference is that the
    # measurement is based on the analytical formula rather than
    # a custom kernel.
    run_simulations: bool = True

    # Random source for seedable numpy / torch / random
    def reseed(self) -> None:
        random.seed(self.seed)
        try:
            import numpy as np

            np.random.seed(self.seed)
        except Exception:
            pass
        try:
            import torch

            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        except Exception:
            pass

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)


# Convenience constant for ad-hoc usage
GLOBAL_CONFIG = BenchmarkConfig()
