#!/usr/bin/env bash
# Install dependencies for the KV cache benchmarking framework.
# Works on Linux / macOS / WSL.

set -euo pipefail

PYTHON="${PYTHON:-python3}"

echo "==> Creating virtual environment (.venv) if it does not exist"
if [ ! -d ".venv" ]; then
  $PYTHON -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrading pip"
pip install --upgrade pip wheel setuptools

echo "==> Installing CPU-first dependencies"
pip install -r requirements.txt || true

echo "==> Detecting CUDA"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA GPU detected. vLLM will be used for PagedAttention."
else
  echo "No NVIDIA GPU detected. The framework will run in CPU simulation mode."
fi

echo "==> Installing PyTorch (auto-detect CUDA)"
if command -v nvidia-smi >/dev/null 2>&1; then
  pip install torch --index-url https://download.pytorch.org/whl/cu121 || pip install torch
else
  pip install torch --index-url https://download.pytorch.org/whl/cpu || pip install torch
fi

echo "==> Installing transformers, accelerate, and friends"
pip install transformers accelerate sentencepiece tqdm psutil pyyaml tabulate matplotlib pandas seaborn

echo "==> Trying to install vLLM and bitsandbytes (Linux/WSL only)"
if [ "$(uname -s)" = "Linux" ]; then
  pip install vllm bitsandbytes || echo "vLLM install failed; PagedAttention experiments will be skipped."
else
  echo "Non-Linux platform detected; skipping vLLM and bitsandbytes."
fi

echo "==> Installing the kvbench package in editable mode"
pip install -e .

echo "==> Done. Activate with:  source .venv/bin/activate"
