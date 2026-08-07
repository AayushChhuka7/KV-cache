#!/usr/bin/env bash
# Run the full set of KV cache benchmark experiments.
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate 2>/dev/null || true

OUT_DIR="results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

python scripts/run_all.py --output "$OUT_DIR" "$@"
