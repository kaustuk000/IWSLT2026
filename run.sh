#!/bin/bash
set -e

echo "=== Step 1: uv sync ==="
uv sync

echo "=== Step 2: Running training ==="
cd e2e_v1
uv run python train.py --hf_token "$HF_TOKEN" 2>&1 | tee ../train_output.log

echo "=== Done ==="
