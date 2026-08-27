#!/usr/bin/env bash
set -euo pipefail

cd /workspace
source /venv/main/bin/activate
mkdir -p artifacts/t5_rft_r1 data/rft_r1
exec > >(tee -a artifacts/t5_rft_r1/finalize.log) 2>&1

python -u -m src.build_rft \
  --canonical data/canonical/train.csv \
  --ids data/rft_pool_ids.txt \
  --generations artifacts/t5_rft_r1/generations.jsonl \
  --generation-metadata artifacts/t5_rft_r1/run-metadata.json \
  --config configs/t5_rft_r1.json \
  --data-output-dir data/rft_r1 \
  --artifact-output-dir artifacts/t5_rft_r1 \
  --expected-n 16
