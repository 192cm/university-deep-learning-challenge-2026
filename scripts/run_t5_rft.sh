#!/usr/bin/env bash
set -euo pipefail

cd /workspace
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/workspace/.hf_home
export TOKENIZERS_PARALLELISM=true
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p artifacts/t5_rft_r1
exec > >(tee -a artifacts/t5_rft_r1/generation.log) 2>&1

python -u src/generate.py \
  --config configs/t5_rft_r1.json \
  --input data/canonical/train.csv \
  --ids-file data/rft_pool_ids.txt \
  --output artifacts/t5_rft_r1/generations.jsonl \
  --metadata artifacts/t5_rft_r1/run-metadata.json \
  --engine vllm
