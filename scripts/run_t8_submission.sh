#!/usr/bin/env bash
set -euo pipefail

cd /workspace
source /venv/main/bin/activate

export HF_HOME=/workspace/.hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export VLLM_WORKER_MULTIPROC_METHOD=spawn

artifact_root=artifacts/submissions/t8_majority_k32
mkdir -p "$artifact_root"

python -u src/generate.py \
  --config configs/t8_self_consistency.json \
  --input data/deep_chal_math_leaderboard.csv \
  --output "$artifact_root/generations.jsonl" \
  --metadata "$artifact_root/run-metadata.json" \
  --engine vllm

python -m src.submit \
  --input data/deep_chal_math_leaderboard.csv \
  --generations "$artifact_root/generations.jsonl" \
  --config configs/t8_self_consistency.json \
  --metadata "$artifact_root/run-metadata.json" \
  --output "$artifact_root/submission-prepared.json" \
  --k 32

echo '{"event":"t8_submission_generation_complete"}'
