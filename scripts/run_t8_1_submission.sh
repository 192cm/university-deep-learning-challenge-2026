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

artifact_root=artifacts/submissions/t8_1_rft_majority_k32
adapter=artifacts/t6_sft_v1/adapters/rft_r1
mkdir -p "$artifact_root"

python -m pytest -q \
  tests/test_extract.py \
  tests/test_evaluate.py \
  tests/test_generate.py \
  tests/test_self_consistency.py \
  --junitxml="$artifact_root/tests.xml"

python -u src/generate.py \
  --config configs/t8_1_rft_self_consistency.json \
  --input data/deep_chal_math_leaderboard.csv \
  --output "$artifact_root/generations.jsonl" \
  --metadata "$artifact_root/run-metadata.json" \
  --engine vllm \
  --adapter "$adapter"

echo '{"event":"t8_1_submission_generation_complete"}'
