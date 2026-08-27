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

artifact_root=artifacts/submissions/t10a_c1_filtered_k32
mkdir -p "$artifact_root"

python -m pytest -q \
  tests/test_extract.py \
  tests/test_submit.py \
  tests/test_vote_filter.py \
  tests/test_t10a_c1_submission.py \
  --junitxml="$artifact_root/tests.xml"

python -u src/generate.py \
  --config configs/t10a_prompt_improvement.json \
  --input data/deep_chal_math_leaderboard.csv \
  --output "$artifact_root/generations.jsonl" \
  --metadata "$artifact_root/run-metadata.json" \
  --engine vllm \
  --prompt-mode cot_boxed

python -m analysis.t10a_c1_submission \
  --input data/deep_chal_math_leaderboard.csv \
  --generations "$artifact_root/generations.jsonl" \
  --metadata "$artifact_root/run-metadata.json" \
  --t10a-config configs/t10a_prompt_improvement.json \
  --c1-config configs/t10a_c1_vote_filter.json \
  --output-dir "$artifact_root"

echo '{"event":"t10a_c1_submission_pipeline_complete"}'
