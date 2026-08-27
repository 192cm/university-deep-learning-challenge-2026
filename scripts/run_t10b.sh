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

artifact_root=artifacts/t10b_prompt_diversity
config=configs/t10b_prompt_diversity.json
union_ids=artifacts/t8_self_consistency/holdout_union_ids.txt
mkdir -p "$artifact_root/arm_C"

python -m src.prompt_diversity snapshot-invariants \
  --config "$config" \
  --output "$artifact_root/invariant-snapshot.json"

python -m src.prompt_diversity preflight \
  --config "$config" \
  --output "$artifact_root/preflight.json"

python -m pytest -q \
  tests/test_extract.py \
  tests/test_evaluate.py \
  tests/test_generate.py \
  tests/test_self_consistency.py \
  tests/test_submit.py \
  tests/test_vote_filter.py \
  tests/test_prompt_improvement.py \
  tests/test_prompt_diversity.py \
  --junitxml="$artifact_root/tests.xml"

python -u src/generate.py \
  --config "$config" \
  --input data/canonical/train.csv \
  --ids-file "$union_ids" \
  --output "$artifact_root/arm_C/generations.jsonl" \
  --metadata "$artifact_root/arm_C/run-metadata.json" \
  --engine vllm

python -m src.prompt_diversity evaluate \
  --config "$config" \
  --tests-xml "$artifact_root/tests.xml"

python -m src.prompt_diversity verify-snapshot \
  --snapshot "$artifact_root/invariant-snapshot.json"

tar -czf "$artifact_root/t10b-final-metadata.tgz" \
  "$artifact_root/invariant-snapshot.json" \
  "$artifact_root/preflight.json" \
  "$artifact_root/prediction-freeze.json" \
  "$artifact_root/comparison.json" \
  "$artifact_root/comparison.md" \
  "$artifact_root/inter-prompt-agreement.json" \
  "$artifact_root/extraction-path-analysis.json" \
  "$artifact_root/filter-interaction.json" \
  "$artifact_root/runtime.json" \
  "$artifact_root/final_config.json" \
  "$artifact_root/manifest.json" \
  "$artifact_root/tests.xml"

echo '{"event":"t10b_complete"}'
