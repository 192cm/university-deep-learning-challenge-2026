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

artifact_root=artifacts/t8_self_consistency
adaptive_root="$artifact_root/adaptive"
mkdir -p "$artifact_root" "$adaptive_root/stage1" "$adaptive_root/stage2"

python -m pytest -q \
  tests/test_extract.py \
  tests/test_evaluate.py \
  tests/test_generate.py \
  tests/test_self_consistency.py \
  --junitxml="$artifact_root/tests.xml"

# One immutable k=32 pool supplies paired prefixes for the 4/8/16/32 sweep.
python -u src/generate.py \
  --config configs/t8_self_consistency.json \
  --input data/canonical/train.csv \
  --ids-file "$artifact_root/holdout_union_ids.txt" \
  --output "$artifact_root/generations.jsonl" \
  --metadata "$artifact_root/run-metadata.json" \
  --engine vllm

# Execute the production-shaped adaptive path separately: four samples for all
# questions, then 28 more only for label-blind disagreement/invalid prefixes.
python -u src/generate.py \
  --config configs/t8_self_consistency.json \
  --input data/canonical/train.csv \
  --ids-file "$artifact_root/holdout_union_ids.txt" \
  --output "$adaptive_root/stage1/generations.jsonl" \
  --metadata "$adaptive_root/stage1/run-metadata.json" \
  --engine vllm \
  --n 4 \
  --seed 42004

python -m src.self_consistency prepare-stage2 \
  --stage1-generations "$adaptive_root/stage1/generations.jsonl" \
  --union-ids "$artifact_root/holdout_union_ids.txt" \
  --output-ids "$adaptive_root/stage2/ids.txt" \
  --output-json "$adaptive_root/stage2/preparation.json" \
  --initial-k 4 \
  --continuation-samples 28

python -u src/generate.py \
  --config configs/t8_self_consistency.json \
  --input data/canonical/train.csv \
  --ids-file "$adaptive_root/stage2/ids.txt" \
  --output "$adaptive_root/stage2/generations.jsonl" \
  --metadata "$adaptive_root/stage2/run-metadata.json" \
  --engine vllm \
  --n 28 \
  --seed 42032

python -m src.self_consistency finalize \
  --config configs/t8_self_consistency.json \
  --canonical data/canonical/train.csv \
  --union-ids "$artifact_root/holdout_union_ids.txt" \
  --split random_holdout=data/splits/random_holdout.csv \
  --split template_holdout=data/splits/template_holdout.csv \
  --split hard_diagnostic=data/splits/hard_diagnostic.csv \
  --split format_diagnostic=data/splits/format_diagnostic.csv \
  --generations "$artifact_root/generations.jsonl" \
  --metadata "$artifact_root/run-metadata.json" \
  --stage1-generations "$adaptive_root/stage1/generations.jsonl" \
  --stage1-metadata "$adaptive_root/stage1/run-metadata.json" \
  --stage2-preparation "$adaptive_root/stage2/preparation.json" \
  --stage2-ids "$adaptive_root/stage2/ids.txt" \
  --stage2-generations "$adaptive_root/stage2/generations.jsonl" \
  --stage2-metadata "$adaptive_root/stage2/run-metadata.json" \
  --greedy-generations artifacts/t4_output_contract/generations.jsonl \
  --greedy-metadata artifacts/t4_output_contract/run-metadata.json \
  --output-dir "$artifact_root"

tar -czf "$artifact_root/t8-final-metadata.tgz" \
  "$artifact_root/sweep.json" \
  "$artifact_root/curve.md" \
  "$artifact_root/final_config.json" \
  "$artifact_root/manifest.json" \
  "$artifact_root/tests.xml" \
  "$adaptive_root/stage2/preparation.json" \
  "$adaptive_root/stage2/ids.txt"

echo '{"event":"t8_complete"}'
