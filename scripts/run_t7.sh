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

artifact_root=artifacts/t7_rft_r2
data_root=data/rft_r2
suspect_root=data/suspect_set
mkdir -p "$artifact_root/calibration" "$data_root" "$suspect_root"

python -m pytest -q \
  tests/test_extract.py \
  tests/test_generate.py \
  tests/test_build_rft.py \
  tests/test_build_rft_r2.py \
  --junitxml="$artifact_root/tests.xml"

python -m src.build_rft_r2 prepare \
  --canonical data/canonical/train.csv \
  --rft-ids data/rft_pool_ids.txt \
  --r1-audit data/rft_r1/audit.csv \
  --t6-manifest artifacts/t6_1_sft_v1r/manifest.json \
  --adapter-root artifacts/t6_1_sft_v1r/adapters \
  --target-ids "$data_root/target_ids.txt" \
  --output "$artifact_root/preparation.json" \
  --expected-target-count 1801 \
  --seed 42

# The revised T7 explicitly forbids every unadopted T6/T6-1 adapter.
jq -e \
  '.status == "complete" and .generation_source.kind == "base" and .generation_source.adapter_path == null and .counts.target_questions == 1801 and .counts.expected_generations == 57632' \
  "$artifact_root/preparation.json" >/dev/null

# Short, separate throughput remeasurement.  These 32 prompts are not mixed
# into R2; the full generation below still supplies exactly k=32 per target.
python -u src/generate.py \
  --config configs/t7_rft_r2.json \
  --input data/canonical/train.csv \
  --ids-file "$data_root/target_ids.txt" \
  --output "$artifact_root/calibration/generations.jsonl" \
  --metadata "$artifact_root/calibration/run-metadata.json" \
  --engine vllm \
  --max-prompts 32 \
  --selection-seed 42

# Raw T5 generations remain untouched; R2 has its own resume-safe output.
python -u src/generate.py \
  --config configs/t7_rft_r2.json \
  --input data/canonical/train.csv \
  --ids-file "$data_root/target_ids.txt" \
  --output "$data_root/generations.jsonl" \
  --metadata "$data_root/run-metadata.json" \
  --engine vllm

python -m src.build_rft_r2 build \
  --canonical data/canonical/train.csv \
  --rft-ids data/rft_pool_ids.txt \
  --r1-audit data/rft_r1/audit.csv \
  --r1-sft data/rft_r1/sft.jsonl \
  --r1-generations artifacts/t5_rft_r1/generations.jsonl \
  --target-ids "$data_root/target_ids.txt" \
  --generations "$data_root/generations.jsonl" \
  --generation-metadata "$data_root/run-metadata.json" \
  --calibration-metadata "$artifact_root/calibration/run-metadata.json" \
  --config configs/t7_rft_r2.json \
  --preparation "$artifact_root/preparation.json" \
  --data-dir "$data_root" \
  --artifact-dir "$artifact_root" \
  --suspect-dir "$suspect_root" \
  --expected-n 32 \
  --expected-target-count 1801 \
  --seed 42

echo '{"event":"t7_manual_review_required","path":"data/suspect_set/sample20_review.template.json"}'
while [[ ! -f "$suspect_root/sample20_review.json" ]]; do
  sleep 30
done

python -m src.build_rft_r2 finalize-review \
  --template "$suspect_root/sample20_review.template.json" \
  --review "$suspect_root/sample20_review.json" \
  --suspect-manifest "$suspect_root/manifest.json" \
  --rft-manifest "$data_root/manifest.json" \
  --artifact-manifest "$artifact_root/manifest.json" \
  --output "$suspect_root/sample20_review.md"

echo '{"event":"t7_cumulative_record_update_required","path":"docs/strategy/execution-prompts.md"}'
while [[ ! -f "$artifact_root/record-table.ready" ]]; do
  sleep 30
done

python -m src.build_rft_r2 finalize-record-table \
  --document docs/strategy/execution-prompts.md \
  --rft-manifest "$data_root/manifest.json" \
  --suspect-manifest "$suspect_root/manifest.json" \
  --artifact-manifest "$artifact_root/manifest.json"

tar -czf "$artifact_root/t7-final-metadata.tgz" \
  "$artifact_root/preparation.json" \
  "$artifact_root/calibration/run-metadata.json" \
  "$artifact_root/metrics.json" \
  "$artifact_root/manifest.json" \
  "$data_root/manifest.json" \
  "$suspect_root/manifest.json" \
  "$suspect_root/sample20_review.md"

echo '{"event":"t7_complete"}'
