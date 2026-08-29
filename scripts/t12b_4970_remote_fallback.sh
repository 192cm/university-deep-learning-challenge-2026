#!/usr/bin/env bash
set -euo pipefail

cd /workspace
python_bin=/venv/main/bin/python
stage="${1:?expected stage1, stage2, dev-stage1, or dev-stage2}"
needs_repair_input=false

case "$stage" in
  stage1)
    artifact_root=artifacts/t12b_4970_override/leaderboard-label-blind
    config=configs/t4_output_contract.json
    output="$artifact_root/fallback-stage1-generations.jsonl"
    metadata="$artifact_root/fallback-stage1-metadata.json"
    seed=42
    ;;
  stage2)
    artifact_root=artifacts/t12b_4970_override/leaderboard-label-blind
    config=configs/t12b_fallback_repair.json
    output="$artifact_root/fallback-stage2-generations.jsonl"
    metadata="$artifact_root/fallback-stage2-metadata.json"
    seed=12042
    needs_repair_input=true
    ;;
  dev-stage1)
    artifact_root=artifacts/t12b_4970_override/development
    config=configs/t4_output_contract.json
    output="$artifact_root/fallback-stage1-generations.jsonl"
    metadata="$artifact_root/fallback-stage1-metadata.json"
    seed=42
    ;;
  dev-stage2)
    artifact_root=artifacts/t12b_4970_override/development
    config=configs/t12b_fallback_repair.json
    output="$artifact_root/fallback-stage2-generations.jsonl"
    metadata="$artifact_root/fallback-stage2-metadata.json"
    seed=12042
    needs_repair_input=true
    ;;
  *)
    echo "unknown fallback stage: $stage" >&2
    exit 2
    ;;
esac
input="$artifact_root/fallback-questions.csv"

if [[ "$needs_repair_input" == true ]]; then
  repair_input="$artifact_root/fallback-repair-questions.csv"
  repair_audit="$artifact_root/fallback-repair-input-audit.json"
  "$python_bin" -m src.build_fallback_repair_input \
    --input "$input" \
    --output "$repair_input" \
    --audit "$repair_audit"
  input="$repair_input"
fi

if [[ ! -s "$input" ]]; then
  echo "fallback input is missing or empty: $input" >&2
  exit 3
fi
exec env CUDA_VISIBLE_DEVICES=0 "$python_bin" -m src.generate \
  --config "$config" \
  --input "$input" \
  --output "$output" \
  --metadata "$metadata" \
  --engine hf \
  --n 1 \
  --seed "$seed" \
  --max-new-tokens 2048
