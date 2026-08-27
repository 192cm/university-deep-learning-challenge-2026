#!/bin/bash
set -euo pipefail

cd /workspace
source /venv/main/bin/activate

export HF_HOME=/workspace/.hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export VLLM_WORKER_MULTIPROC_METHOD=spawn

root=artifacts/t6_sft_v1
config=configs/t6_sft_v1.json
mkdir -p "$root/adapters" "$root/calibration-probes"

python -m pytest -q \
  tests/test_extract.py \
  tests/test_evaluate.py \
  tests/test_generate.py \
  tests/test_train_sft.py \
  tests/test_finalize_t6.py \
  --junitxml="$root/tests.xml"

python src/finalize_t6.py base \
  --t4-metrics artifacts/t4_output_contract/metrics_c.json \
  --output "$root/base/metrics.json"

prepare_cache() {
  local experiment=$1
  shift
  local args=()
  local input
  for input in "$@"; do
    args+=(--input "$input")
  done
  python src/train_sft.py prepare \
    --config "$config" \
    "${args[@]}" \
    --cache "$root/$experiment/cache" \
    --metadata "$root/$experiment/cache-metadata.json"
}

prepare_cache answer_only data/answer_only/sft.jsonl
prepare_cache external_cot data/external_cot/sft.jsonl
prepare_cache rft_r1 data/rft_r1/sft.jsonl
prepare_cache rft_external data/rft_r1/sft.jsonl data/external_cot/sft.jsonl

if [[ ! -f "$root/calibration.json" ]] || ! jq -e '.status == "complete"' "$root/calibration.json" >/dev/null; then
  python src/train_sft.py calibrate \
    --config "$config" \
    --cache "$root/rft_external/cache" \
    --cache-metadata "$root/rft_external/cache-metadata.json" \
    --output "$root/calibration.json" \
    --work-dir "$root/calibration-probes" \
    --environment environment.json
fi

run_experiment() {
  local experiment=$1
  local experiment_root="$root/$experiment"
  local adapter="$root/adapters/$experiment"
  python src/train_sft.py train \
    --config "$config" \
    --cache "$experiment_root/cache" \
    --cache-metadata "$experiment_root/cache-metadata.json" \
    --calibration "$root/calibration.json" \
    --output "$experiment_root/training-metrics.json" \
    --work-dir "$experiment_root/trainer" \
    --adapter-dir "$adapter" \
    --experiment "$experiment"

  mkdir -p "$experiment_root/evaluation"
  if [[ ! -f "$experiment_root/evaluation/run-metadata.json" ]] || \
     ! jq -e '.status == "complete"' "$experiment_root/evaluation/run-metadata.json" >/dev/null; then
    python src/generate.py \
      --config configs/t4_output_contract.json \
      --input data/canonical/train.csv \
      --ids-file artifacts/t3_baseline/holdout_union_ids.txt \
      --output "$experiment_root/evaluation/generations.jsonl" \
      --metadata "$experiment_root/evaluation/run-metadata.json" \
      --engine vllm \
      --adapter "$adapter"
  fi

  python src/finalize_t6.py experiment \
    --name "$experiment" \
    --config "$config" \
    --generations "$experiment_root/evaluation/generations.jsonl" \
    --generation-metadata "$experiment_root/evaluation/run-metadata.json" \
    --training-metrics "$experiment_root/training-metrics.json" \
    --adapter "$adapter" \
    --output "$experiment_root/metrics.json"
}

run_experiment answer_only
run_experiment external_cot
run_experiment rft_r1
run_experiment rft_external

python src/finalize_t6.py finalize \
  --root "$root" \
  --config "$config" \
  --calibration "$root/calibration.json" \
  --environment environment.json \
  --answer-only-manifest data/answer_only/manifest.json \
  --rft-audit data/rft_r1/audit.csv

python - <<'PY'
import json
from pathlib import Path

manifest = json.loads(Path("artifacts/t6_sft_v1/manifest.json").read_text())
print(json.dumps({"event": "t6_complete", "decision": manifest["decision"]}, ensure_ascii=False))
PY
