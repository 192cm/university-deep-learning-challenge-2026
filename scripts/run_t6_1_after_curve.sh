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

root=artifacts/t6_1_sft_v1r
base_config=configs/t6_1_sft_v1r.json
resolved="$root/resolved-configs"

jq -e '.status == "complete" and (.candidates | length) == 6' \
  "$root/checkpoint-curve.json" >/dev/null

materialize_checkpoint_adapter() {
  local checkpoint=$1
  local destination=$2
  mkdir -p "$destination"
  local name
  for name in adapter_config.json adapter_model.safetensors chat_template.jinja README.md tokenizer_config.json tokenizer.json; do
    if [[ -f "$checkpoint/$name" ]]; then
      cp -f "$checkpoint/$name" "$destination/$name"
    fi
  done
  [[ -f "$destination/adapter_config.json" ]]
  [[ -f "$destination/adapter_model.safetensors" ]]
}

mkdir -p "$root/adapters" "$resolved"
python -m src.finalize_t6_1 materialize-adapter \
  --checkpoint-curve "$root/checkpoint-curve.json" \
  --output-dir "$root/adapters/rft_v2" > "$root/adapters/rft_v2-selection.json"

python -m src.finalize_t6_1 resolve-final-config \
  --curve-config "$resolved/curve.json" \
  --checkpoint-curve "$root/checkpoint-curve.json" \
  --output "$resolved/final.json"

# Stage 5B is allowed only after both preregistered termination checks pass.
if [[ ! -f data/external_cot_v2/manifest.json ]] || \
   ! jq -e '([.completion_checks[]] | all) and (.filtering.max_solution_words == 400) and (.counts.selected_rows == 15000)' data/external_cot_v2/manifest.json >/dev/null; then
  bash scripts/run_t6_1_external.sh
fi

python src/train_sft.py prepare \
  --config "$resolved/final.json" \
  --input data/rft_r1_v2/sft.jsonl \
  --input data/external_cot_v2/sft.jsonl \
  --cache "$root/rft_v2_external/cache" \
  --metadata "$root/rft_v2_external/cache-metadata.json"

run_b=true
if ! jq -e '.tokenization.assistant_eos_labeled_100_percent == true' "$root/rft_v2_external/cache-metadata.json" >/dev/null; then
  python - <<'PY'
import json
from pathlib import Path
Path("artifacts/t6_1_sft_v1r/b-skipped.json").write_text(
    json.dumps({"status": "skipped", "reason": "assistant EOS is not supervised on every mixed-data row"}, indent=2) + "\n"
)
PY
  run_b=false
fi

if [[ "$run_b" == true ]]; then
  if [[ ! -f "$root/rft_v2_external/training-metrics.json" ]] || \
     ! jq -e '.status == "complete"' "$root/rft_v2_external/training-metrics.json" >/dev/null; then
    python src/train_sft.py train \
      --config "$resolved/final.json" \
      --cache "$root/rft_v2_external/cache" \
      --cache-metadata "$root/rft_v2_external/cache-metadata.json" \
      --calibration "$root/calibration.json" \
      --output "$root/rft_v2_external/training-metrics.json" \
      --work-dir "$root/rft_v2_external/trainer" \
      --adapter-dir "$root/rft_v2_external/final-adapter" \
      --experiment rft_v2_external
  fi
  python -m src.finalize_t6_1 checkpoint-plan \
    --training-metrics "$root/rft_v2_external/training-metrics.json" \
    > "$root/rft_v2_external/selected-checkpoint.tsv"
  IFS=$'\t' read -r b_step b_epoch b_checkpoint < "$root/rft_v2_external/selected-checkpoint.tsv"
  materialize_checkpoint_adapter "$b_checkpoint" "$root/adapters/rft_v2_external"
fi

evaluate_arm() {
  local name=$1
  local adapter=$2
  mkdir -p "$root/$name/evaluation"
  python -u src/generate.py \
    --config configs/t4_output_contract.json \
    --input data/canonical/train.csv \
    --ids-file artifacts/t3_baseline/holdout_union_ids.txt \
    --output "$root/$name/evaluation/generations.jsonl" \
    --metadata "$root/$name/evaluation/run-metadata.json" \
    --engine vllm \
    --adapter "$adapter"
}

evaluate_arm rft_v2 "$root/adapters/rft_v2"
if [[ "$run_b" == true ]]; then
  evaluate_arm rft_v2_external "$root/adapters/rft_v2_external"
fi

finalize_args=(
  --root "$root" \
  --config "$base_config" \
  --labels data/canonical/train.csv \
  --union-ids artifacts/t3_baseline/holdout_union_ids.txt \
  --random-holdout data/splits/random_holdout.csv \
  --template-holdout data/splits/template_holdout.csv \
  --hard-diagnostic data/splits/hard_diagnostic.csv \
  --format-diagnostic data/splits/format_diagnostic.csv \
  --base-generations artifacts/t4_output_contract/generations.jsonl \
  --arm rft_v2 "$root/rft_v2/evaluation/generations.jsonl" "$root/curve/training-metrics.json" \
  --precision "$root/precision-probe.json" \
  --hp-sweep "$root/hp-sweep.json" \
  --checkpoint-curve "$root/checkpoint-curve.json" \
  --rft-manifest data/rft_r1_v2/manifest.json \
  --calibration "$root/calibration.json" \
  --manifest "$root/manifest.json" \
  --comparison "$root/comparison.md"
)
if [[ "$run_b" == true ]]; then
  finalize_args+=(
    --arm rft_v2_external "$root/rft_v2_external/evaluation/generations.jsonl" "$root/rft_v2_external/training-metrics.json"
  )
fi
python -m src.finalize_t6_1 finalize "${finalize_args[@]}"

tar -czf "$root/t6-1-final-metadata.tgz" \
  "$root/precision-probe.json" \
  "$root/hp-sweep.json" \
  "$root/checkpoint-curve.json" \
  "$root/comparison.md" \
  "$root/manifest.json" \
  data/rft_r1_v2/manifest.json \
  artifacts/t5_rft_targeted/manifest.json

echo '{"event":"t6_1_complete"}'
