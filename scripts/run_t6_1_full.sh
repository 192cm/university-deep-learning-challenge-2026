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
mkdir -p "$root" "$resolved" artifacts/t5_rft_targeted data/rft_r1_v2

python -m pytest -q \
  tests/test_extract.py \
  tests/test_evaluate.py \
  tests/test_generate.py \
  tests/test_train_sft.py \
  tests/test_build_rft_v2.py \
  tests/test_finalize_t6_1.py \
  --junitxml="$root/tests.xml"

# Stage 0: the HF NF4 generations are resume-safe and usually already complete
# from the short bootstrap service run.
python -u src/generate.py \
  --config configs/t4_output_contract.json \
  --input data/canonical/train.csv \
  --ids-file data/splits/random_holdout_ids.txt \
  --output "$root/precision-probe/hf_nf4/generations.jsonl" \
  --metadata "$root/precision-probe/hf_nf4/run-metadata.json" \
  --engine hf \
  --adapter artifacts/t6_sft_v1/adapters/rft_r1 \
  --hf-load-in-4bit

python -m src.finalize_t6_1 precision \
  --base-config "$base_config" \
  --labels data/canonical/train.csv \
  --ids data/splits/random_holdout_ids.txt \
  --nf4-generations "$root/precision-probe/hf_nf4/generations.jsonl" \
  --bf16-generations artifacts/t6_sft_v1/rft_r1/evaluation/generations.jsonl \
  --output "$root/precision-probe.json" \
  --resolved-dir "$resolved"

# Stage 1: deterministic validation/target scopes.
python -m src.build_rft_v2 prepare \
  --canonical data/canonical/train.csv \
  --rft-audit data/rft_r1/audit.csv \
  --rft-ids data/rft_pool_ids.txt \
  --holdout-ids data/splits/random_holdout_ids.txt \
  --holdout-ids data/splits/template_holdout_ids.txt \
  --holdout-ids data/splits/hard_diagnostic_ids.txt \
  --holdout-ids data/splits/format_diagnostic_ids.txt \
  --validation-csv data/splits/rft_validation_500.csv \
  --validation-ids data/splits/rft_validation_500_ids.txt \
  --validation-audit data/splits/rft_validation_500_audit.csv \
  --targeted-ids artifacts/t5_rft_targeted/target_ids.txt \
  --output "$root/preparation.json" \
  --seed 42

# Stage 2: extra k=48 only for original-c=1..7 questions.
python -u src/generate.py \
  --config configs/t5_rft_targeted.json \
  --input data/canonical/train.csv \
  --ids-file artifacts/t5_rft_targeted/target_ids.txt \
  --output artifacts/t5_rft_targeted/generations.jsonl \
  --metadata artifacts/t5_rft_targeted/run-metadata.json \
  --engine vllm

# Stage 3: original c remains defined by T5's first 16 samples.  A failed data
# gate is an expected experimental stop, not a supervisor crash: preserve the
# failed manifest and exit cleanly so training cannot start or restart-loop.
set +e
python -m src.build_rft_v2 build \
  --canonical data/canonical/train.csv \
  --rft-audit data/rft_r1/audit.csv \
  --rft-ids data/rft_pool_ids.txt \
  --validation-ids data/splits/rft_validation_500_ids.txt \
  --original-generations artifacts/t5_rft_r1/generations.jsonl \
  --targeted-generations artifacts/t5_rft_targeted/generations.jsonl \
  --targeted-metadata artifacts/t5_rft_targeted/run-metadata.json \
  --targeted-ids artifacts/t5_rft_targeted/target_ids.txt \
  --config "$base_config" \
  --output-dir data/rft_r1_v2 \
  --targeted-artifact-dir artifacts/t5_rft_targeted \
  --seed 42
build_status=$?
set -e
if [[ $build_status -ne 0 ]]; then
  echo '{"event":"rft_v2_gate_failed","action":"stopped_before_training"}'
  exit 0
fi

calibration_config="$resolved/lr_3em05.json"
python src/train_sft.py prepare \
  --config "$calibration_config" \
  --input data/rft_r1_v2/sft.jsonl \
  --cache "$root/rft_v2/cache" \
  --metadata "$root/rft_v2/cache-metadata.json"

if [[ ! -f "$root/calibration.json" ]] || ! jq -e '.status == "complete"' "$root/calibration.json" >/dev/null; then
  python src/train_sft.py calibrate \
    --config "$calibration_config" \
    --cache "$root/rft_v2/cache" \
    --cache-metadata "$root/rft_v2/cache-metadata.json" \
    --output "$root/calibration.json" \
    --work-dir "$root/calibration-probes" \
    --environment environment.json
fi

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

evaluate_checkpoints() {
  local name=$1
  local learning_rate=$2
  local training_metrics=$3
  local validation_dir=$4
  local score_output=$5
  mkdir -p "$validation_dir"
  local plan="$validation_dir/checkpoint-plan.tsv"
  python -m src.finalize_t6_1 checkpoint-plan \
    --training-metrics "$training_metrics" > "$plan"
  local step epoch checkpoint output_dir adapter
  while IFS=$'\t' read -r step epoch checkpoint; do
    output_dir="$validation_dir/checkpoint-$step"
    adapter="$output_dir/adapter"
    materialize_checkpoint_adapter "$checkpoint" "$adapter"
    python -u src/generate.py \
      --config configs/t4_output_contract.json \
      --input data/canonical/train.csv \
      --ids-file data/splits/rft_validation_500_ids.txt \
      --output "$output_dir/generations.jsonl" \
      --metadata "$output_dir/run-metadata.json" \
      --engine vllm \
      --adapter "$adapter"
  done < "$plan"
  python -m src.finalize_t6_1 score-checkpoints \
    --name "$name" \
    --learning-rate "$learning_rate" \
    --training-metrics "$training_metrics" \
    --validation-dir "$validation_dir" \
    --labels data/splits/rft_validation_500.csv \
    --ids data/splits/rft_validation_500_ids.txt \
    --output "$score_output"
}

run_hp_arm() {
  local name=$1
  local learning_rate=$2
  local config="$resolved/$name.json"
  local arm_root="$root/hp/$name"
  mkdir -p "$arm_root"
  python src/train_sft.py train \
    --config "$config" \
    --cache "$root/rft_v2/cache" \
    --cache-metadata "$root/rft_v2/cache-metadata.json" \
    --calibration "$root/calibration.json" \
    --output "$arm_root/training-metrics.json" \
    --work-dir "$arm_root/trainer" \
    --adapter-dir "$arm_root/final-adapter" \
    --experiment "$name"
  evaluate_checkpoints \
    "$name" "$learning_rate" "$arm_root/training-metrics.json" \
    "$arm_root/validation" "$arm_root/checkpoint-scores.json"
}

# Stage 4: exactly three one-epoch LR arms, selected only on validation-500.
run_hp_arm lr_1em05 0.00001
run_hp_arm lr_3em05 0.00003
run_hp_arm lr_1em04 0.0001

python -m src.finalize_t6_1 select-hp \
  --score "$root/hp/lr_1em05/checkpoint-scores.json" \
  --score "$root/hp/lr_3em05/checkpoint-scores.json" \
  --score "$root/hp/lr_1em04/checkpoint-scores.json" \
  --output "$root/hp-sweep.json"

# Extend only the selected LR to the preregistered 2-epoch curve.
python -m src.finalize_t6_1 resolve-curve-config \
  --base-config "$base_config" \
  --precision "$root/precision-probe.json" \
  --hp-sweep "$root/hp-sweep.json" \
  --output "$resolved/curve.json"

python src/train_sft.py train \
  --config "$resolved/curve.json" \
  --cache "$root/rft_v2/cache" \
  --cache-metadata "$root/rft_v2/cache-metadata.json" \
  --calibration "$root/calibration.json" \
  --output "$root/curve/training-metrics.json" \
  --work-dir "$root/curve/trainer" \
  --adapter-dir "$root/curve/final-adapter" \
  --experiment curve

selected_lr=$(jq -r '.selected.learning_rate' "$root/hp-sweep.json")
evaluate_checkpoints \
  curve "$selected_lr" "$root/curve/training-metrics.json" \
  "$root/curve/validation" "$root/checkpoint-curve.json"

bash scripts/run_t6_1_after_curve.sh
