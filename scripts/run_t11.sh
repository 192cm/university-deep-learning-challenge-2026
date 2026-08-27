#!/usr/bin/env bash
set -euo pipefail

cd /workspace
source /venv/main/bin/activate

export HF_HOME=/workspace/.hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_BATCH_INVARIANT=1

config=configs/t11_aimo_generation_quality.json
data_root=data/t11_aimo_generation_quality
artifact_root=artifacts/t11_aimo_generation_quality
submission_root=artifacts/submissions/t11_c1_filtered_k32
resolved_root="$artifact_root/resolved-configs"
validation_root="$artifact_root/validation"
mkdir -p "$data_root" "$artifact_root/holdout" "$artifact_root/adapters/sft" \
  "$artifact_root/adapters/dpo" "$resolved_root" "$validation_root"

run_tests() {
  python -m pytest -q \
    tests/test_extract.py \
    tests/test_evaluate.py \
    tests/test_generate.py \
    tests/test_train_sft.py \
    tests/test_vote_filter.py \
    tests/test_t11.py \
    --junitxml="$artifact_root/tests.xml"
}

preflight_data() {
  python -m src.build_t11_hard_cot preflight-data --config "$config"
}

probe() {
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  python -u src/generate.py \
    --config "$config" \
    --input data/canonical/train.csv \
    --ids-file "$data_root/eligible_ids.txt" \
    --output "$data_root/student_probe.jsonl" \
    --metadata "$artifact_root/student-probe-run-metadata.json" \
    --engine vllm \
    --prompt-mode cot_boxed \
    --n 8 \
    --seed 42000
  python -m src.build_t11_hard_cot analyze-probe \
    --config "$config" \
    --generations "$data_root/student_probe.jsonl"
}

teacher_preflight() {
  export HF_HUB_OFFLINE=0
  export TRANSFORMERS_OFFLINE=0
  python -u -m src.build_t11_hard_cot teacher-generate \
    --config "$config" \
    --ids "$data_root/teacher_preflight_ids.txt" \
    --output "$data_root/teacher_generations.jsonl" \
    --metadata "$artifact_root/teacher-run-metadata.json" \
    --sample-start 0 \
    --sample-count 4 \
    --scope teacher_preflight
  set +e
  python -m src.build_t11_hard_cot teacher-gate \
    --config "$config" \
    --generations "$data_root/teacher_generations.jsonl" \
    --metadata "$artifact_root/teacher-run-metadata.json" \
    --output "$artifact_root/teacher-preflight.json"
  local gate_status=$?
  set -e
  return "$gate_status"
}

build_data() {
  export HF_HUB_OFFLINE=0
  export TRANSFORMERS_OFFLINE=0
  python -u -m src.build_t11_hard_cot teacher-generate \
    --config "$config" \
    --ids "$data_root/hard_ids.txt" \
    --output "$data_root/teacher_generations.jsonl" \
    --metadata "$artifact_root/teacher-run-metadata.json" \
    --sample-start 0 \
    --sample-count 4 \
    --scope teacher_full_first_round
  python -m src.build_t11_hard_cot prepare-second-round \
    --config "$config" \
    --generations "$data_root/teacher_generations.jsonl" \
    --output "$data_root/teacher_second_round_ids.txt"
  if [[ -s "$data_root/teacher_second_round_ids.txt" ]]; then
    python -u -m src.build_t11_hard_cot teacher-generate \
      --config "$config" \
      --ids "$data_root/teacher_second_round_ids.txt" \
      --output "$data_root/teacher_generations.jsonl" \
      --metadata "$artifact_root/teacher-run-metadata.json" \
      --sample-start 4 \
      --sample-count 4 \
      --scope teacher_full_second_round
  fi
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  python -m src.build_t11_hard_cot build-data \
    --config "$config" \
    --teacher-generations "$data_root/teacher_generations.jsonl" \
    --student-generations "$data_root/student_probe.jsonl" \
    --second-round-ids "$data_root/teacher_second_round_ids.txt"
}

materialize_adapter() {
  local source_dir=$1
  local destination=$2
  mkdir -p "$destination"
  local filename
  for filename in adapter_config.json adapter_model.safetensors chat_template.jinja \
    README.md tokenizer_config.json tokenizer.json special_tokens_map.json; do
    if [[ -f "$source_dir/$filename" ]]; then
      cp -f "$source_dir/$filename" "$destination/$filename"
    fi
  done
  [[ -f "$destination/adapter_config.json" ]]
  [[ -f "$destination/adapter_model.safetensors" ]]
}

score_adapter_checkpoint() {
  local stage=$1
  local learning_rate=$2
  local target_epoch=$3
  local actual_epoch=$4
  local checkpoint=$5
  local name=$6
  local destination="$validation_root/$stage/$name"
  local adapter="$destination/adapter"
  mkdir -p "$destination"
  materialize_adapter "$checkpoint" "$adapter"
  python -u src/generate.py \
    --config "$config" \
    --input data/canonical/train.csv \
    --ids-file data/splits/rft_validation_500_ids.txt \
    --output "$destination/generations.jsonl" \
    --metadata "$destination/run-metadata.json" \
    --engine vllm \
    --prompt-mode cot_boxed \
    --adapter "$adapter" \
    --n 8 \
    --seed 52000
  python -m src.finalize_t11 score-validation \
    --config "$config" \
    --name "$name" \
    --stage "$stage" \
    --learning-rate "$learning_rate" \
    --target-epoch "$target_epoch" \
    --actual-epoch "$actual_epoch" \
    --adapter "$adapter" \
    --generations "$destination/generations.jsonl" \
    --metadata "$destination/run-metadata.json" \
    --output "$destination/score.json"
}

score_base_validation() {
  local destination="$validation_root/base"
  mkdir -p "$destination"
  python -u src/generate.py \
    --config "$config" \
    --input data/canonical/train.csv \
    --ids-file data/splits/rft_validation_500_ids.txt \
    --output "$destination/generations.jsonl" \
    --metadata "$destination/run-metadata.json" \
    --engine vllm \
    --prompt-mode cot_boxed \
    --n 8 \
    --seed 52000
  python -m src.finalize_t11 score-validation \
    --config "$config" \
    --name base \
    --stage base \
    --target-epoch 0 \
    --actual-epoch 0 \
    --generations "$destination/generations.jsonl" \
    --metadata "$destination/run-metadata.json" \
    --output "$destination/score.json"
}

run_sft_arm() {
  local name=$1
  local learning_rate=$2
  local resolved="$resolved_root/$name.json"
  local arm_root="$artifact_root/adapters/sft/$name"
  python -m src.build_t11_hard_cot resolve-sft-config \
    --config "$config" \
    --output "$resolved" \
    --learning-rate "$learning_rate"
  python src/train_sft.py train \
    --config "$resolved" \
    --cache "$artifact_root/sft-cache" \
    --cache-metadata "$artifact_root/sft-cache-metadata.json" \
    --calibration "$artifact_root/sft-calibration.json" \
    --output "$arm_root/training-metrics.json" \
    --work-dir "$arm_root/trainer" \
    --adapter-dir "$arm_root/final-adapter" \
    --experiment "$name"
  while IFS=$'\t' read -r target_epoch actual_epoch step checkpoint; do
    local tag
    tag=$(python -c 'import sys; print(f"e{float(sys.argv[1]):.2f}".replace(".", "p"))' "$target_epoch")
    score_adapter_checkpoint sft "$learning_rate" "$target_epoch" "$actual_epoch" \
      "$checkpoint" "${name}_${tag}_step${step}"
  done < <(python -m src.build_t11_hard_cot checkpoint-plan \
    --training-metrics "$arm_root/training-metrics.json")
}

train_sft_stage() {
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  python -m src.build_t11_hard_cot resolve-sft-config \
    --config "$config" \
    --output "$resolved_root/lr_3em05.json" \
    --learning-rate 0.00003
  python src/train_sft.py prepare \
    --config "$resolved_root/lr_3em05.json" \
    --input "$data_root/sft_train.jsonl" \
    --cache "$artifact_root/sft-cache" \
    --metadata "$artifact_root/sft-cache-metadata.json"
  if [[ ! -f "$artifact_root/sft-calibration.json" ]] || \
    ! jq -e '.status == "complete"' "$artifact_root/sft-calibration.json" >/dev/null; then
    python src/train_sft.py calibrate \
      --config "$resolved_root/lr_3em05.json" \
      --cache "$artifact_root/sft-cache" \
      --cache-metadata "$artifact_root/sft-cache-metadata.json" \
      --output "$artifact_root/sft-calibration.json" \
      --work-dir "$artifact_root/sft-calibration-probes"
  fi
  score_base_validation
  run_sft_arm lr_1em05 0.00001
  run_sft_arm lr_3em05 0.00003
  run_sft_arm lr_1em04 0.0001
  local score_args=()
  local score
  while IFS= read -r score; do
    score_args+=(--score "$score")
  done < <(find "$validation_root/sft" -name score.json -type f | sort)
  python -m src.finalize_t11 summarize-scores \
    --config "$config" \
    "${score_args[@]}" \
    --stage sft \
    --output "$artifact_root/sft-hp-sweep.json" \
    --selection "$artifact_root/sft-selection.json"
}

train_dpo_stage() {
  if ! jq -e '.dpo_gate_passed == true' "$data_root/data-gates.json" >/dev/null; then
    python -m src.finalize_t11 record-dpo-skip \
      --config "$config" \
      --data-gates "$data_root/data-gates.json" \
      --output "$artifact_root/dpo-checkpoint-curve.json"
    return 0
  fi
  local sft_adapter
  sft_adapter=$(jq -r '.selected.adapter.path' "$artifact_root/sft-selection.json")
  python -u -m src.train_dpo train \
    --config "$config" \
    --input "$data_root/dpo_train.jsonl" \
    --sft-adapter "$sft_adapter" \
    --output "$artifact_root/adapters/dpo/training-metrics.json" \
    --work-dir "$artifact_root/adapters/dpo/trainer" \
    --adapter-dir "$artifact_root/adapters/dpo/final-adapter"
  while IFS=$'\t' read -r target_epoch actual_epoch step checkpoint; do
    local tag
    tag=$(python -c 'import sys; print(f"e{float(sys.argv[1]):.2f}".replace(".", "p"))' "$target_epoch")
    score_adapter_checkpoint dpo 0.000001 "$target_epoch" "$actual_epoch" \
      "$checkpoint" "dpo_${tag}_step${step}"
  done < <(python -m src.build_t11_hard_cot checkpoint-plan \
    --training-metrics "$artifact_root/adapters/dpo/training-metrics.json")
  local score_args=()
  local score
  while IFS= read -r score; do
    score_args+=(--score "$score")
  done < <(find "$validation_root/dpo" -name score.json -type f | sort)
  python -m src.finalize_t11 summarize-scores \
    --config "$config" \
    "${score_args[@]}" \
    --stage dpo \
    --output "$artifact_root/dpo-checkpoint-curve.json"
}

freeze_candidate_stage() {
  local sft_args=()
  local dpo_args=()
  local score
  while IFS= read -r score; do
    sft_args+=(--sft-score "$score")
  done < <(find "$validation_root/sft" -name score.json -type f | sort)
  if [[ -d "$validation_root/dpo" ]]; then
    while IFS= read -r score; do
      dpo_args+=(--dpo-score "$score")
    done < <(find "$validation_root/dpo" -name score.json -type f | sort)
  fi
  python -m src.finalize_t11 freeze-candidate \
    --config "$config" \
    --base-score "$validation_root/base/score.json" \
    "${sft_args[@]}" \
    "${dpo_args[@]}" \
    --output-dir "$artifact_root"
}

evaluate_holdout_stage() {
  if ! jq -e '.status == "frozen_for_holdout"' "$artifact_root/final_config.json" >/dev/null; then
    return 20
  fi
  local adapter
  adapter=$(jq -r '.adapter.path' "$artifact_root/final_config.json")
  python -u src/generate.py \
    --config "$config" \
    --input data/canonical/train.csv \
    --ids-file artifacts/t8_self_consistency/holdout_union_ids.txt \
    --output "$artifact_root/holdout/generations.jsonl" \
    --metadata "$artifact_root/holdout/run-metadata.json" \
    --engine vllm \
    --prompt-mode cot_boxed \
    --adapter "$adapter"
  python -m src.finalize_t11 evaluate-holdout \
    --config "$config" \
    --candidate-generations "$artifact_root/holdout/generations.jsonl" \
    --candidate-metadata "$artifact_root/holdout/run-metadata.json" \
    --output-dir "$artifact_root" \
    --tests-xml "$artifact_root/tests.xml"
}

build_leaderboard_submission() {
  if ! jq -e '.decision == "adopt"' "$artifact_root/manifest.json" >/dev/null; then
    return 20
  fi
  local adapter
  adapter=$(jq -r '.adapter.path' "$artifact_root/final_config.json")
  mkdir -p "$submission_root"
  python -u src/generate.py \
    --config "$config" \
    --input data/deep_chal_math_leaderboard_filtered.csv \
    --output "$submission_root/generations.jsonl" \
    --metadata "$submission_root/run-metadata.json" \
    --engine vllm \
    --prompt-mode cot_boxed \
    --adapter "$adapter"
  python -m src.finalize_t11 build-submission \
    --config "$config" \
    --generations "$submission_root/generations.jsonl" \
    --metadata "$submission_root/run-metadata.json" \
    --output-dir "$submission_root"
  cp -f "$submission_root/submission.csv" submission.csv
}

run_all() {
  run_tests
  preflight_data
  probe
  if ! teacher_preflight; then
    python -m src.finalize_t11 record-early-stop \
      --config "$config" \
      --status teacher_gate_failed \
      --output-dir "$artifact_root" \
      --tests-xml "$artifact_root/tests.xml"
    echo '{"event":"t11_stopped","status":"teacher_gate_failed"}'
    return 0
  fi
  build_data
  train_sft_stage
  train_dpo_stage
  freeze_candidate_stage
  if ! evaluate_holdout_stage; then
    python -m src.finalize_t11 record-early-stop \
      --config "$config" \
      --status validation_reject \
      --output-dir "$artifact_root" \
      --tests-xml "$artifact_root/tests.xml"
    echo '{"event":"t11_stopped","status":"validation_reject"}'
    return 0
  fi
  if jq -e '.decision == "adopt"' "$artifact_root/manifest.json" >/dev/null; then
    build_leaderboard_submission
  fi
  echo '{"event":"t11_complete"}'
}

command=${1:-all}
case "$command" in
  all) run_all ;;
  tests) run_tests ;;
  preflight-data) preflight_data ;;
  probe) probe ;;
  teacher-preflight) teacher_preflight ;;
  build-data) build_data ;;
  train-sft) train_sft_stage ;;
  train-dpo) train_dpo_stage ;;
  freeze-candidate) freeze_candidate_stage ;;
  evaluate-holdout) evaluate_holdout_stage ;;
  build-leaderboard-submission) build_leaderboard_submission ;;
  *)
    echo "unknown T11 stage: $command" >&2
    exit 2
    ;;
esac
