#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${T12_PYTHON:-/venv/main/bin/python}"
config="configs/t12_cmu_orm.json"
artifact_root="artifacts/t12_cmu_orm"
data_root="data/cmu_orm"
fresh_root="$artifact_root/fresh-validation"
hardware_root="$artifact_root/hardware-smoke"
score_smoke_root="$artifact_root/scoring-smoke"
reused_root="$artifact_root/reused-t8"

mkdir -p "$artifact_root/logs" "$data_root" "$fresh_root" "$hardware_root"

export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="/workspace/.hf_home"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_ASYNC_ERROR_HANDLING=1

gpu0_uuid="$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
gpu1_uuid="$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"

json_complete() {
  local path="$1"
  [[ -f "$path" ]] || return 1
  "$python_bin" - "$path" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if value.get("status") == "complete" else 1)
PY
}

wait_pair() {
  local pid0="$1"
  local pid1="$2"
  local label="$3"
  local status0=0
  local status1=0
  set +e
  wait "$pid0"
  status0=$?
  wait "$pid1"
  status1=$?
  set -e
  if [[ "$status0" -ne 0 || "$status1" -ne 0 ]]; then
    echo "$label failed: rank0=$status0 rank1=$status1" >&2
    return 1
  fi
}

launch_model_smoke() {
  local physical_index="$1"
  local expected_uuid="$2"
  local output="$3"
  local log_path="$4"
  CUDA_VISIBLE_DEVICES="$physical_index" "$python_bin" -m src.orm_score model-smoke \
    --config "$config" \
    --physical-index "$physical_index" \
    --expected-uuid "$expected_uuid" \
    --output "$output" 2>&1 | tee -a "$log_path"
}

launch_generation_worker() {
  local logical_rank="$1"
  local physical_index="$2"
  local expected_uuid="$3"
  local manifest="$4"
  local generation_config="$5"
  local input_csv="$6"
  local ids_file="$7"
  local run_dir="$8"
  shift 8
  CUDA_VISIBLE_DEVICES="$physical_index" "$python_bin" -m src.t12_sharding generation-worker \
    --manifest "$manifest" \
    --logical-rank "$logical_rank" \
    --physical-index "$physical_index" \
    --expected-uuid "$expected_uuid" \
    --config "$generation_config" \
    --input "$input_csv" \
    --ids "$ids_file" \
    --output "$run_dir/shard-$logical_rank.jsonl" \
    --generation-metadata "$run_dir/generation-metadata-$logical_rank.json" \
    --worker-metadata "$run_dir/worker-metadata-$logical_rank.json" \
    "$@" 2>&1 | tee -a "$run_dir/worker-$logical_rank.log"
}

merge_generation_phase() {
  local manifest="$1"
  local run_dir="$2"
  local output="$3"
  local audit="$4"
  "$python_bin" -m src.t12_sharding merge \
    --manifest "$manifest" \
    --shard "$run_dir/shard-0.jsonl" \
    --shard "$run_dir/shard-1.jsonl" \
    --worker-metadata "$run_dir/worker-metadata-0.json" \
    --worker-metadata "$run_dir/worker-metadata-1.json" \
    --output "$output" \
    --audit "$audit"
}

run_generation_phase() {
  local manifest="$1"
  local generation_config="$2"
  local input_csv="$3"
  local shard_dir="$4"
  local run_dir="$5"
  local output="$6"
  local audit="$7"
  mkdir -p "$run_dir"
  if json_complete "$audit"; then
    return
  fi

  local complete0=0
  local complete1=0
  json_complete "$run_dir/worker-metadata-0.json" && complete0=1 || true
  json_complete "$run_dir/worker-metadata-1.json" && complete1=1 || true
  if [[ "$complete0" -eq 0 && "$complete1" -eq 0 ]]; then
    launch_generation_worker 0 0 "$gpu0_uuid" "$manifest" "$generation_config" \
      "$input_csv" "$shard_dir/shard-0-ids.txt" "$run_dir" &
    local pid0=$!
    launch_generation_worker 1 1 "$gpu1_uuid" "$manifest" "$generation_config" \
      "$input_csv" "$shard_dir/shard-1-ids.txt" "$run_dir" &
    local pid1=$!
    wait_pair "$pid0" "$pid1" "generation phase $run_dir"
  elif [[ "$complete0" -eq 0 ]]; then
    launch_generation_worker 0 0 "$gpu0_uuid" "$manifest" "$generation_config" \
      "$input_csv" "$shard_dir/shard-0-ids.txt" "$run_dir" &
    local pid0=$!
    launch_model_smoke 1 "$gpu1_uuid" "$run_dir/resume-readiness-gpu1.json" \
      "$run_dir/resume-readiness-gpu1.log" &
    local pid1=$!
    wait_pair "$pid0" "$pid1" "generation resume $run_dir"
  elif [[ "$complete1" -eq 0 ]]; then
    launch_model_smoke 0 "$gpu0_uuid" "$run_dir/resume-readiness-gpu0.json" \
      "$run_dir/resume-readiness-gpu0.log" &
    local pid0=$!
    launch_generation_worker 1 1 "$gpu1_uuid" "$manifest" "$generation_config" \
      "$input_csv" "$shard_dir/shard-1-ids.txt" "$run_dir" &
    local pid1=$!
    wait_pair "$pid0" "$pid1" "generation resume $run_dir"
  fi
  merge_generation_phase "$manifest" "$run_dir" "$output" "$audit"
}

launch_score_worker() {
  local logical_rank="$1"
  local physical_index="$2"
  local expected_uuid="$3"
  local manifest="$4"
  local candidates="$5"
  local questions="$6"
  local run_dir="$7"
  CUDA_VISIBLE_DEVICES="$physical_index" "$python_bin" -m src.orm_score worker \
    --config "$config" \
    --manifest "$manifest" \
    --logical-rank "$logical_rank" \
    --physical-index "$physical_index" \
    --expected-uuid "$expected_uuid" \
    --candidates "$candidates" \
    --questions "$questions" \
    --adapter "$artifact_root/adapter" \
    --output "$run_dir/shard-$logical_rank.jsonl" \
    --metadata "$run_dir/worker-metadata-$logical_rank.json" \
    2>&1 | tee -a "$run_dir/worker-$logical_rank.log"
}

merge_score_phase() {
  local manifest="$1"
  local run_dir="$2"
  local output="$3"
  local audit="$4"
  "$python_bin" -m src.t12_sharding merge \
    --manifest "$manifest" \
    --shard "$run_dir/shard-0.jsonl" \
    --shard "$run_dir/shard-1.jsonl" \
    --worker-metadata "$run_dir/worker-metadata-0.json" \
    --worker-metadata "$run_dir/worker-metadata-1.json" \
    --output "$output" \
    --audit "$audit"
}

run_score_phase() {
  local manifest="$1"
  local candidates="$2"
  local questions="$3"
  local run_dir="$4"
  local output="$5"
  local audit="$6"
  mkdir -p "$run_dir"
  if json_complete "$audit"; then
    return
  fi

  local complete0=0
  local complete1=0
  json_complete "$run_dir/worker-metadata-0.json" && complete0=1 || true
  json_complete "$run_dir/worker-metadata-1.json" && complete1=1 || true
  if [[ "$complete0" -eq 0 && "$complete1" -eq 0 ]]; then
    launch_score_worker 0 0 "$gpu0_uuid" "$manifest" "$candidates" "$questions" "$run_dir" &
    local pid0=$!
    launch_score_worker 1 1 "$gpu1_uuid" "$manifest" "$candidates" "$questions" "$run_dir" &
    local pid1=$!
    wait_pair "$pid0" "$pid1" "score phase $run_dir"
  elif [[ "$complete0" -eq 0 ]]; then
    launch_score_worker 0 0 "$gpu0_uuid" "$manifest" "$candidates" "$questions" "$run_dir" &
    local pid0=$!
    launch_model_smoke 1 "$gpu1_uuid" "$run_dir/resume-readiness-gpu1.json" \
      "$run_dir/resume-readiness-gpu1.log" &
    local pid1=$!
    wait_pair "$pid0" "$pid1" "score resume $run_dir"
  elif [[ "$complete1" -eq 0 ]]; then
    launch_model_smoke 0 "$gpu0_uuid" "$run_dir/resume-readiness-gpu0.json" \
      "$run_dir/resume-readiness-gpu0.log" &
    local pid0=$!
    launch_score_worker 1 1 "$gpu1_uuid" "$manifest" "$candidates" "$questions" "$run_dir" &
    local pid1=$!
    wait_pair "$pid0" "$pid1" "score resume $run_dir"
  fi
  merge_score_phase "$manifest" "$run_dir" "$output" "$audit"
}

echo "[T12] sealing top-level run marker and submission sentinel"
"$python_bin" -m src.t12_sharding run-marker \
  --config "$config" \
  --submission submission.csv \
  --output "$artifact_root/run-marker.json"

if [[ ! -f "$artifact_root/tests.xml" ]]; then
  echo "[T12] focused unit tests"
  "$python_bin" scripts/run_unittest_junit.py \
    --output "$artifact_root/tests.xml" \
    --suite-name "T12 CMU ORM focused tests" \
    --module-prefix t12_test \
    tests/test_t12_sharding.py tests/test_orm_data.py tests/test_orm_vote.py
fi

if [[ ! -f "$artifact_root/hardware-snapshot.json" ]]; then
  "$python_bin" -m src.t12_sharding hardware-snapshot \
    --output "$artifact_root/hardware-snapshot.json"
fi
if ! json_complete "$hardware_root/fixture-preparation.json"; then
  "$python_bin" -m src.build_orm_data prepare-hardware-smoke --config "$config"
fi

echo "[T12] two-GPU model-load, vLLM reproducibility/resume, and NCCL DDP smokes"
if ! json_complete "$hardware_root/model-smoke-gpu0.json" || \
   ! json_complete "$hardware_root/model-smoke-gpu1.json"; then
  launch_model_smoke 0 "$gpu0_uuid" "$hardware_root/model-smoke-gpu0.json" \
    "$hardware_root/model-smoke-gpu0.log" &
  pid0=$!
  launch_model_smoke 1 "$gpu1_uuid" "$hardware_root/model-smoke-gpu1.json" \
    "$hardware_root/model-smoke-gpu1.log" &
  pid1=$!
  wait_pair "$pid0" "$pid1" "two-GPU model load smoke"
fi

smoke_manifest="$hardware_root/generation-shard-manifest.json"
smoke_config="$hardware_root/generation-config.json"
smoke_questions="$hardware_root/questions.csv"
smoke_shards="$hardware_root/fixture-shards"
for repeat in a b; do
  repeat_dir="$hardware_root/repeat-$repeat"
  run_generation_phase "$smoke_manifest" "$smoke_config" "$smoke_questions" \
    "$smoke_shards" "$repeat_dir" "$repeat_dir/merged.jsonl" "$repeat_dir/merge-audit.json"
done

resume_dir="$hardware_root/forced-resume"
mkdir -p "$resume_dir"
if ! json_complete "$resume_dir/merge-audit.json"; then
  if [[ ! -f "$resume_dir/worker-metadata-0.json" ]]; then
    set +e
    launch_generation_worker 0 0 "$gpu0_uuid" "$smoke_manifest" "$smoke_config" \
      "$smoke_questions" "$smoke_shards/shard-0-ids.txt" "$resume_dir" \
      --force-fail-before-start &
    pid0=$!
    launch_generation_worker 1 1 "$gpu1_uuid" "$smoke_manifest" "$smoke_config" \
      "$smoke_questions" "$smoke_shards/shard-1-ids.txt" "$resume_dir" &
    pid1=$!
    wait "$pid0"
    status0=$?
    wait "$pid1"
    status1=$?
    set -e
    if [[ "$status0" -eq 0 || "$status1" -ne 0 ]]; then
      echo "Forced failure smoke did not produce the expected rank statuses" >&2
      exit 1
    fi
    set +e
    merge_generation_phase "$smoke_manifest" "$resume_dir" \
      "$resume_dir/should-not-exist.jsonl" "$resume_dir/should-not-exist-audit.json"
    failed_merge_status=$?
    set -e
    if [[ "$failed_merge_status" -eq 0 || -e "$resume_dir/should-not-exist.jsonl" ]]; then
      echo "Incomplete forced-failure shards were incorrectly merged" >&2
      exit 1
    fi
  fi
  if ! json_complete "$resume_dir/worker-metadata-0.json"; then
    launch_generation_worker 0 0 "$gpu0_uuid" "$smoke_manifest" "$smoke_config" \
      "$smoke_questions" "$smoke_shards/shard-0-ids.txt" "$resume_dir" &
    pid0=$!
    launch_model_smoke 1 "$gpu1_uuid" "$resume_dir/resume-readiness-gpu1.json" \
      "$resume_dir/resume-readiness-gpu1.log" &
    pid1=$!
    wait_pair "$pid0" "$pid1" "forced generation resume"
  fi
  merge_generation_phase "$smoke_manifest" "$resume_dir" \
    "$resume_dir/merged.jsonl" "$resume_dir/merge-audit.json"
fi

if ! json_complete "$hardware_root/generation-smoke.json"; then
  "$python_bin" -m src.t12_sharding generation-smoke-finalize \
    --merge-a "$hardware_root/repeat-a/merge-audit.json" \
    --merge-b "$hardware_root/repeat-b/merge-audit.json" \
    --merge-resumed "$resume_dir/merge-audit.json" \
    --failed-merge "$resume_dir/should-not-exist.jsonl" \
    --output "$hardware_root/generation-smoke.json"
fi

if ! json_complete "$hardware_root/ddp-smoke.json"; then
  CUDA_VISIBLE_DEVICES=0,1 "$python_bin" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node=2 -m src.train_orm \
    ddp-smoke --config "$config" --output "$hardware_root/ddp-smoke.json" \
    2>&1 | tee -a "$hardware_root/ddp-smoke.log"
fi

if ! json_complete "$artifact_root/hardware-preflight.json"; then
  "$python_bin" -m src.t12_sharding hardware-finalize \
    --snapshot "$artifact_root/hardware-snapshot.json" \
    --smoke "$hardware_root/model-smoke-gpu0.json" \
    --smoke "$hardware_root/model-smoke-gpu1.json" \
    --smoke "$hardware_root/generation-smoke.json" \
    --smoke "$hardware_root/ddp-smoke.json" \
    --output "$artifact_root/hardware-preflight.json"
fi

echo "[T12] freezing never-used validation before training or full generation"
"$python_bin" -m src.build_orm_data freeze-validation --config "$config"
if ! json_complete "$fresh_root/generation-preparation.json"; then
  "$python_bin" -m src.build_orm_data prepare-fresh-generation --config "$config"
fi

"$python_bin" -m src.t12_sharding run-marker \
  --config "$config" --output "$fresh_root/pipeline-marker.json"
run_generation_phase \
  "$fresh_root/generation-shard-manifest.json" \
  "configs/t8_self_consistency.json" \
  "$data_root/validation.csv" \
  "$fresh_root/generation-shards" \
  "$fresh_root/generation-shards" \
  "$fresh_root/generations.jsonl" \
  "$fresh_root/generation-merge-audit.json"

echo "[T12] generating ORM-train-only hard negatives and enforcing the data gate"
if ! json_complete "$data_root/candidate-generation-preparation.json"; then
  "$python_bin" -m src.build_orm_data prepare-candidate-generation --config "$config"
fi
train_candidate_root="$artifact_root/orm-train-candidates"
run_generation_phase \
  "$train_candidate_root/generation-shard-manifest.json" \
  "$train_candidate_root/generation-config.json" \
  "$data_root/candidate-generation-input.csv" \
  "$train_candidate_root/generation-shards" \
  "$train_candidate_root/generation-shards" \
  "$train_candidate_root/generations.jsonl" \
  "$train_candidate_root/merged-metadata.json"

if ! json_complete "$data_root/train-manifest.json"; then
  "$python_bin" -m src.build_orm_data finalize --config "$config"
fi

echo "[T12] tokenizing and training the frozen epoch-2 pointwise LoRA ORM with DDP"
if ! json_complete "$artifact_root/tokenized-dataset-metadata.json"; then
  "$python_bin" -m src.train_orm prepare --config "$config" \
    2>&1 | tee -a "$artifact_root/logs/tokenize.log"
fi
if ! json_complete "$artifact_root/train-metrics.json"; then
  CUDA_VISIBLE_DEVICES=0,1 "$python_bin" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node=2 -m src.train_orm \
    train --config "$config" 2>&1 | tee -a "$artifact_root/logs/train.log"
fi

adapter_sha="$($python_bin - <<'PY'
from pathlib import Path
from src.train_orm import sha256_tree
print(sha256_tree(Path("artifacts/t12_cmu_orm/adapter")))
PY
)"

echo "[T12] running the frozen 32-candidate distributed-vs-single scoring smoke"
if ! json_complete "$score_smoke_root/fixture-preparation.json"; then
  "$python_bin" -m src.build_orm_data prepare-scoring-smoke --config "$config"
fi
if [[ ! -f "$score_smoke_root/score-shard-manifest.json" ]]; then
  "$python_bin" -m src.t12_sharding create-score-manifest \
    --candidates "$score_smoke_root/candidates.jsonl" \
    --adapter-sha256 "$adapter_sha" \
    --config "$config" \
    --output "$score_smoke_root/score-shard-manifest.json"
fi
run_score_phase \
  "$score_smoke_root/score-shard-manifest.json" \
  "$score_smoke_root/candidates.jsonl" \
  "$score_smoke_root/questions.csv" \
  "$score_smoke_root/score-shards" \
  "$score_smoke_root/distributed-scores.jsonl" \
  "$score_smoke_root/score-merge-audit.json"
if [[ ! -f "$score_smoke_root/reference-scores.jsonl" ]]; then
  CUDA_VISIBLE_DEVICES=0 "$python_bin" -m src.orm_score reference \
    --config "$config" \
    --candidates "$score_smoke_root/candidates.jsonl" \
    --questions "$score_smoke_root/questions.csv" \
    --adapter "$artifact_root/adapter" \
    --output "$score_smoke_root/reference-scores.jsonl" \
    --batch-size 4 2>&1 | tee -a "$score_smoke_root/reference.log"
fi
if ! json_complete "$score_smoke_root/score-compare.json"; then
  "$python_bin" -m src.orm_score compare \
    --distributed "$score_smoke_root/distributed-scores.jsonl" \
    --reference "$score_smoke_root/reference-scores.jsonl" \
    --tolerance 0.0001 \
    --output "$score_smoke_root/score-compare.json"
fi
if [[ ! -f "$score_smoke_root/distributed-vote/label-blind-freeze.json" ]]; then
  "$python_bin" -m src.orm_vote freeze \
    --config "$config" \
    --questions "$score_smoke_root/questions.csv" \
    --generations "$score_smoke_root/candidates.jsonl" \
    --scores "$score_smoke_root/distributed-scores.jsonl" \
    --output-dir "$score_smoke_root/distributed-vote" --k 32
fi
if [[ ! -f "$score_smoke_root/reference-vote/label-blind-freeze.json" ]]; then
  "$python_bin" -m src.orm_vote freeze \
    --config "$config" \
    --questions "$score_smoke_root/questions.csv" \
    --generations "$score_smoke_root/candidates.jsonl" \
    --scores "$score_smoke_root/reference-scores.jsonl" \
    --output-dir "$score_smoke_root/reference-vote" --k 32
fi

echo "[T12] scoring all 32,000 fresh candidates on two independent replicas"
if [[ ! -f "$fresh_root/score-shard-manifest.json" ]]; then
  "$python_bin" -m src.t12_sharding create-score-manifest \
    --candidates "$fresh_root/generations.jsonl" \
    --adapter-sha256 "$adapter_sha" \
    --config "$config" \
    --output "$fresh_root/score-shard-manifest.json"
fi
run_score_phase \
  "$fresh_root/score-shard-manifest.json" \
  "$fresh_root/generations.jsonl" \
  "$data_root/validation.csv" \
  "$fresh_root/score-shards" \
  "$fresh_root/candidate-scores.jsonl" \
  "$fresh_root/score-merge-audit.json"

"$python_bin" -m src.t12_sharding run-marker \
  --config "$config" --output "$fresh_root/aggregation-marker.json"
if [[ ! -f "$fresh_root/label-blind-freeze.json" ]]; then
  "$python_bin" -m src.orm_vote freeze \
    --config "$config" \
    --questions "$data_root/validation.csv" \
    --generations "$fresh_root/generations.jsonl" \
    --scores "$fresh_root/candidate-scores.jsonl" \
    --output-dir "$fresh_root" --k 32
fi
if ! json_complete "$fresh_root/runtime.json"; then
  "$python_bin" -m src.t12_sharding runtime-finalize \
    --pipeline-marker "$fresh_root/pipeline-marker.json" \
    --aggregation-marker "$fresh_root/aggregation-marker.json" \
    --generation-audit "$fresh_root/generation-merge-audit.json" \
    --score-audit "$fresh_root/score-merge-audit.json" \
    --freeze "$fresh_root/label-blind-freeze.json" \
    --output "$fresh_root/runtime.json"
fi

if [[ ! -f "$fresh_root/evaluation.json" ]]; then
  echo "[T12] joining gold only after the label-blind freeze and applying the preregistered gate"
  "$python_bin" -m src.orm_vote evaluate \
    --config "$config" \
    --freeze "$fresh_root/label-blind-freeze.json" \
    --labels data/canonical/train.csv \
    --runtime "$fresh_root/runtime.json" \
    --output-json "$fresh_root/evaluation.json" \
    --output-markdown "$fresh_root/evaluation.md"
fi

if ! json_complete "$artifact_root/integration-smoke.json"; then
  "$python_bin" -m src.t12_sharding integration-smoke-finalize \
    --generation-smoke "$hardware_root/generation-smoke.json" \
    --ddp-smoke "$hardware_root/ddp-smoke.json" \
    --model-smoke "$hardware_root/model-smoke-gpu0.json" \
    --model-smoke "$hardware_root/model-smoke-gpu1.json" \
    --score-compare "$score_smoke_root/score-compare.json" \
    --distributed-group "$score_smoke_root/distributed-vote/group-weights.jsonl" \
    --reference-group "$score_smoke_root/reference-vote/group-weights.jsonl" \
    --distributed-prediction "$score_smoke_root/distributed-vote/predictions.jsonl" \
    --reference-prediction "$score_smoke_root/reference-vote/predictions.jsonl" \
    --output "$artifact_root/integration-smoke.json"
fi

echo "[T12] fresh decision is frozen; starting reused-T8 diagnostic-only replay"
if ! json_complete "$data_root/reused-t8-preparation.json"; then
  "$python_bin" -m src.build_orm_data prepare-reused-t8-diagnostic --config "$config"
fi
mkdir -p "$reused_root"
if [[ ! -f "$reused_root/score-shard-manifest.json" ]]; then
  "$python_bin" -m src.t12_sharding create-score-manifest \
    --candidates artifacts/t8_self_consistency/generations.jsonl \
    --adapter-sha256 "$adapter_sha" \
    --config "$config" \
    --output "$reused_root/score-shard-manifest.json"
fi
run_score_phase \
  "$reused_root/score-shard-manifest.json" \
  artifacts/t8_self_consistency/generations.jsonl \
  "$data_root/reused-t8-questions.csv" \
  "$reused_root/score-shards" \
  "$reused_root/candidate-scores.jsonl" \
  "$reused_root/score-merge-audit.json"
if [[ ! -f "$reused_root/label-blind-freeze.json" ]]; then
  "$python_bin" -m src.orm_vote freeze \
    --config "$config" \
    --questions "$data_root/reused-t8-questions.csv" \
    --generations artifacts/t8_self_consistency/generations.jsonl \
    --scores "$reused_root/candidate-scores.jsonl" \
    --output-dir "$reused_root" --k 32
fi
if [[ ! -f "$artifact_root/reused-t8-diagnostic.json" ]]; then
  "$python_bin" -m src.orm_vote diagnostic \
    --freeze "$reused_root/label-blind-freeze.json" \
    --labels data/canonical/train.csv \
    --output "$artifact_root/reused-t8-diagnostic.json"
fi

if ! json_complete "$artifact_root/manifest.json"; then
  "$python_bin" -m src.t12_sharding finalize-run \
    --config "$config" \
    --run-marker "$artifact_root/run-marker.json" \
    --distributed-output "$artifact_root/distributed-run-manifest.json" \
    --output "$artifact_root/manifest.json"
fi

echo "[T12] complete"
"$python_bin" - "$fresh_root/evaluation.json" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(json.dumps({
    "decision": value["decision"],
    "accuracies": value["accuracies"],
    "delta_vs_stronger_baseline": value["delta_vs_stronger_baseline"],
    "gate_checks": value["gate_checks"],
}, ensure_ascii=False, sort_keys=True))
PY
