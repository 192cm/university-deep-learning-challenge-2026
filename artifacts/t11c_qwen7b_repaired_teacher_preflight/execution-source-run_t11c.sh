#!/usr/bin/env bash
set -euo pipefail

workspace=${T11C_WORKSPACE:-/workspace}
cd "$workspace"

t11c_venv=${T11C_VENV:-/workspace/.venvs/t11b}
if [[ -f "$t11c_venv/bin/activate" ]]; then
  source "$t11c_venv/bin/activate"
else
  source /venv/main/bin/activate
fi

export HF_HOME=/workspace/.hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_BATCH_INVARIANT=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export HF_HUB_DISABLE_XET=1

config=configs/t11c_qwen7b_repaired_teacher_preflight.json
data_root=data/t11c_qwen7b_repaired_teacher_preflight
artifact_root=artifacts/t11c_qwen7b_repaired_teacher_preflight

mkdir -p "$data_root" "$artifact_root"

freeze_execution_sources() {
  local source_path target_path
  while IFS='|' read -r source_path target_path; do
    if [[ -e "$target_path" ]]; then
      cmp -s "$source_path" "$target_path" || {
        echo "frozen execution source differs: $source_path" >&2
        return 5
      }
    else
      cp -p "$source_path" "$target_path"
    fi
  done <<EOF
src/run_t11c_qwen_teacher.py|$artifact_root/execution-source-run_t11c_qwen_teacher.py
src/extract.py|$artifact_root/execution-source-extract.py
src/build_t11_hard_cot.py|$artifact_root/execution-source-build_t11_hard_cot.py
$config|$artifact_root/execution-source-config.json
scripts/run_t11c.sh|$artifact_root/execution-source-run_t11c.sh
scripts/supervisor_t11c.sh|$artifact_root/execution-source-supervisor_t11c.sh
configs/supervisor_t11c.conf|$artifact_root/execution-source-supervisor_t11c.conf
tests/test_t11c_qwen_teacher.py|$artifact_root/execution-source-tests.py
EOF
}

run_tests() {
  freeze_execution_sources
  python -m pytest -q \
    tests/test_extract.py \
    tests/test_t11.py \
    tests/test_normalize_teacher_trace.py \
    tests/test_t11c_qwen_teacher.py \
    --junitxml="$artifact_root/tests.xml"
}

run_command() {
  local command=$1
  python -u -m src.run_t11c_qwen_teacher "$command" --config "$config"
}

verify_inputs() {
  freeze_execution_sources
  run_command verify-inputs
}

smoke() {
  run_command smoke
}

prepare_manifests() {
  run_command prepare-manifests
}

first_round_generate() {
  run_command first-round-generate
}

first_round_normalize_and_select() {
  run_command first-round-normalize-and-select
}

second_round_generate() {
  run_command second-round-generate
}

finalize() {
  run_command finalize
}

terminal_failure() {
  local status=$1
  python -m src.run_t11c_qwen_teacher terminal-failure \
    --config "$config" \
    --status "$status"
}

run_all() {
  run_tests
  if ! verify_inputs; then
    terminal_failure input_identity_failed
    return 2
  fi
  if ! smoke; then
    local smoke_status
    smoke_status=$(python -c 'import json; print(json.load(open("artifacts/t11c_qwen7b_repaired_teacher_preflight/load-and-seed-smoke.json"))["status"])')
    terminal_failure "$smoke_status"
    return 3
  fi
  if ! prepare_manifests; then
    terminal_failure input_too_long
    return 4
  fi
  first_round_generate
  first_round_normalize_and_select
  second_round_generate
  finalize
}

case "${1:-all}" in
  tests) run_tests ;;
  verify-inputs) verify_inputs ;;
  smoke) smoke ;;
  prepare-manifests) prepare_manifests ;;
  first-round-generate) first_round_generate ;;
  first-round-normalize-and-select) first_round_normalize_and_select ;;
  second-round-generate) second_round_generate ;;
  finalize) finalize ;;
  all) run_all ;;
  *)
    echo "usage: $0 {tests|verify-inputs|smoke|prepare-manifests|first-round-generate|first-round-normalize-and-select|second-round-generate|finalize|all}" >&2
    exit 2
    ;;
esac
