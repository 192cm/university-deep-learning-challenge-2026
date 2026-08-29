#!/usr/bin/env bash
set -euo pipefail

cd /workspace

t11b_venv=${T11B_VENV:-/workspace/.venvs/t11b}
if [[ -f "$t11b_venv/bin/activate" ]]; then
  source "$t11b_venv/bin/activate"
else
  source /venv/main/bin/activate
fi

export HF_HOME=/workspace/.hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_BATCH_INVARIANT=1
export HF_HUB_DISABLE_XET=1

config=configs/t11b_deepseek14b_teacher_preflight.json
data_root=data/t11b_deepseek14b_teacher_preflight
artifact_root=artifacts/t11b_deepseek14b_teacher_preflight

mkdir -p "$data_root" "$artifact_root"

run_tests() {
  python -m pytest -q \
    tests/test_extract.py \
    tests/test_t11.py \
    tests/test_normalize_teacher_trace.py \
    --junitxml="$artifact_root/tests.xml"
}

verify_inputs() {
  python -m src.normalize_teacher_trace verify-inputs --config "$config"
}

historical_replay() {
  python -m src.normalize_teacher_trace historical-replay --config "$config"
}

smoke() {
  python -u -m src.normalize_teacher_trace smoke --config "$config"
}

teacher_preflight_generate() {
  python -u -m src.normalize_teacher_trace teacher-generate --config "$config"
}

normalize_frozen_raw() {
  python -m src.normalize_teacher_trace normalize \
    --config "$config" \
    --input "$data_root/raw_teacher_generations.jsonl" \
    --output "$data_root/normalized_teacher_generations.jsonl" \
    --audit "$data_root/normalization-audit.jsonl"
}

evaluate_gate() {
  python -m src.normalize_teacher_trace evaluate --config "$config"
}

finalize() {
  python -m src.normalize_teacher_trace finalize --config "$config"
}

run_all() {
  run_tests
  verify_inputs
  historical_replay
  if ! smoke; then
    python -m src.normalize_teacher_trace finalize \
      --config "$config" \
      --forced-status teacher_load_failed
    return 3
  fi
  teacher_preflight_generate
  normalize_frozen_raw
  evaluate_gate
  finalize
}

case "${1:-all}" in
  tests) run_tests ;;
  verify-inputs) verify_inputs ;;
  historical-replay) historical_replay ;;
  smoke) smoke ;;
  teacher-generate) teacher_preflight_generate ;;
  normalize) normalize_frozen_raw ;;
  evaluate) evaluate_gate ;;
  finalize) finalize ;;
  all) run_all ;;
  *)
    echo "usage: $0 {tests|verify-inputs|historical-replay|smoke|teacher-generate|normalize|evaluate|finalize|all}" >&2
    exit 2
    ;;
esac
