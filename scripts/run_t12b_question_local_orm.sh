#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python_bin="${PYTHON_BIN:-python3}"
config="configs/t12b_question_local_orm.json"
artifact_root="artifacts/t12b_question_local_orm"
data_root="data/cmu_orm_v2"

mkdir -p "$artifact_root" "$data_root"

hash_file() {
  shasum -a 256 "$1" | awk '{print $1}'
}

submission_before="missing"
t12_manifest_before="missing"
if [[ -f submission.csv ]]; then
  submission_before="$(hash_file submission.csv)"
fi
if [[ -f artifacts/t12_cmu_orm/manifest.json ]]; then
  t12_manifest_before="$(hash_file artifacts/t12_cmu_orm/manifest.json)"
fi

echo "[T12b] preregistering the fixed ranking/aggregation/override grids"
"$python_bin" -m src.train_question_local_orm preregister --config "$config"

echo "[T12b] running focused CPU contract tests"
"$python_bin" scripts/run_unittest_junit.py \
  --output "$artifact_root/tests.xml" \
  --suite-name "T12b question-local ORM focused tests" \
  --module-prefix t12b_test \
  tests/test_question_local_orm_data.py \
  tests/test_question_local_orm_loss.py \
  tests/test_orm_group_selector.py \
  tests/test_orm_selective_override.py

echo "[T12b] sealing T12 diagnosis-only inputs and leaderboard safety records"
"$python_bin" -m src.build_question_local_orm_data \
  freeze-inputs --config "$config"

echo "[T12b] freezing outer-5 / inner-4 template-group ownership"
"$python_bin" -m src.build_question_local_orm_data \
  freeze-splits --config "$config"

echo "[T12b] verifying the separate 6,034 x 16 T5 development inference pool"
"$python_bin" -m src.build_question_local_orm_data \
  verify-dev-pool --config "$config"

echo "[T12b] auditing source-balanced question-local corpus feasibility"
set +e
"$python_bin" -m src.build_question_local_orm_data \
  build-corpus --config "$config"
corpus_status=$?
set -e

submission_after="missing"
t12_manifest_after="missing"
if [[ -f submission.csv ]]; then
  submission_after="$(hash_file submission.csv)"
fi
if [[ -f artifacts/t12_cmu_orm/manifest.json ]]; then
  t12_manifest_after="$(hash_file artifacts/t12_cmu_orm/manifest.json)"
fi
if [[ "$submission_before" != "$submission_after" ]]; then
  echo "T12b changed root submission.csv" >&2
  exit 1
fi
if [[ "$t12_manifest_before" != "$t12_manifest_after" ]]; then
  echo "T12b changed the frozen T12 manifest" >&2
  exit 1
fi

if [[ "$corpus_status" -eq 2 ]]; then
  echo "[T12b] source-balanced corpus data gate failed; no sampling rule was relaxed" >&2
  echo "[T12b] no ORM training, nested OOF evaluation, T13 change, leaderboard run, or submission was produced" >&2
  exit 2
fi
if [[ "$corpus_status" -ne 0 ]]; then
  echo "[T12b] question-local corpus build failed unexpectedly" >&2
  exit "$corpus_status"
fi

if [[ "$(uname -s)" != "Linux" ]] || ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[T12b] the frozen split passed, but this host is not the required 2x RTX 4090 Linux host" >&2
  exit 3
fi
gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
non_4090="$(nvidia-smi --query-gpu=name --format=csv,noheader | awk '$0 != "NVIDIA GeForce RTX 4090" {count++} END {print count+0}')"
if [[ "$gpu_count" != "2" || "$non_4090" != "0" ]]; then
  echo "[T12b] exactly two RTX 4090 GPUs are required; no fallback is allowed" >&2
  exit 3
fi

echo "[T12b] data and hardware gates passed"
echo "[T12b] launch the preregistered nested-CV fold jobs with train_question_local_orm.py train-fold"
echo "[T12b] diagnosis-only T12 fresh/reused results remain excluded from all fit and selection"
