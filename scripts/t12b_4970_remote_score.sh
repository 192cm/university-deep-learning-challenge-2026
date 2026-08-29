#!/usr/bin/env bash
set -euo pipefail

cd /workspace
python_bin=/venv/main/bin/python
config=configs/t12_cmu_orm.json
artifact_root=artifacts/t12b_4970_override
mode="${1:?expected dev or final}"

case "$mode" in
  dev)
    adapter="$artifact_root/development/model/adapter"
    candidates="$artifact_root/development/generations.jsonl"
    questions="$artifact_root/development/questions.csv"
    score_root="$artifact_root/development/scoring"
    ;;
  final)
    adapter="$artifact_root/final-model/adapter"
    candidates=artifacts/submissions/t12_cmu_orm_831/generations.jsonl
    questions=data/deep_chal_math_leaderboard_filtered.csv
    score_root="$artifact_root/leaderboard-label-blind/scoring"
    ;;
  *)
    echo "unknown scoring mode: $mode" >&2
    exit 2
    ;;
esac

mkdir -p "$score_root/shards"
if [[ -f "$score_root/score-merge-audit.json" && -f "$score_root/candidate-scores.jsonl" ]]; then
  echo "$mode scoring is already complete"
  exit 0
fi
if [[ ! -f "$adapter/adapter_model.safetensors" ]]; then
  echo "adapter is incomplete: $adapter" >&2
  exit 3
fi

adapter_sha="$($python_bin - "$adapter" <<'PY'
import sys
from pathlib import Path
from src.train_orm import sha256_tree
print(sha256_tree(Path(sys.argv[1])))
PY
)"

manifest="$score_root/score-shard-manifest.json"
if [[ ! -f "$manifest" ]]; then
  "$python_bin" -m src.t12_sharding create-score-manifest \
    --candidates "$candidates" \
    --adapter-sha256 "$adapter_sha" \
    --config "$config" \
    --output "$manifest"
fi

mapfile -t gpu_uuids < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
if [[ "${#gpu_uuids[@]}" -ne 2 ]]; then
  echo "expected exactly two GPUs" >&2
  exit 4
fi

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

for rank in 0 1; do
  CUDA_VISIBLE_DEVICES="$rank" "$python_bin" -m src.orm_score worker \
    --config "$config" \
    --manifest "$manifest" \
    --logical-rank "$rank" \
    --physical-index "$rank" \
    --expected-uuid "${gpu_uuids[$rank]}" \
    --candidates "$candidates" \
    --questions "$questions" \
    --adapter "$adapter" \
    --output "$score_root/shards/shard-$rank.jsonl" \
    --metadata "$score_root/shards/worker-metadata-$rank.json" \
    >"$score_root/shards/worker-$rank.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
  tail -80 "$score_root"/shards/worker-*.log >&2 || true
  exit 5
fi
trap - EXIT INT TERM

"$python_bin" -m src.t12_sharding merge \
  --manifest "$manifest" \
  --shard "$score_root/shards/shard-0.jsonl" \
  --shard "$score_root/shards/shard-1.jsonl" \
  --worker-metadata "$score_root/shards/worker-metadata-0.json" \
  --worker-metadata "$score_root/shards/worker-metadata-1.json" \
  --output "$score_root/candidate-scores.jsonl" \
  --audit "$score_root/score-merge-audit.json"
