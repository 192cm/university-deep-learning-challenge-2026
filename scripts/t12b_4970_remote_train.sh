#!/usr/bin/env bash
set -euo pipefail

cd /workspace
python_bin=/venv/main/bin/python
runtime_config=data/cmu_orm_v2_4970/runtime-config.json
artifact_root=artifacts/t12b_4970_override
mode="${1:?expected dev or final}"

case "$mode" in
  dev)
    output_dir="$artifact_root/development/model"
    command_name=train-fold
    fold_args=(--fold 0)
    ;;
  final)
    output_dir="$artifact_root/final-model"
    command_name=train-final
    fold_args=()
    ;;
  *)
    echo "unknown training mode: $mode" >&2
    exit 2
    ;;
esac

if [[ -f "$output_dir/train-metrics.json" ]]; then
  echo "$mode training is already complete"
  exit 0
fi

exec env CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$python_bin" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=2 \
  -m src.train_question_local_orm "$command_name" \
  --config "$runtime_config" \
  "${fold_args[@]}" \
  --tau 1.0 --lambda-pair 1.0 --lambda-list 0.0 \
  --output-dir "$output_dir"
