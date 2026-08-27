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
mkdir -p "$root/precision-probe/hf_nf4"

python -u src/generate.py \
  --config configs/t4_output_contract.json \
  --input data/canonical/train.csv \
  --ids-file data/splits/random_holdout_ids.txt \
  --output "$root/precision-probe/hf_nf4/generations.jsonl" \
  --metadata "$root/precision-probe/hf_nf4/run-metadata.json" \
  --engine hf \
  --adapter artifacts/t6_sft_v1/adapters/rft_r1 \
  --hf-load-in-4bit

