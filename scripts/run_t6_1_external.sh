#!/usr/bin/env bash
set -euo pipefail

cd /workspace
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home

mkdir -p data/external_cot_v2
exec > >(tee -a data/external_cot_v2/build.log) 2>&1

if [[ ! -f data/external_cot_v2/manifest.json ]] || \
   ! jq -e '([.completion_checks[]] | all)' data/external_cot_v2/manifest.json >/dev/null; then
  python -u -m src.build_external_cot --config configs/t6_1_external_cot.json
fi
