#!/usr/bin/env bash
set -euo pipefail

cd /workspace
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home

mkdir -p data/external_cot
exec > >(tee -a data/external_cot/build.log) 2>&1

python -u -m src.build_external_cot --config configs/t5_external_cot.json
