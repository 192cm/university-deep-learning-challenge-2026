#!/bin/bash
set -o pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
source /venv/main/bin/activate
cd /workspace
mkdir -p /workspace/artifacts/t10b_prompt_diversity

bash /workspace/scripts/run_t10b.sh 2>&1 \
  | tee -a /workspace/artifacts/t10b_prompt_diversity/supervisor.log
