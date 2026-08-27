#!/bin/bash
set -o pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
source /venv/main/bin/activate
cd /workspace
mkdir -p /workspace/artifacts/t11_aimo_generation_quality

bash /workspace/scripts/run_t11.sh all 2>&1 \
  | tee -a /workspace/artifacts/t11_aimo_generation_quality/supervisor.log
