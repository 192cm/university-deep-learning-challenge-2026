#!/bin/bash
set -o pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
source /venv/main/bin/activate
cd /workspace
mkdir -p /workspace/artifacts/t7_rft_r2

bash /workspace/scripts/run_t7.sh 2>&1 | tee -a /workspace/artifacts/t7_rft_r2/supervisor.log
