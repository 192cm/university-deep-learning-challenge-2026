#!/bin/bash
set -o pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
source /venv/main/bin/activate
cd /workspace

bash /workspace/scripts/run_t6.sh 2>&1 | tee -a /workspace/artifacts/t6_sft_v1/supervisor.log
