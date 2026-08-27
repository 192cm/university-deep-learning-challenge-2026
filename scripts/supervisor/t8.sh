#!/bin/bash
set -o pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
source /venv/main/bin/activate
cd /workspace
mkdir -p /workspace/artifacts/t8_self_consistency

bash /workspace/scripts/run_t8.sh 2>&1 | tee -a /workspace/artifacts/t8_self_consistency/supervisor.log
