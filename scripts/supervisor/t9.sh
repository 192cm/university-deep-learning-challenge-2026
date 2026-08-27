#!/bin/bash
set -o pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
source /venv/main/bin/activate
cd /workspace
mkdir -p /workspace/artifacts/t9_genselect

bash /workspace/scripts/run_t9.sh 2>&1 | tee -a /workspace/artifacts/t9_genselect/supervisor.log
