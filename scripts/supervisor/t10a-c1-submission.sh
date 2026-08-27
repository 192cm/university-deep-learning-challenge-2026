#!/bin/bash
set -o pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
source /venv/main/bin/activate
cd /workspace
mkdir -p /workspace/artifacts/submissions/t10a_c1_filtered_k32

bash /workspace/scripts/run_t10a_c1_submission.sh 2>&1 \
  | tee -a /workspace/artifacts/submissions/t10a_c1_filtered_k32/pipeline.log
