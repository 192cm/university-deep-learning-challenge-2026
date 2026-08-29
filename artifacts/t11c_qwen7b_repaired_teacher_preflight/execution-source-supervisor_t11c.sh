#!/bin/bash
set -o pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"

stage=${1:?usage: supervisor_t11c.sh STAGE}
cd /workspace
mkdir -p /workspace/artifacts/t11c_qwen7b_repaired_teacher_preflight

bash /workspace/scripts/run_t11c.sh "${stage}" 2>&1 \
  | tee -a "/workspace/artifacts/t11c_qwen7b_repaired_teacher_preflight/supervisor-${stage}.log"
