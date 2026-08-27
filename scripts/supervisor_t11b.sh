#!/bin/bash
set -o pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"

stage=${1:?usage: supervisor_t11b.sh STAGE}
cd /workspace
mkdir -p /workspace/artifacts/t11b_deepseek14b_teacher_preflight

bash /workspace/scripts/run_t11b.sh "${stage}" 2>&1 \
  | tee -a "/workspace/artifacts/t11b_deepseek14b_teacher_preflight/supervisor-${stage}.log"

