#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

artifact_root=artifacts/t10a_c1_vote_filter
mkdir -p "$artifact_root"

python scripts/run_unittest_junit.py \
  --output "$artifact_root/tests.xml" \
  --suite-name "T10a C-1 focused tests" \
  --module-prefix t10a_c1_test \
  tests/test_vote_filter.py \
  tests/test_t10a_c1_vote_filter.py

python -m analysis.t10a_c1_vote_filter \
  --config configs/t10a_c1_vote_filter.json \
  --output "$artifact_root/experiment.json" \
  --summary "$artifact_root/summary.md"

echo '{"event":"t10a_c1_vote_filter_complete"}'
