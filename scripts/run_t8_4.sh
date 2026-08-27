#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

artifact_root=artifacts/t8_4_rft_vote_filter
mkdir -p "$artifact_root"

python scripts/run_unittest_junit.py \
  --output "$artifact_root/tests.xml" \
  --suite-name "T8-4 focused tests" \
  --module-prefix t8_4_test \
  tests/test_submit.py \
  tests/test_vote_filter.py

python -m analysis.t8_4_rft_vote_filter \
  --config configs/t8_4_rft_vote_filter.json \
  --output "$artifact_root/experiment.json"

echo '{"event":"t8_4_rft_vote_filter_complete"}'
