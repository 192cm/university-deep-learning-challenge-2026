#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

artifact_root=artifacts/t8_6_base_vote_policy
mkdir -p "$artifact_root"

python scripts/run_unittest_junit.py \
  --output "$artifact_root/tests.xml" \
  --suite-name "T8-6 focused tests" \
  --module-prefix t8_6_test \
  tests/test_extract.py \
  tests/test_submit.py \
  tests/test_vote_filter.py \
  tests/test_rft_vote_policy_search.py \
  tests/test_t8_6_base_vote_policy.py

python -m analysis.t8_6_base_vote_policy \
  --config configs/t8_6_base_vote_policy.json

echo '{"event":"t8_6_base_vote_policy_complete"}'
