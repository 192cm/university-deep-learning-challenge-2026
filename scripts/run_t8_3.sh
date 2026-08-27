#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

artifact_root=artifacts/t8_3_vote_filter
mkdir -p "$artifact_root"

python scripts/run_unittest_junit.py \
  --output "$artifact_root/tests.xml" \
  tests/test_extract.py \
  tests/test_evaluate.py \
  tests/test_submit.py \
  tests/test_vote_filter.py

python -m src.vote_filter \
  --config configs/t8_3_vote_filter.json

echo '{"event":"t8_3_vote_filter_complete"}'
