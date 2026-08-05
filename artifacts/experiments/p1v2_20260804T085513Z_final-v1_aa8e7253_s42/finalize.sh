#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/workspace/university-deep-learning-challenge-2026-phase1-v2-final-v1-20260804T085513Z
EXP="$ROOT/artifacts/experiments/p1v2_20260804T085513Z_final-v1_aa8e7253_s42"
PY=/venv/main/bin/python
CONFIG=configs/phase1_v2_final_v1.json
TRAIN=data/deep_chal_math_train_filtered_final_v1.csv
SPLITS=data/splits/phase1_v2_final_v1
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$ROOT"
echo "Waiting for B2 generation to finish"
while pgrep -f '[r]un_baseline.py.*--baseline-id B2' >/dev/null; do
  sleep 20
done

test "$(wc -l < "$EXP/baselines/B2/generations.jsonl")" -eq 11157
echo "B2 complete; evaluating all baselines"
"$PY" scripts/evaluate_generations.py \
  --config "$CONFIG" \
  --train-filtered "$TRAIN" \
  --split-dir "$SPLITS" \
  --generation "B0=$EXP/baselines/B0/generations.jsonl" \
  --generation "B1=$EXP/baselines/B1/generations.jsonl" \
  --generation "B2=$EXP/baselines/B2/generations.jsonl" \
  --output-dir "$EXP/evaluation"

echo "Running independent reproduction checks"
for name in greedy-a greedy-b; do
  "$PY" scripts/run_baseline.py \
    --config "$CONFIG" \
    --baseline-id B0 \
    --train-filtered "$TRAIN" \
    --split-dir "$SPLITS" \
    --id-file "$EXP/reproduction/ids.txt" \
    --max-ids 8 \
    --batch-size 8 \
    --output-dir "$EXP/reproduction/$name"
done
for name in sampling-a sampling-b; do
  "$PY" scripts/run_baseline.py \
    --config "$CONFIG" \
    --baseline-id B2 \
    --train-filtered "$TRAIN" \
    --split-dir "$SPLITS" \
    --id-file "$EXP/reproduction/ids.txt" \
    --max-ids 8 \
    --batch-size 8 \
    --output-dir "$EXP/reproduction/$name"
done

echo "Running final Phase 1 verification"
"$PY" scripts/verify_phase1.py \
  --repo-root . \
  --config "$CONFIG" \
  --split-dir "$SPLITS" \
  --split-rerun-dir "$EXP/split-rerun" \
  --provenance "$EXP/provenance.json" \
  --metrics "$EXP/evaluation/metrics.json" \
  --baseline-dir "B0=$EXP/baselines/B0" \
  --baseline-dir "B1=$EXP/baselines/B1" \
  --baseline-dir "B2=$EXP/baselines/B2" \
  --greedy-repro-a "$EXP/reproduction/greedy-a" \
  --greedy-repro-b "$EXP/reproduction/greedy-b" \
  --sampling-repro-a "$EXP/reproduction/sampling-a" \
  --sampling-repro-b "$EXP/reproduction/sampling-b" \
  --phase0-verification artifacts/experiments/p0_20260803T102000Z_env-smoke_aa8e7253_s42/verification.final.json \
  --output "$EXP/verification.json"

echo "Collecting final environment and running tests"
"$PY" scripts/collect_environment.py --repo-root . --output "$EXP/environment.final.json"
"$PY" -m unittest discover -s tests -p 'test_*.py'

echo "Building report"
mkdir -p report/phase1_v2
"$PY" scripts/build_phase1_v2_report.py \
  --config "$CONFIG" \
  --provenance "$EXP/provenance.json" \
  --split-manifest "$SPLITS/manifest.json" \
  --split-comparison "$EXP/split-comparison.json" \
  --environment "$EXP/environment.final.json" \
  --metrics "$EXP/evaluation/metrics.json" \
  --historical-metrics artifacts/experiments/p1_20260803T110900Z_eval-foundation_aa8e7253_s42/evaluation/metrics.json \
  --verification "$EXP/verification.json" \
  --artifact-dir "$EXP" \
  --output report/phase1_v2/phase1-v2-final-v1-report.md

find "$EXP" -type f ! -name artifact-sha256.txt -print0 \
  | sort -z \
  | xargs -0 sha256sum > "$EXP/artifact-sha256.txt"
echo "FINALIZE_COMPLETE"
