#!/usr/bin/env bash
set -euo pipefail

cd /workspace
source /venv/main/bin/activate

export HF_HOME=/workspace/.hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export VLLM_WORKER_MULTIPROC_METHOD=spawn

artifact_root=artifacts/t8_2_cot_routing
reference_root=artifacts/t8_self_consistency
strong_root="$artifact_root/strong_cot"
smoke_root="$artifact_root/smoke"
runtime_root="$artifact_root/runtime_probe"
mkdir -p \
  "$artifact_root" \
  "$strong_root" \
  "$smoke_root/base" \
  "$smoke_root/strong_cot" \
  "$runtime_root/stage1" \
  "$runtime_root/stage2_base" \
  "$runtime_root/stage2_strong_cot"

# This snapshot is created once. On resume the command verifies it instead of
# replacing the pre-run baseline.
python -m src.cot_routing snapshot-invariants \
  --path configs/t8_self_consistency.json \
  --path scripts/run_t8.sh \
  --path configs/t8_1_rft_self_consistency.json \
  --path scripts/run_t8_1.sh \
  --path configs/t9_genselect.json \
  --path scripts/run_t9.sh \
  --path src/extract.py \
  --tree artifacts/t8_self_consistency \
  --tree artifacts/t8_1_rft_self_consistency \
  --tree artifacts/t9_genselect \
  --output "$artifact_root/invariant-snapshot.json"

python -m src.cot_routing validate-reference \
  --config configs/t8_2_cot_routing.json \
  --reference-config configs/t8_self_consistency.json \
  --reference-generations "$reference_root/generations.jsonl" \
  --reference-metadata "$reference_root/run-metadata.json" \
  --reference-final-config "$reference_root/final_config.json" \
  --reference-manifest "$reference_root/manifest.json" \
  --union-ids "$reference_root/holdout_union_ids.txt" \
  --hard-split data/splits/hard_diagnostic.csv \
  --output "$artifact_root/reference-validation.json"

python -m pytest -q \
  tests/test_extract.py \
  tests/test_evaluate.py \
  tests/test_generate.py \
  tests/test_self_consistency.py \
  tests/test_genselect.py \
  tests/test_cot_routing.py \
  --junitxml="$artifact_root/tests.xml"

# A 32-question two-prompt smoke validates prompt hashes, sample indices,
# extraction, routing, voting, and complete-cache resume before the main run.
python -m src.cot_routing select-ids \
  --source "$reference_root/holdout_union_ids.txt" \
  --output "$smoke_root/ids.txt" \
  --count 32 \
  --seed 8202

python -u src/generate.py \
  --config configs/t8_2_cot_routing.json \
  --input data/canonical/train.csv \
  --ids-file "$smoke_root/ids.txt" \
  --output "$smoke_root/base/generations.jsonl" \
  --metadata "$smoke_root/base/run-metadata.json" \
  --engine vllm \
  --prompt-mode base \
  2>&1 | tee "$smoke_root/base-generation.log"

python -u src/generate.py \
  --config configs/t8_2_cot_routing.json \
  --input data/canonical/train.csv \
  --ids-file "$smoke_root/ids.txt" \
  --output "$smoke_root/strong_cot/generations.jsonl" \
  --metadata "$smoke_root/strong_cot/run-metadata.json" \
  --engine vllm \
  --prompt-mode strong_cot \
  2>&1 | tee "$smoke_root/strong-cot-generation.log"

# A second identical invocation must be a byte-preserving cache hit.
python -u src/generate.py \
  --config configs/t8_2_cot_routing.json \
  --input data/canonical/train.csv \
  --ids-file "$smoke_root/ids.txt" \
  --output "$smoke_root/base/generations.jsonl" \
  --metadata "$smoke_root/base/run-metadata.json" \
  --engine vllm \
  --prompt-mode base \
  2>&1 | tee "$smoke_root/base-resume.log"

python -u src/generate.py \
  --config configs/t8_2_cot_routing.json \
  --input data/canonical/train.csv \
  --ids-file "$smoke_root/ids.txt" \
  --output "$smoke_root/strong_cot/generations.jsonl" \
  --metadata "$smoke_root/strong_cot/run-metadata.json" \
  --engine vllm \
  --prompt-mode strong_cot \
  2>&1 | tee "$smoke_root/strong-cot-resume.log"

python -m src.cot_routing route \
  --config configs/t8_2_cot_routing.json \
  --union-ids "$smoke_root/ids.txt" \
  --reference-generations "$smoke_root/base/generations.jsonl" \
  --reference-metadata "$smoke_root/base/run-metadata.json" \
  --reference-task T8-2 \
  --strong-generations "$smoke_root/strong_cot/generations.jsonl" \
  --strong-metadata "$smoke_root/strong_cot/run-metadata.json" \
  --output-routes "$smoke_root/routes.jsonl" \
  --output-predictions "$smoke_root/predictions.jsonl" \
  --output-freeze "$smoke_root/routing-freeze.json"

# Immutable strong-CoT k=32 ablation pool over the exact preserved T8 union.
python -u src/generate.py \
  --config configs/t8_2_cot_routing.json \
  --input data/canonical/train.csv \
  --ids-file "$reference_root/holdout_union_ids.txt" \
  --output "$strong_root/generations.jsonl" \
  --metadata "$strong_root/run-metadata.json" \
  --engine vllm \
  --prompt-mode strong_cot

# The primary A/B pool composition and prediction freeze remain label-free.
python -m src.cot_routing route \
  --config configs/t8_2_cot_routing.json \
  --union-ids "$reference_root/holdout_union_ids.txt" \
  --reference-generations "$reference_root/generations.jsonl" \
  --reference-metadata "$reference_root/run-metadata.json" \
  --reference-task T8 \
  --strong-generations "$strong_root/generations.jsonl" \
  --strong-metadata "$strong_root/run-metadata.json" \
  --output-routes "$artifact_root/routes.jsonl" \
  --output-predictions "$artifact_root/predictions.jsonl" \
  --output-freeze "$artifact_root/routing-freeze.json"

# Execute a production-shaped staged path on a fixed 256-question probe.
python -m src.cot_routing select-ids \
  --source "$reference_root/holdout_union_ids.txt" \
  --output "$runtime_root/ids.txt" \
  --count 256 \
  --seed 820256

python -u src/generate.py \
  --config configs/t8_2_cot_routing.json \
  --input data/canonical/train.csv \
  --ids-file "$runtime_root/ids.txt" \
  --output "$runtime_root/stage1/generations.jsonl" \
  --metadata "$runtime_root/stage1/run-metadata.json" \
  --engine vllm \
  --prompt-mode base \
  --n 4 \
  --seed 820204

python -m src.cot_routing prepare-runtime \
  --ids "$runtime_root/ids.txt" \
  --stage1-generations "$runtime_root/stage1/generations.jsonl" \
  --stage1-metadata "$runtime_root/stage1/run-metadata.json" \
  --stage1-seed 820204 \
  --output-base-ids "$runtime_root/base-ids.txt" \
  --output-strong-ids "$runtime_root/strong-cot-ids.txt" \
  --output "$runtime_root/preparation.json"

python -u src/generate.py \
  --config configs/t8_2_cot_routing.json \
  --input data/canonical/train.csv \
  --ids-file "$runtime_root/base-ids.txt" \
  --output "$runtime_root/stage2_base/generations.jsonl" \
  --metadata "$runtime_root/stage2_base/run-metadata.json" \
  --engine vllm \
  --prompt-mode base \
  --n 28 \
  --seed 820232

python -u src/generate.py \
  --config configs/t8_2_cot_routing.json \
  --input data/canonical/train.csv \
  --ids-file "$runtime_root/strong-cot-ids.txt" \
  --output "$runtime_root/stage2_strong_cot/generations.jsonl" \
  --metadata "$runtime_root/stage2_strong_cot/run-metadata.json" \
  --engine vllm \
  --prompt-mode strong_cot \
  --n 28 \
  --seed 820232

python -m src.cot_routing build-runtime \
  --ids "$runtime_root/ids.txt" \
  --preparation "$runtime_root/preparation.json" \
  --base-ids "$runtime_root/base-ids.txt" \
  --strong-ids "$runtime_root/strong-cot-ids.txt" \
  --stage1-generations "$runtime_root/stage1/generations.jsonl" \
  --stage1-metadata "$runtime_root/stage1/run-metadata.json" \
  --base-generations "$runtime_root/stage2_base/generations.jsonl" \
  --base-metadata "$runtime_root/stage2_base/run-metadata.json" \
  --strong-generations "$runtime_root/stage2_strong_cot/generations.jsonl" \
  --strong-metadata "$runtime_root/stage2_strong_cot/run-metadata.json" \
  --stage1-seed 820204 \
  --stage2-seed 820232 \
  --extrapolated-questions 1000 \
  --output "$artifact_root/runtime.json"

# Only this final command receives labels, after route/prediction hashes exist.
python -m src.cot_routing evaluate \
  --config configs/t8_2_cot_routing.json \
  --canonical data/canonical/train.csv \
  --union-ids "$reference_root/holdout_union_ids.txt" \
  --split random_holdout=data/splits/random_holdout.csv \
  --split template_holdout=data/splits/template_holdout.csv \
  --split hard_diagnostic=data/splits/hard_diagnostic.csv \
  --split format_diagnostic=data/splits/format_diagnostic.csv \
  --reference-generations "$reference_root/generations.jsonl" \
  --reference-metadata "$reference_root/run-metadata.json" \
  --strong-generations "$strong_root/generations.jsonl" \
  --strong-metadata "$strong_root/run-metadata.json" \
  --routes "$artifact_root/routes.jsonl" \
  --predictions "$artifact_root/predictions.jsonl" \
  --routing-freeze "$artifact_root/routing-freeze.json" \
  --runtime "$artifact_root/runtime.json" \
  --reference-validation "$artifact_root/reference-validation.json" \
  --invariant-snapshot "$artifact_root/invariant-snapshot.json" \
  --tests-xml "$artifact_root/tests.xml" \
  --output-dir "$artifact_root"

python -m src.cot_routing verify-snapshot \
  --snapshot "$artifact_root/invariant-snapshot.json"

tar -czf "$artifact_root/t8-2-final-metadata.tgz" \
  "$artifact_root/invariant-snapshot.json" \
  "$artifact_root/reference-validation.json" \
  "$artifact_root/routing-freeze.json" \
  "$artifact_root/routes.jsonl" \
  "$artifact_root/predictions.jsonl" \
  "$artifact_root/comparison.json" \
  "$artifact_root/comparison.md" \
  "$artifact_root/runtime.json" \
  "$artifact_root/final_config.json" \
  "$artifact_root/manifest.json" \
  "$artifact_root/tests.xml"

echo '{"event":"t8_2_complete"}'
