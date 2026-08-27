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

artifact_root=artifacts/t9_genselect
data_root=data/genselect
mkdir -p "$artifact_root" "$artifact_root/resolved-configs" "$artifact_root/adapters" "$data_root"

python -m pytest -q \
  tests/test_extract.py \
  tests/test_generate.py \
  tests/test_train_sft.py \
  tests/test_genselect.py \
  --junitxml="$artifact_root/tests-preflight.xml"

python -u -m src.genselect build-data \
  --config configs/t9_genselect.json \
  --canonical data/canonical/train.csv \
  --rft-ids data/rft_pool_ids.txt \
  --r1-audit data/rft_r1/audit.csv \
  --r1-generations artifacts/t5_rft_r1/generations.jsonl \
  --r2-candidates data/rft_r2/candidates.jsonl \
  --holdout-ids data/splits/random_holdout.csv \
  --holdout-ids data/splits/template_holdout.csv \
  --holdout-ids data/splits/hard_diagnostic.csv \
  --holdout-ids data/splits/format_diagnostic.csv \
  --output-dir "$data_root"

for spec in lr_1em05:0.00001 lr_3em05:0.00003 lr_1em04:0.0001; do
  label=${spec%%:*}
  lr=${spec##*:}
  config="$artifact_root/resolved-configs/$label.json"
  hp_root="$artifact_root/hp/$label"
  mkdir -p "$hp_root"
  python -m src.genselect resolve-config \
    --config configs/t9_genselect.json \
    --learning-rate "$lr" \
    --output "$config"
  python -u src/train_sft.py prepare \
    --config "$config" \
    --input "$data_root/train.jsonl" \
    --cache "$hp_root/cache" \
    --metadata "$hp_root/cache-metadata.json"
done

python -u src/train_sft.py calibrate \
  --config "$artifact_root/resolved-configs/lr_3em05.json" \
  --cache "$artifact_root/hp/lr_3em05/cache" \
  --cache-metadata "$artifact_root/hp/lr_3em05/cache-metadata.json" \
  --output "$artifact_root/calibration.json" \
  --work-dir "$artifact_root/calibration-probes" \
  --environment environment.json

for spec in lr_1em05:0.00001 lr_3em05:0.00003 lr_1em04:0.0001; do
  label=${spec%%:*}
  lr=${spec##*:}
  config="$artifact_root/resolved-configs/$label.json"
  hp_root="$artifact_root/hp/$label"
  python -u src/train_sft.py train \
    --config "$config" \
    --cache "$hp_root/cache" \
    --cache-metadata "$hp_root/cache-metadata.json" \
    --calibration "$artifact_root/calibration.json" \
    --output "$hp_root/training-metrics.json" \
    --work-dir "$hp_root/trainer" \
    --adapter-dir "$hp_root/final-adapter" \
    --experiment "t9_$label"

  for checkpoint in "$hp_root"/trainer/checkpoints/checkpoint-*; do
    checkpoint_name=$(basename "$checkpoint")
    validation_root="$hp_root/validation/$checkpoint_name"
    adapter_root="$validation_root/adapter"
    mkdir -p "$adapter_root"
    cp "$checkpoint/adapter_config.json" "$adapter_root/adapter_config.json"
    cp "$checkpoint/adapter_model.safetensors" "$adapter_root/adapter_model.safetensors"
    python -u src/generate.py \
      --config configs/t9_genselect.json \
      --input "$data_root/validation.csv" \
      --output "$validation_root/generations.jsonl" \
      --metadata "$validation_root/run-metadata.json" \
      --engine vllm \
      --adapter "$adapter_root"
    python -m src.genselect score-selection \
      --cases "$data_root/validation.jsonl" \
      --generations "$validation_root/generations.jsonl" \
      --checkpoint "$checkpoint" \
      --learning-rate "$lr" \
      --output "$validation_root/score.json"
  done
done

python -m src.genselect select-hp \
  --artifact-root "$artifact_root" \
  --adapter-dir "$artifact_root/adapters/selector" \
  --output "$artifact_root/hp-sweep.json"

python -u -m src.genselect prepare-evaluation \
  --config configs/t9_genselect.json \
  --canonical data/canonical/train.csv \
  --union-ids artifacts/t8_self_consistency/holdout_union_ids.txt \
  --t8-generations artifacts/t8_self_consistency/generations.jsonl \
  --output-dir "$artifact_root/evaluation"

mkdir -p "$artifact_root/calibration" "$artifact_root/evaluation/fewshot" "$artifact_root/evaluation/adapter" "$artifact_root/evaluation/shuffle"
python -u src/generate.py \
  --config configs/t9_genselect.json \
  --input "$data_root/validation.csv" \
  --output "$artifact_root/calibration/generations.jsonl" \
  --metadata "$artifact_root/calibration/run-metadata.json" \
  --engine vllm \
  --max-prompts 128

python -u src/generate.py \
  --config configs/t9_genselect.json \
  --input "$artifact_root/evaluation/evaluation.csv" \
  --output "$artifact_root/evaluation/fewshot/generations.jsonl" \
  --metadata "$artifact_root/evaluation/fewshot/run-metadata.json" \
  --engine vllm

python -u src/generate.py \
  --config configs/t9_genselect.json \
  --input "$artifact_root/evaluation/evaluation.csv" \
  --output "$artifact_root/evaluation/adapter/generations.jsonl" \
  --metadata "$artifact_root/evaluation/adapter/run-metadata.json" \
  --engine vllm \
  --adapter "$artifact_root/adapters/selector"

python -u src/generate.py \
  --config configs/t9_genselect.json \
  --input "$artifact_root/evaluation/shuffle.csv" \
  --output "$artifact_root/evaluation/shuffle/generations.jsonl" \
  --metadata "$artifact_root/evaluation/shuffle/run-metadata.json" \
  --engine vllm \
  --adapter "$artifact_root/adapters/selector"

python -u -m src.genselect finalize \
  --config configs/t9_genselect.json \
  --canonical data/canonical/train.csv \
  --union-ids artifacts/t8_self_consistency/holdout_union_ids.txt \
  --split random=data/splits/random_holdout.csv \
  --split template=data/splits/template_holdout.csv \
  --split hard=data/splits/hard_diagnostic.csv \
  --split format=data/splits/format_diagnostic.csv \
  --t8-generations artifacts/t8_self_consistency/generations.jsonl \
  --t8-metadata artifacts/t8_self_consistency/run-metadata.json \
  --t8-final-config artifacts/t8_self_consistency/final_config.json \
  --cases "$artifact_root/evaluation/evaluation-cases.jsonl" \
  --shuffle-cases "$artifact_root/evaluation/shuffle-cases.jsonl" \
  --adapter-generations "$artifact_root/evaluation/adapter/generations.jsonl" \
  --adapter-metadata "$artifact_root/evaluation/adapter/run-metadata.json" \
  --fewshot-generations "$artifact_root/evaluation/fewshot/generations.jsonl" \
  --fewshot-metadata "$artifact_root/evaluation/fewshot/run-metadata.json" \
  --shuffle-generations "$artifact_root/evaluation/shuffle/generations.jsonl" \
  --shuffle-metadata "$artifact_root/evaluation/shuffle/run-metadata.json" \
  --adapter-dir "$artifact_root/adapters/selector" \
  --data-manifest "$data_root/manifest.json" \
  --hp-sweep "$artifact_root/hp-sweep.json" \
  --output-dir "$artifact_root"

python -m pytest -q \
  tests/test_extract.py \
  tests/test_evaluate.py \
  tests/test_generate.py \
  tests/test_train_sft.py \
  tests/test_genselect.py \
  --junitxml="$artifact_root/tests.xml"

tar -czf "$artifact_root/t9-final-metadata.tgz" \
  "$artifact_root/manifest.json" \
  "$artifact_root/metrics.json" \
  "$artifact_root/comparison.md" \
  "$artifact_root/hp-sweep.json" \
  "$artifact_root/calibration.json" \
  "$artifact_root/tests.xml" \
  "$data_root/manifest.json"

echo '{"event":"t9_complete"}'
