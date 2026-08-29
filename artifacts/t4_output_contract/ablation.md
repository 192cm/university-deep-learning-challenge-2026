# T4 output-contract ablation

The three rows isolate extraction fallback from the larger output budget. Conditions (a) and (b) read the exact same T3 `generations.jsonl` bytes; (b) incurred no new GPU generation.

## Random holdout primary ablation

| condition | max new tokens | extractor | accuracy | delta vs (a) | invalid | hit max | mean output tokens | source generation | incremental T4 GPU | evaluation |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| (a) | 1024 | historical B0 | 64.203% | +0.000pp | 15.027% | 9.530% | 443.768 | 282.356s | 0.000s | 0.377s |
| (b) | 1024 | T1 fallback | 67.074% | +2.871pp | 0.489% | 9.530% | 443.768 | 282.356s | 0.000s | 0.378s |
| (c) | 2048 | T1 fallback | 67.990% | +3.787pp | 0.611% | 2.810% | 491.143 | 434.725s | 434.725s | 0.383s |

## All fixed holdouts

| condition | split | accuracy | invalid | hit max | mean output tokens |
|---|---|---:|---:|---:|---:|
| (a) | format_diagnostic | 32.031% | 43.359% | 35.547% | 732.965 |
| (a) | hard_diagnostic | 28.727% | 31.818% | 20.182% | 631.909 |
| (a) | random_holdout | 64.203% | 15.027% | 9.530% | 443.768 |
| (a) | template_holdout | 64.630% | 14.539% | 9.102% | 442.389 |
| (b) | format_diagnostic | 38.281% | 1.562% | 35.547% | 732.965 |
| (b) | hard_diagnostic | 30.909% | 1.273% | 20.182% | 631.909 |
| (b) | random_holdout | 67.074% | 0.489% | 9.530% | 443.768 |
| (b) | template_holdout | 66.158% | 0.672% | 9.102% | 442.389 |
| (c) | format_diagnostic | 46.875% | 1.953% | 5.078% | 870.676 |
| (c) | hard_diagnostic | 32.545% | 1.636% | 4.909% | 736.600 |
| (c) | random_holdout | 67.990% | 0.611% | 2.810% | 491.143 |
| (c) | template_holdout | 66.952% | 0.672% | 2.749% | 491.351 |

## Format diagnostic integer regressions

| condition | category | questions | accuracy | invalid |
|---|---|---:|---:|---:|
| (a) | all_format | 256 | 32.031% | 43.359% |
| (a) | large_integer_gt_10_digits | 11 | 0.000% | 18.182% |
| (a) | negative | 68 | 51.471% | 29.412% |
| (a) | zero | 68 | 54.412% | 29.412% |
| (b) | all_format | 256 | 38.281% | 1.562% |
| (b) | large_integer_gt_10_digits | 11 | 0.000% | 0.000% |
| (b) | negative | 68 | 58.824% | 1.471% |
| (b) | zero | 68 | 60.294% | 0.000% |
| (c) | all_format | 256 | 46.875% | 1.953% |
| (c) | negative | 68 | 67.647% | 1.471% |
| (c) | zero | 68 | 64.706% | 0.000% |
| (c) | large_integer_gt_10_digits | 11 | 0.000% | 0.000% |

## 2048-token calibration

Selected `bench_vllm_512_s256` at 7.126 generations/s, mean GPU utilization 98.938%, peak VRAM 23770.3 MiB, and zero OOM events.

| trial | max num seqs | generations/s | mean GPU | peak VRAM MiB | OOM events |
|---|---:|---:|---:|---:|---:|
| bench_vllm_512_s128 | 128 | 5.512 | 98.925% | 23206.3 | 0 |
| bench_vllm_512_s192 | 192 | 6.378 | 98.758% | 23528.3 | 0 |
| bench_vllm_512_s256 | 256 | 7.126 | 98.938% | 23770.3 | 0 |
| determinism_200_a | 256 | 3.665 | 99.091% | 23770.3 | 0 |
| determinism_200_b | 256 | 3.661 | 99.091% | 23770.3 | 0 |

## Completion checks

- [x] `all_three_ablation_rows_present`
- [x] `all_four_holdouts_covered`
- [x] `controls_share_identical_t3_generation_bytes`
- [x] `fallback_measured_without_new_generation`
- [x] `format_invalid_rate_below_3_percent`
- [x] `random_invalid_rate_below_5_percent`
- [x] `random_accuracy_improved_over_condition_a`
- [x] `format_numeric_categories_covered`
- [x] `max_new_tokens_is_2048`
- [x] `hf_batch_token_budget_preserved`
- [x] `hf_batch_size_reduced`
- [x] `vllm_sequence_slots_reduced`
- [x] `selected_calibration_no_oom`
- [x] `selected_calibration_gpu_mean_at_least_90_percent`
- [x] `full_2048_run_no_oom`
- [x] `full_2048_run_gpu_mean_at_least_90_percent`
- [x] `same_seed_calibration_probe_byte_identical`
- [x] `raw_t3_and_t4_generations_preserved`

Ground-truth labels were used only for metrics. Extraction performs notation-only string parsing and no mathematical calculation or candidate reranking.
