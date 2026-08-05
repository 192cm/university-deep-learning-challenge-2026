# f0_local_answer_only_final_v1_v1 QA

Generated at: `2026-08-05T17:13:48.274219+00:00`

## Dataset

- Source: `data/deep_chal_math_train_filtered_final_v1.csv`
- Source SHA-256: `9d5de9320861830bf6d0a9113685dd00dcc90b02baa877c4fecc4eb22cd61b4a`
- Input rows: 16,359
- Output rows: 12,631
- Excluded unique rows: 3,728
- Output JSONL SHA-256: `51e572b03ee6456e8eea4b6aec8a794ac6e35709f892ddb7ef79175754d2b6d7`
- Audit CSV SHA-256: `f9034bd90cc0ebb3610f3a197a084c01333769853d96fc061b00ca7af7f94566`
- Protected ID list SHA-256: `549b072707990198dbd5cb4b46ae1ae8e9303eca360a42e82ea20b261896669c`

## Protection counts

The Phase 1 source counts below are non-exclusive; their union is 3,719 IDs.

- `random_validation`: 1,636
- `template_validation`: 1,636
- `hard_diagnostic`: 554
- `format_diagnostic`: 256
- `phase2_quality_exclusion`: 9
- `final_sft_protected`: 0
- Combined protected union: 3,728

## Pool decision

The configured dataset uses the answer-only-specific pool (12,631 rows), not the
strict Phase 2 teacher-eligible pool (12,209 rows). It adds 422 rows
that Phase 2 reserves or filters for teacher generation, while retaining every explicit Phase 1,
confirmed label/problem-quality, and final-SFT protection required for F0.

## Checks

- PASS — `source_schema_exact`
- PASS — `source_ids_unique`
- PASS — `source_answers_canonical`
- PASS — `audit_covers_every_source_row_once`
- PASS — `output_ids_unique`
- PASS — `output_ids_match_selected_pool`
- PASS — `protected_output_intersection_zero`
- PASS — `all_targets_one_line_canonical`
- PASS — `grade_is_local_answer_only`
- PASS — `strict_pool_is_subset_of_dedicated_pool`
- PASS — `source_unchanged`
- PASS — `two_full_size_runs_byte_identical`

No model training, external API call, answer repair, or mathematical verification was performed.
