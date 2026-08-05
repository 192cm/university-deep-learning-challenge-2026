# phase2_v4_final_v1_terra verification

Overall: **PASS** (48/48)

| check | result | detail |
|---|---|---|
| filtered_sha256 | PASS | `"9d5de9320861830bf6d0a9113685dd00dcc90b02baa877c4fecc4eb22cd61b4a"` |
| filtered_rows_and_schema | PASS | `{"columns": ["id", "question", "answer"], "rows": 16359}` |
| filtered_unique_ids | PASS | `16359` |
| filtered_integer_label_audit | PASS | `[]` |
| immutable_train_sha256 | PASS | `"94f3302a6240b91b6fb3d093696b898750b8c4ca1d8ae1eb54210358664af9df"` |
| immutable_leaderboard_sha256 | PASS | `"f00b83805479140fb4d59fedb01c092e16c6cd35ac588f387b281ffea55eb2d7"` |
| leaderboard_full_protection_rows | PASS | `1000` |
| phase2_holdout_ids_unique | PASS | `100` |
| phase2_holdout_ids_subset_filtered | PASS | `0` |
| phase2_audit_ids_unique | PASS | `150` |
| phase2_audit_ids_subset_filtered | PASS | `0` |
| phase2_eligible_ids_unique | PASS | `12209` |
| phase2_eligible_ids_subset_filtered | PASS | `0` |
| teacher_request_ids_unique | PASS | `50` |
| teacher_request_ids_subset_filtered | PASS | `0` |
| final_sft_ids_unique | PASS | `0` |
| final_sft_ids_subset_filtered | PASS | `0` |
| holdout_audit_eligible_disjoint | PASS | `{"audit_eligible": 0, "holdout_audit": 0, "holdout_eligible": 0}` |
| protected_ids_transmitted | PASS | `[]` |
| answer_hidden_request_contract | PASS | `[]` |
| teacher_request_ids_match_manifest | PASS | `{"id_file": 50, "manifest": 50}` |
| experiment_scope_request_stages | PASS | `{"allowed": ["comparison", "smoke"], "observed": ["comparison", "smoke"]}` |
| quality_audit_not_executed | PASS | `0` |
| main_batch_not_executed | PASS | `{"batch_artifacts": [], "batch_events_exists": false, "main_records": 0}` |
| fixed_comparison_ids_match_v3 | PASS | `"f274b0de8242fd50f4ade519a33975af9fc4bfbd5153162fba7f23103cd34961"` |
| teacher_prompt_matches_v3 | PASS | `"840f3789c73479ebabdd3a7144016d8becb162e1b42448e8ca4eb90f2f639072"` |
| teacher_schema_matches_v3 | PASS | `"ff96634b249f9af72cbb23121df79b09a3826286a5c8c16284e9417e691369b8"` |
| raw_responses_match_request_manifest | PASS | `{"manifest": 200, "raw": 200}` |
| input_manifest_static_output_hashes | PASS | `[]` |
| final_jsonl_schema_and_integer_contract | PASS | `[]` |
| final_id_uniqueness | PASS | `0` |
| final_ids_subset_phase2_eligible | PASS | `[]` |
| final_row_limit | PASS | `0` |
| audit_ids_excluded_from_final | PASS | `[]` |
| historical_audit_ids_excluded_from_final | PASS | `[]` |
| final_sft_ids_match_jsonl | PASS | `0` |
| phase1_protected_ids_excluded_from_final | PASS | `[]` |
| legacy_v1_reuse_conditions_audited | PASS | `{"errors": [], "rows": 2}` |
| leaderboard_exact_template_near_duplicates_final | PASS | `{"exact": [], "near": [], "template": []}` |
| leaderboard_decontamination_audit_complete | PASS | `1000` |
| cumulative_paid_cost_hard_limit | PASS | `3.4784912` |
| committed_cost_hard_limit | PASS | `3.4784912` |
| no_active_reservations | PASS | `{}` |
| safety_reserve_maintained | PASS | `{"remaining_usd": 1.0215087999999999, "required_reserve_usd": 0.5}` |
| api_key_leaks | PASS | `[]` |
| dataset_manifest_exists | PASS | `"data\\phase2\\phase2_v4_final_v1_terra\\dataset_manifest.json"` |
| manifest_final_hash | PASS | `"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"` |
| manifest_runtime_artifact_hashes | PASS | `[]` |
