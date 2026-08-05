# Phase 2 v2 verification

Overall: **PASS** (38/38)

| check | result | detail |
|---|---|---|
| filtered_sha256 | PASS | `"2844386e4d5c7355f773ac58f4b735be4e9ce1caa70b4b1a3576244e1346ff98"` |
| filtered_rows_and_schema | PASS | `{"columns": ["id", "question", "answer"], "rows": 16528}` |
| filtered_unique_ids | PASS | `16528` |
| filtered_integer_label_audit | PASS | `[]` |
| immutable_train_sha256 | PASS | `"94f3302a6240b91b6fb3d093696b898750b8c4ca1d8ae1eb54210358664af9df"` |
| immutable_leaderboard_sha256 | PASS | `"f00b83805479140fb4d59fedb01c092e16c6cd35ac588f387b281ffea55eb2d7"` |
| leaderboard_full_protection_rows | PASS | `1000` |
| phase2_holdout_ids_unique | PASS | `100` |
| phase2_holdout_ids_subset_filtered | PASS | `0` |
| phase2_audit_ids_unique | PASS | `150` |
| phase2_audit_ids_subset_filtered | PASS | `0` |
| phase2_eligible_ids_unique | PASS | `12255` |
| phase2_eligible_ids_subset_filtered | PASS | `0` |
| teacher_request_ids_unique | PASS | `50` |
| teacher_request_ids_subset_filtered | PASS | `0` |
| final_sft_ids_unique | PASS | `0` |
| final_sft_ids_subset_filtered | PASS | `0` |
| holdout_audit_eligible_disjoint | PASS | `{"audit_eligible": 0, "holdout_audit": 0, "holdout_eligible": 0}` |
| protected_ids_transmitted | PASS | `[]` |
| answer_hidden_request_contract | PASS | `[]` |
| teacher_request_ids_match_manifest | PASS | `{"id_file": 50, "manifest": 50}` |
| input_manifest_static_output_hashes | PASS | `[]` |
| final_jsonl_schema_and_integer_contract | PASS | `[]` |
| final_id_uniqueness | PASS | `0` |
| final_ids_subset_phase2_eligible | PASS | `[]` |
| final_row_limit | PASS | `0` |
| audit_ids_excluded_from_final | PASS | `[]` |
| historical_audit_ids_excluded_from_final | PASS | `[]` |
| final_sft_ids_match_jsonl | PASS | `0` |
| legacy_v1_reuse_conditions_audited | PASS | `{"errors": [], "rows": 400}` |
| leaderboard_exact_template_near_duplicates_final | PASS | `{"exact": [], "near": [], "template": []}` |
| cumulative_paid_cost_hard_limit | PASS | `0.5429184` |
| committed_cost_hard_limit | PASS | `0.5429184` |
| no_active_reservations | PASS | `{}` |
| api_key_leaks | PASS | `[]` |
| dataset_manifest_exists | PASS | `"data\\phase2\\phase2_verified_cot_luna_3k_v2\\dataset_manifest.json"` |
| manifest_final_hash | PASS | `"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"` |
| manifest_runtime_artifact_hashes | PASS | `[]` |
