# Phase 2 verification

Artifact verification: **PASS**

Phase 2 remains incomplete because the Luna quality gate failed.

| check | result | detail |
|---|---|---|
| immutable_train_sha256 | PASS | `"94f3302a6240b91b6fb3d093696b898750b8c4ca1d8ae1eb54210358664af9df"` |
| immutable_leaderboard_sha256 | PASS | `"f00b83805479140fb4d59fedb01c092e16c6cd35ac588f387b281ffea55eb2d7"` |
| env_is_git_ignored | PASS | `true` |
| api_key_leaks | PASS | `{"count": 0, "paths": []}` |
| phase1_ids_transmitted | PASS | `0` |
| local_quality_holdout_ids_transmitted | PASS | `0` |
| all_request_ids_allowed | PASS | `0` |
| audit_requests_only_use_luna_audit_ids | PASS | `433` |
| answer_hidden_request_body_audit | PASS | `{"checked": 433, "errors": {}}` |
| main_batch_not_submitted_after_failed_gate | PASS | `0` |
| paid_cost_hard_limit | PASS | `0.3957592` |
| no_active_cost_reservations | PASS | `0` |
| final_sft_jsonl | PASS | `{"errors": [], "rows": 0, "unique_ids": 0}` |
| audit_ids_in_final_sft | PASS | `0` |
| grade_d_excluded | PASS | `0` |
| manifest_blocked_status_honest | PASS | `"blocked_quality_gate"` |
| leaderboard_duplicates_in_final_sft | PASS | `{"exact": 0, "near": 0, "template": 0}` |
| external_curriculum_schema_and_ids | PASS | `{"errors": {}, "rows": 50000, "unique_ids": 50000}` |
| external_curriculum_hash | PASS | `"0e61b2abef33fa303f98474830ad28f8e3ed32e3c4ab94412ac71934de9ae8c8"` |
| external_contamination_removed | PASS | `{"accepted_exact_or_template": 0, "accepted_near": 0}` |
| external_source_hash | PASS | `"93700f39cdc87994f2c9a6ad62e1d62e09467793f57d376cfab722757ff26e1c"` |
| audit_candidate_table_rows | PASS | `400` |
| stratified_audit_table_rows | PASS | `{"efforts": 2, "rows_per_effort": 100}` |
| generation_status_rows | PASS | `12428` |
| manifest_output_hashes | PASS | `{}` |
