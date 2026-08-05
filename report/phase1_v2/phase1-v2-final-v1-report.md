# Phase 1 v2 evaluation on the final_v1 canonical dataset

## Conclusion

The `data/deep_chal_math_train_filtered_final_v1.csv` canonical dataset (16,359 rows)
was used to reproduce `phase1_v2_final_v1` splits and the B0/B1/B2 base-model baselines
in a fresh remote workspace. Provenance and final Phase 1 verification are
`PASS` and
`PASS`, respectively. Historical generations from
the prior 16,528-row dataset were used only for the explicit comparison below and were not
recycled into this evaluation.

## Canonical data and provenance

| Item | Value |
|---|---|
| canonical path | `data/deep_chal_math_train_filtered_final_v1.csv` |
| canonical rows | 16359 |
| canonical SHA-256 | `9d5de9320861830bf6d0a9113685dd00dcc90b02baa877c4fecc4eb22cd61b4a` |
| source rows | 17000 |
| organizer exclusions | 627 |
| supplemental exclusions | 14 |
| exclusion union / overlap | 641 / 0 |
| unexpected additional removals | 0 |
| canonical manifest SHA-256 | `520d7f2e9291c107cbd00f6ab6d17afaca7f02b661a4d81e54b2c9324d584e90` |
| provenance report SHA-256 | `a7a9a73d50de0c0732723f4f27c4424293cf45d86f7311ddedac623dad67d6d6` |

## Splits

The Random and Template-group validation sets contain
1636 and
1636 rows. The Hard and Format diagnostics contain
554 and
256 rows.

| split file | SHA-256 | bytes |
|---|---|---:|
| format_diagnostic.csv | b9caca62ff63bc5512b7f4bfecb825e7bce5ecefd8cf9815924e8e0c26f61658 | 17406 |
| format_diagnostic_ids.txt | 0253769deeb9d453df0108a92361cf72774ab267bdc39058faf38ba3f6ce68a1 | 3328 |
| hard_diagnostic.csv | 0e41b021412d87b41385ce98c335c5e9c2460f1667eefaa6456fd4470187c369 | 37678 |
| hard_diagnostic_ids.txt | 7ee780bdb1191db57bafa4513431654131c771c1ed82449c72722d38dcee31fb | 7202 |
| leaderboard_filter_audit.csv | b8cc89c8627916259fae78c05479022a331ab7550b57a3692e6d78fdce4dfe95 | 195209 |
| leaderboard_filtered_reproduced.csv | e7a8ebbd7ab617fdc2f1ee18ea42399d8ae3058f5e0655e1cd4963663d3685a2 | 207479 |
| random_split_audit.csv | cc7bbe526d6ca54c7c8b542d44cd9136c1c8b8269859d658b88cf241e53e6cac | 1607782 |
| random_train_ids.txt | a5d3a67681e9648227bcf82c3fce0d0374653a7af4f7eac4168c618fc69a0c68 | 191399 |
| random_validation_ids.txt | 2db070dbd335779a5af1f604a65b297cf1075f11cb8a25e2e320182ff87656ce | 21268 |
| template_split_audit.csv | bb012063c1d055eae8c970252404388cb7063382377282ae0797bbb5cbaf4815 | 6731772 |
| template_train_ids.txt | 2ec3202f7641a75df9427e7b24d58e08cf645fa5e274d877609444955aaa675f | 191399 |
| template_validation_ids.txt | 9ca38bca621a0c144cb5a69e3e104b18d1f1e58819b798b6f71edcb33761231d | 21268 |

### ID changes relative to historical phase1_v1

| ID file | old | new | common | added | removed | symmetric diff |
|---|---:|---:|---:|---:|---:|---:|
| random_train_ids.txt | 14875 | 14723 | 14439 | 284 | 436 | 720 |
| random_validation_ids.txt | 1653 | 1636 | 1586 | 50 | 67 | 117 |
| template_train_ids.txt | 14875 | 14723 | 14459 | 264 | 416 | 680 |
| template_validation_ids.txt | 1653 | 1636 | 1605 | 31 | 48 | 79 |
| hard_diagnostic_ids.txt | 553 | 554 | 511 | 43 | 42 | 85 |
| format_diagnostic_ids.txt | 256 | 256 | 242 | 14 | 14 | 28 |

Comparison CSV: `artifacts/experiments/p1v2_20260804T085513Z_final-v1_aa8e7253_s42/split-comparison.csv` (`9dde92eb3898458e1c70e7843cb283d7054b6863a1762765a549caee0bec3451`)

## Environment and pinned model

| Item | Value |
|---|---|
| GPU | `0, NVIDIA GeForce RTX 4090, GPU-018e4003-2c32-502e-01a4-8037493858bf, 24564, 580.95.05` |
| Python | `3.12.13` |
| model / revision | `Qwen/Qwen2.5-3B-Instruct` / `aa8e72537993ba99e69dfaafa59ed015b17504d1` |
| tokenizer revision | `aa8e72537993ba99e69dfaafa59ed015b17504d1` |
| dtype | `bfloat16` |
| local-files-only generation | `true` |

| package | version |
|---|---|
| accelerate | 1.14.0 |
| bitsandbytes | 0.50.0 |
| huggingface-hub | 1.18.0 |
| peft | 0.20.0 |
| safetensors | 0.8.0 |
| tokenizers | 0.22.2 |
| torch | 2.12.0+cu130 |
| transformers | 5.14.1 |
| trl | 1.9.2 |

## Baseline contract

| baseline | sampling | seeds | max new tokens | temperature | top-p |
|---|---|---|---:|---:|---:|
| B0 | false | `[42]` | 1024 | - | - |
| B1 | false | `[42]` | 1024 | - | - |
| B2 | true | `[42, 2026, 3407]` | 1024 | 0.7000 | 0.9000 |

B3 was not part of the completed historical Phase 1 implementation, so the reproducible
comparison scope remains B0/B1/B2.

## Phase 1 v2 metrics

| baseline | scope | questions | greedy | sample | pass@k | majority@k | agreement@k | invalid | median tokens | p95 latency(s) | est. 1,000q(h) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | random | 1636 | 0.6247 | 0.6247 | 0.6247 | 0.6247 | 0.8527 | 0.1473 | 352.0 | 0.672 | 0.167 |
| B0 | template | 1636 | 0.6363 | 0.6363 | 0.6363 | 0.6363 | 0.8600 | 0.1400 | 336.0 | 0.672 | 0.166 |
| B0 | hard | 554 | 0.2455 | 0.2455 | 0.2455 | 0.2455 | 0.7040 | 0.2960 | 634.5 | 1.016 | 0.201 |
| B0 | format | 256 | 0.4219 | 0.4219 | 0.4219 | 0.4219 | 0.7305 | 0.2695 | 635.5 | 1.016 | 0.172 |
| B1 | random | 1636 | 0.6125 | 0.6125 | 0.6125 | 0.6125 | 0.8252 | 0.1748 | 506.5 | 0.689 | 0.175 |
| B1 | template | 1636 | 0.6339 | 0.6339 | 0.6339 | 0.6339 | 0.8350 | 0.1650 | 502.0 | 0.689 | 0.174 |
| B1 | hard | 554 | 0.2744 | 0.2744 | 0.2744 | 0.2744 | 0.6931 | 0.3069 | 704.5 | 0.812 | 0.197 |
| B1 | format | 256 | 0.4258 | 0.4258 | 0.4258 | 0.4258 | 0.7188 | 0.2812 | 690.5 | 0.812 | 0.177 |
| B2 | random | 1636 | - | 0.5925 | 0.7103 | 0.6559 | 0.6987 | 0.1758 | 504.0 | 0.745 | 0.571 |
| B2 | template | 1636 | - | 0.6151 | 0.7194 | 0.6760 | 0.7213 | 0.1661 | 499.0 | 0.745 | 0.570 |
| B2 | hard | 554 | - | 0.2647 | 0.3646 | 0.3051 | 0.4898 | 0.2888 | 709.5 | 0.869 | 0.637 |
| B2 | format | 256 | - | 0.4076 | 0.5195 | 0.4531 | 0.5404 | 0.2878 | 682.0 | 0.868 | 0.579 |

## Change relative to the historical 16,528-row evaluation

| baseline | scope | score | old | new | delta | invalid-rate delta |
|---|---|---|---:|---:|---:|---:|
| B0 | random | `greedy_accuracy` | 0.6213 | 0.6247 | 0.0034 | 0.0166 |
| B0 | template | `greedy_accuracy` | 0.6407 | 0.6363 | -0.0043 | -0.0034 |
| B0 | hard | `greedy_accuracy` | 0.2532 | 0.2455 | -0.0077 | -0.0132 |
| B0 | format | `greedy_accuracy` | 0.4023 | 0.4219 | 0.0195 | 0.0391 |
| B1 | random | `greedy_accuracy` | 0.6140 | 0.6125 | -0.0016 | 0.0109 |
| B1 | template | `greedy_accuracy` | 0.6261 | 0.6339 | 0.0077 | -0.0037 |
| B1 | hard | `greedy_accuracy` | 0.2495 | 0.2744 | 0.0248 | -0.0060 |
| B1 | format | `greedy_accuracy` | 0.4414 | 0.4258 | -0.0156 | 0.0117 |
| B2 | random | `sample_accuracy` | 0.5787 | 0.5925 | 0.0138 | 0.0046 |
| B2 | template | `sample_accuracy` | 0.5977 | 0.6151 | 0.0174 | -0.0100 |
| B2 | hard | `sample_accuracy` | 0.2411 | 0.2647 | 0.0236 | -0.0114 |
| B2 | format | `sample_accuracy` | 0.3880 | 0.4076 | 0.0195 | 0.0091 |

## Verification

| check | result |
|---|---|
| split_deterministic_hashes_identical | PASS |
| random_split_has_no_id_overlap | PASS |
| template_split_has_no_id_or_group_leakage | PASS |
| leaderboard_audit_has_exactly_1000_unique_ids | PASS |
| data_provenance_passed | PASS |
| source_and_derived_data_match_fixed_hashes | PASS |
| split_manifest_matches_configured_version_and_canonical_hash | PASS |
| all_split_ids_are_canonical_and_exclusions_are_absent | PASS |
| all_baseline_generations_complete_unique_pinned_and_offline | PASS |
| metrics_cover_all_baselines_and_scopes | PASS |
| inference_code_has_no_forbidden_imports_or_execution_calls | PASS |
| independent_greedy_reproduction_exact | PASS |
| seeded_sampling_reproduction_within_tolerance | PASS |
| phase0_verification_remains_passed | PASS |

Independent greedy and seeded-sampling reproduction runs were executed in separate output
directories. The verification artifact records exact text/answer agreement and accuracy tolerance.

## Test-suite result

The full repository suite ran 56 tests:
55 passed and 1 failed.
The single failure was `test_phase0_environment.Phase0EnvironmentTests.test_exact_core_package_versions` because the historical
Phase 0 image expected PyTorch `2.11.0+cu128`, while
this fresh server supplied and the experiment recorded `2.12.0+cu130`.
The focused suite excluding that image-specific test module ran
52 tests and
`PASS`.

## Compliance and limitations

- Generation loaded only `Qwen/Qwen2.5-3B-Instruct` at the pinned revision in offline mode.
- Candidate selection uses extracted model text and vote counts only, with early stopping only
  after model generation finishes.
- No Python/SymPy/solver/calculating verifier/external API/dynamic retrieval is used at inference.
- The remote workspace and files were retained; no recycle or destroy action was issued.
- The PyTorch/CUDA image differs from the historical Phase 0 image. This is recorded as an
  environment deviation; results should not be interpreted as a one-variable data-only ablation.
- Artifact directory: `artifacts/experiments/p1v2_20260804T085513Z_final-v1_aa8e7253_s42`
