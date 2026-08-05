# phase2_v4_final_v1_terra summary

Status: **blocked_comparison_gate**. Terra smoke passed at both efforts, but neither fixed comparison passed. No 100-row audit or main Batch was run.

## v3 Luna vs v4 Terra

| model | effort | completion | first exact | pass@2 | unsuitable | truncation | non-integer | gate |
|---|---|---:|---:|---:|---:|---:|---:|---|
| gpt-5.6-luna | low | 100.0% | 52.5% | 52.5% | 18/80 | 0/80 | 0 | FAIL |
| gpt-5.6-luna | medium | 97.5% | 52.5% | 55.0% | 18/80 | 2/80 | 0 | FAIL |
| gpt-5.6-terra | low | 100.0% | 50.0% | 55.0% | 21/80 | 0/80 | 1 | FAIL |
| gpt-5.6-terra | medium | 96.2% | 55.0% | 57.5% | 21/80 | 3/80 | 0 | FAIL |

- Low: first exact -2.5%p, pass@2 +2.5%p versus v3 Luna.
- Medium: first exact +2.5%p, pass@2 +2.5%p versus v3 Luna.
- Medium added pass@2 correctness on 2 rows and lost it on 1 rows versus Terra low.
- Terra low/medium answer agreement: 60.0% / 65.0%.

## Fixed comparison gates

| gate | low | medium |
|---|---:|---:|
| response_completion | PASS | FAIL |
| completed_json_parse | PASS | PASS |
| completed_schema | PASS | PASS |
| canonical_integer_extraction | FAIL | PASS |
| first_candidate_accuracy | FAIL | FAIL |
| pass_at_2 | FAIL | FAIL |
| verifier_fatal_error_rate | PASS | PASS |
| non_integer_final_answers | FAIL | PASS |

## Unsuitable and truncation impact

- Low unsuitable: 21/80 requests across 12 rows; those rows contributed 2 pass@2 successes.
- Medium unsuitable: 21/80 requests across 11 rows; those rows contributed 1 pass@2 successes.
- Medium truncation/incomplete: 3 requests across 2 rows; those rows contributed 0 pass@2 successes. See `report\phase2_v4_final_v1_terra\phase2_v4_final_v1_terra_truncations.csv` for IDs and reasons.

## Cost and hashes

- Cumulative cost / remaining: $3.4784912 / $1.0215088.
- Terra-only cost: $2.7573700; successful/final failed requests: 200/0.
- Raw response tree SHA-256: `06553b0da1fb36997ada83ef75b522f7f2450c04cb99778d29ec6ab74114c50d`.
- Diagnostic bundle SHA-256: `655ab3baf23d1a4ed2e6dd234b7e6d6bc0c57a157f049eef0db5eef03d645939`.

## Recommendation

Do not advance Terra to the 100-row quality audit or main Batch. Terra produced a small mixed accuracy gain over Luna but failed completion/canonical/accuracy gates and cost substantially more. Keep Phase 2 blocked and request separate approval before any Sol experiment.
