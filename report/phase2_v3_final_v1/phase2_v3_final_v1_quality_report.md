# phase2_v3_final_v1 quality report

Pipeline status: **blocked_comparison_gate**.

| stage | effort | completion | truncation | completed JSON | completed schema | canonical integer | first exact | pass@2 | verifier fatal | gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| comparison | low | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% | 52.5% | 52.5% | 0.0% | fail |
| comparison | medium | 97.5% | 2.5% | 100.0% | 100.0% | 100.0% | 52.5% | 55.0% | 0.0% | fail |
| smoke | low | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% | 50.0% | 70.0% | 0.0% | pass |

Noncanonical integer outputs across new measured stages: 0.
Completed-response JSON and schema rates exclude incomplete responses from their denominators.
Complex equations that cannot be checked safely are recorded as `not_checked_complex_expression`, not as errors.
