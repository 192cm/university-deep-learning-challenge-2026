# T8-2 disagreement-routed CoT comparison

All A/B/C predictions use the same 3,737 IDs and exactly 32 model outputs per question. Routes and votes were frozen before labels were loaded.

| Comparison | Reference | Candidate | Δ | Bootstrap 95% CI | A→wrong | A→correct | Discordant | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C routed vs A | 69.31% | 69.31% | +0.00 pp | [-0.70, +0.70] pp | 89 | 89 | 178 | 1 |
| B strong-CoT vs A | 69.31% | 69.52% | +0.21 pp | ablation only | 99 | 107 | 206 | 0.6259 |

## Split guardrails

| Split | A | B | C | C−A |
|---|---:|---:|---:|---:|
| random_holdout | 74.28% | 74.89% | 74.71% | +0.43 pp |
| template_holdout | 73.98% | 73.92% | 73.67% | -0.31 pp |
| hard_diagnostic | 39.64% | 40.18% | 39.82% | +0.18 pp |
| format_diagnostic | 50.39% | 53.12% | 53.12% | +2.73 pp |

## Preregistered decision

**REJECT** — Primary routed accuracy did not improve over preserved T8.

The strong-CoT fixed arm is reported only as the preregistered ablation; it cannot replace the primary candidate post hoc.
