# T10a C-1 — cot-boxed + frozen vote-quality filter

No new generation, training, or filter search was performed.

| Strategy | Union accuracy | Δ vs C-1 |
|---|---:|---:|
| T10a C | 69.36% | -1.18pp |
| T10a C-1 | 70.54% | +0.00pp |
| T8 base | 69.31% | -1.23pp |
| T8-3 | 70.78% | +0.24pp |

| Split | C | C-1 | T8 | T8-3 |
|---|---:|---:|---:|---:|
| random_holdout | 75.44% | 76.05% | 74.28% | 75.57% |
| template_holdout | 73.18% | 74.71% | 73.98% | 75.50% |
| hard_diagnostic | 38.36% | 39.64% | 39.64% | 40.73% |
| format_diagnostic | 53.52% | 54.30% | 50.39% | 53.12% |

## Paired comparisons

- C-1 vs C: +1.177pp, p=5.66e-07, recovered/broken=61/17.
- C-1 vs T8: +1.231pp, p=0.00256.
- C-1 vs T8-3: -0.241pp, p=0.557.

Decision: **HOLD**. The filter materially repairs C, but C-1 misses the +1.5pp adoption threshold versus T8 and does not beat the same filter on the base prompt (T8-3).
