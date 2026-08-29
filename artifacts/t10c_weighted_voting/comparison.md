# T10c weighted-vote comparison

Predictions were frozen before canonical labels were loaded. Candidate weights
use extraction path, output length, hit-max state, and explicit-answer conflict
metadata only; no arithmetic verifier or question-dependent feature is used.

| Policy | Union accuracy | Delta vs T8 | Exact McNemar p | Weighted tie | Fallback | Decision |
|---|---:|---:|---:|---:|---:|---|
| unfiltered_majority_k32 | 69.31% | +0.00pp | 1 | 4.66% | 0 | reference |
| policy1 | 70.78% | +1.47pp | 6.8e-10 | 5.03% | 47 | control |
| policy2 | 70.48% | +1.18pp | 6.21e-08 | 2.44% | 1 | hold |
| policy3 | 70.48% | +1.18pp | 6.21e-08 | 2.41% | 1 | hold |
| policy4 | 70.46% | +1.15pp | 8.91e-07 | 1.90% | 1 | hold |

## Continuous policies versus binary policy 1

| Continuous policy | Delta vs policy 1 | Exact McNemar p | 95% CI |
|---|---:|---:|---:|
| policy2 | -0.29pp | 0.0801 | [-0.60, +0.01]pp |
| policy3 | -0.29pp | 0.0801 | [-0.60, +0.01]pp |
| policy4 | -0.32pp | 0.073 | [-0.64, +0.00]pp |

## Four fixed splits

| Policy | Random | Template | Hard | Format |
|---|---:|---:|---:|---:|
| unfiltered_majority_k32 | 74.28% | 73.98% | 39.64% | 50.39% |
| policy1 | 75.57% | 75.50% | 40.73% | 53.12% |
| policy2 | 75.26% | 75.26% | 40.73% | 51.95% |
| policy3 | 75.26% | 75.26% | 40.73% | 51.95% |
| policy4 | 75.20% | 75.20% | 40.73% | 52.34% |

## Preregistered decision

**HOLD** — policy2 had the largest positive eligible delta but did not pass every adoption gate; retain unfiltered T8 majority@32
