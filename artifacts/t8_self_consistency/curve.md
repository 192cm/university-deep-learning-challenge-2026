# T8 self-consistency accuracy-time curve

All accuracy points are paired prefixes of one immutable k=32 base-model pool. The k=32 wall time is measured; shorter-prefix times are linearized from that run. Tie votes always choose the earliest generated answer.

| Policy | Avg samples | Generations | Majority | Pass | Agreement | Tie | 1,000 questions | Δ vs T4 greedy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_k4 | 4.00 | 14,948 | 64.68% | 73.11% | 74.63% | 19.48% | 0.111 h | +1.98 pp |
| fixed_k8 | 8.00 | 29,896 | 67.22% | 77.71% | 72.26% | 11.85% | 0.216 h | +4.52 pp |
| fixed_k16 | 16.00 | 59,792 | 68.45% | 81.08% | 71.16% | 7.25% | 0.424 h | +5.75 pp |
| fixed_k32 | 32.00 | 119,584 | 69.31% | 84.40% | 70.44% | 4.66% | 0.840 h | +6.61 pp |
| adaptive_4_to_32_replay | 18.12 | 67,700 | 69.28% | 82.79% | 72.82% | 4.66% | 0.479 h | +6.58 pp |
| budget_matched_fixed_control | 18.12 | 67,700 | 68.53% | 81.78% | 70.98% | 6.58% | 0.479 h | +5.83 pp |
| adaptive_4_to_32_staged | 18.15 | 67,840 | 69.31% | 83.22% | 72.72% | 4.36% | 0.743 h | +6.61 pp |

## Adaptive budget comparison

Adaptive replay used 67,700 generations. At exactly the same budget, it changed majority accuracy by +0.75 pp versus the answer- and label-blind fixed allocation control.

## Selected setting

Selected: `fixed_k32`. The 1,000-question estimate is 0.840 hours, leaving more than the required six-hour reserve inside the 24-hour budget.

| Split | Accuracy | Invalid sample rate |
|---|---:|---:|
| format_diagnostic | 50.39% | 2.12% |
| hard_diagnostic | 39.64% | 1.39% |
| random_holdout | 74.28% | 0.53% |
| template_holdout | 73.98% | 0.64% |
