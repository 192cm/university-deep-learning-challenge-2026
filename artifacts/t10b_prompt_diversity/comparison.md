# T10b prompt diversity comparison

| Arm | Strategy | Union accuracy | Δ vs A | McNemar p | Decision |
|---|---|---:|---:|---:|---|
| A | base_single_prompt | 69.31% | — | — | reference |
| C | diverse_prompts | 68.80% | -0.508pp | 0.19496 | reject |

Arm E was excluded before generation because T10a was held and E is byte-identical to A.
Final decision: **reject**. Arm C failed to improve or violated a preregistered guardrail/runtime gate.
T10c input arm: `A`.

## Prompt-level majority@4

| Prompt | Accuracy | Sample accuracy | Agreement@4 | Hit-max | Invalid |
|---|---:|---:|---:|---:|---:|
| en_forward_free_no_check | 65.13% | 60.02% | 74.07% | 1.07% | 0.47% |
| en_forward_numbered_check | 64.49% | 59.92% | 73.86% | 1.16% | 0.44% |
| en_backward_free_check | 65.27% | 60.53% | 73.98% | 1.02% | 0.35% |
| en_backward_numbered_no_check | 61.23% | 55.16% | 69.56% | 1.10% | 0.34% |
| ko_forward_free_check | 62.00% | 54.82% | 68.45% | 1.17% | 0.36% |
| ko_forward_numbered_no_check | 58.39% | 49.19% | 63.64% | 1.02% | 0.29% |
| ko_backward_free_no_check | 59.17% | 50.40% | 64.14% | 0.86% | 0.41% |
| ko_backward_numbered_check | 54.54% | 46.76% | 61.05% | 1.17% | 0.37% |

Predictions were frozen before labels were loaded. No prior generation pool was overwritten.
