# T8-1 equivalent-condition accuracy-time curve

Both fixed curves use the same 3,737 IDs, prompt, extractor, sampling settings, and paired k=32 prefixes. The only solver change is the preregistered T6-4 LoRA.

| Solver | Policy | Avg samples | Majority | Pass | Agreement | Tie | 1,000 questions |
|---|---|---:|---:|---:|---:|---:|---:|
| T4c base | fixed_k4 | 4.00 | 64.68% | 73.11% | 74.63% | 19.48% | 0.111 h |
| T4c base | fixed_k8 | 8.00 | 67.22% | 77.71% | 72.26% | 11.85% | 0.216 h |
| T4c base | fixed_k16 | 16.00 | 68.45% | 81.08% | 71.16% | 7.25% | 0.424 h |
| T4c base | fixed_k32 | 32.00 | 69.31% | 84.40% | 70.44% | 4.66% | 0.840 h |
| T6-4 RFT LoRA | fixed_k4 | 4.00 | 65.59% | 73.94% | 75.47% | 18.49% | 0.089 h |
| T6-4 RFT LoRA | fixed_k8 | 8.00 | 67.84% | 77.84% | 73.30% | 10.78% | 0.167 h |
| T6-4 RFT LoRA | fixed_k16 | 16.00 | 69.04% | 81.46% | 72.13% | 7.20% | 0.323 h |
| T6-4 RFT LoRA | fixed_k32 | 32.00 | 69.84% | 84.43% | 71.61% | 3.88% | 0.635 h |
| T6-4 RFT LoRA | adaptive_4_to_32_replay | 17.70 | 69.87% | 83.17% | 73.52% | 3.85% | 0.356 h |
| T6-4 RFT LoRA | budget_matched_fixed_control | 17.70 | 68.99% | 81.94% | 72.01% | 6.07% | 0.356 h |
| T6-4 RFT LoRA | adaptive_4_to_32_staged | 17.48 | 69.47% | 83.22% | 73.71% | 4.17% | 0.881 h |

## Actual staged adaptive control

The separately executed adaptive path used 65,320 generations and changed accuracy by +0.48 pp versus the answer- and label-blind fixed allocation at exactly the same count.

## Frozen candidate setting

Candidate policy: `fixed_k32`; estimated 1,000-question runtime 0.635 h.
