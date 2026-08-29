# T9 GenSelect comparison

The fixed holdout union contains 3,737 questions. The selector uses 4 independent prompts with 16 summarized candidates each.

| Path | Total generation budget | Union accuracy | Delta vs majority@32 |
|---|---:|---:|---:|
| T8 majority@32 | 32 | 69.31% | — |
| Adapter GenSelect, 32 solve + 4 select | 36 | 55.90% | -13.41pp |
| Few-shot GenSelect, 32 solve + 4 select | 36 | 65.40% | -3.91pp |
| Selected GenSelect, 28 solve + 4 select | 32 | 65.80% | -3.51pp |

Selector adapter vs few-shot: -9.50pp. Adapter output mean: 125.9 tokens. Candidate-order shuffle answer consistency: 58.98%.

Decision: **t8_fixed_majority32**. GenSelect is adopted only if the selected selector beats majority@32 under the strict 28+4=32 generation comparison; the adapter is used only if it also beats the few-shot control.
