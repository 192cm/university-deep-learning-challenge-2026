# T8-3 vote-filter comparison

The frozen filter removes weak-path, truncated, or conflicting votes and uses
the original per-question majority only when every usable vote was removed.
Ground truth was loaded only after both prediction maps were frozen.

| Scope | T8 majority@32 | T8-3 filtered | Delta | McNemar p |
|---|---:|---:|---:|---:|
| union (3737) | 69.31% | 70.78% | +1.47pp | 6.8e-10 |
| random_holdout (1637) | 74.28% | 75.57% | +1.28pp | 0.000104 |
| template_holdout (1637) | 73.98% | 75.50% | +1.53pp | 2.24e-05 |
| hard_diagnostic (550) | 39.64% | 40.73% | +1.09pp | 0.146 |
| format_diagnostic (256) | 50.39% | 53.12% | +2.73pp | 0.0654 |

## Preregistered decision

**HOLD** — union accuracy improved, but the preregistered +1.5pp effect-size gate was not met

The holdout policy was a post-hoc discovery. The filtered 831-row leaderboard
artifact is retained as a label-blind candidate, but the frozen final strategy
remains unfiltered T8 fixed majority@32 unless the adoption gate is changed in a
separate documented decision.
