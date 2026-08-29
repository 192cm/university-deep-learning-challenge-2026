# T12b-dev execution result

Status: `data_gate_failed` (development decision not run)

The outer 5-fold / inner 4-fold template-group split and the separate 6,034 x 16 T5 candidate pool were frozen successfully. After removing two duplicate traces, exact per-source 1:1 label balance can retain at most **4970** questions, below the preregistered minimum of **5000**. The row upper bound is 25602, so the binding failure is question coverage.

## Capacity proof

- `t12_base_cot_brief_high_temperature_k16` label 0: mandatory 1262, balanced capacity 231, exclusions >= 1031
- `t7_base_hard_tail_k32` label 1: mandatory 207, balanced capacity 172, exclusions >= 35

Their mandatory-question overlap is 2, so at least 1,064 of 6,034 questions must be excluded and at most 4,970 remain. No sampling rule was relaxed. GPU training, OOF evaluation, leaderboard inference, T13 promotion, and submission writing were not started.
