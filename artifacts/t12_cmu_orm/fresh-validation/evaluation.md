# T12 CMU-MATH pointwise ORM evaluation

Decision: **HOLD**

The fresh 1,000-question validation is the only adoption set. ORM geometric weighted majority@32 scored 87.6000%; raw majority@32 scored 87.4000%; frozen T8-3 filter@32 scored 87.4000%. The delta to the stronger baseline (raw_majority) is +0.2000%.

Paired McNemar p=0.790527; paired bootstrap 95% CI=[-0.5000%, +0.9000%]; candidate ROC-AUC=0.8123.

## Reproduction basis and deliberate substitution

- CMU-MATH placed second in AIMO Progress Prize 1 with a private score of 22/50.
- Its policy and reward model began from the same DeepSeekMath-7B-RL checkpoint; the pointwise reward model scored one problem-solution trace from 0 to 1.
- Answers were grouped and ranked by vote count multiplied by the geometric mean reward. Its reported public-train ablation was 2/10 for majority@32 and 4/10 for ORM weighting. Ten questions are not treated as generalizable evidence.
- CMU reported about 7,000 unique questions and 37,880 problem-solution pairs with per-problem 1:1 correct/incorrect balance, two epochs, and learning rate 2e-5.
- The local frozen corpus contains 6,034 unique questions and 30,912 problem-solution rows; this observed scale difference is retained rather than described as an exact CMU data reproduction.
- The sole deliberate local substitution is: same-base rank-64 LoRA sequence-classification adapter instead of CMU full reward-model fine-tuning Both solver and ORM remain separate adapters/models from the pinned Qwen2.5-3B-Instruct competition base.
- The aggregation is the preregistered unpenalized formula above; the answer=0 penalty shown in CMU's released inference snippet is not imported or tuned here.
- The [AIMO-2 paper](https://arxiv.org/abs/2504.16891) states that GenSelect was not deployed in the winning Kaggle submission because of time constraints. Its evidence is therefore not mixed with the competition-validated CMU ORM basis used here.

Sources: [CMU ML blog](https://blog.ml.cmu.edu/2024/07/29/cmu-math-teams-innovative-approach-secures-2nd-place-at-the-aimo-prize/), [Kaggle write-up](https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/writeups/cmu-math-2nd-place-solution-all-code-and-datasets-), [CMU-MATH code](https://github.com/AIMO-CMU-MATH/CMU_MATH-AIMO).

## Presentation cumulative-table row

| Stage | Fresh validation accuracy | Delta vs stronger baseline | Decision |
|---|---:|---:|---|
| + CMU-MATH pointwise ORM geometric weighted majority@32 (T12) | 87.60% | +0.20pp | HOLD |

## Gate

```json
{
  "all_five_fold_deltas_positive": false,
  "beats_raw_majority": true,
  "beats_t8_3_filter": true,
  "bootstrap_ci_lower_above_zero": false,
  "both_gpus_oom_zero": true,
  "candidate_roc_auc_at_least_0_65": true,
  "delta_at_least_1_5pp": false,
  "format_drop_within_2pp": true,
  "fresh_makespan_within_18h": true,
  "hard_drop_within_2pp": true,
  "mcnemar_p_below_0_05": false,
  "nan_scores_zero": true
}
```
