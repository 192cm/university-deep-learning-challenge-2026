# T8-1 paired comparison

Primary comparison: T6-4 RFT LoRA fixed majority@32 versus the preserved T4c base fixed majority@32 on the same 3,737 questions.

| Scope | Reference | Candidate | Δ | 95% CI | Ref→wrong | Ref→correct | Discordant | Exact p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| union | 69.31% | 69.84% | +0.54 pp | [-0.29, +1.36] pp | 114 | 134 | 248 | 0.2276 |

| Split | Reference | Candidate | Δ | Guardrail |
|---|---:|---:|---:|---|
| random_holdout | 74.28% | 75.08% | +0.79 pp | report only |
| template_holdout | 73.98% | 74.22% | +0.24 pp | report only |
| hard_diagnostic | 39.64% | 39.45% | -0.18 pp | PASS |
| format_diagnostic | 50.39% | 53.52% | +3.12 pp | PASS |

## Preregistered decision

**HOLD** — union accuracy improved but the preregistered +1.5 pp and p<0.05 adoption gate was not fully met; preserve T4c + T8

Ground truth was used only after candidate generation, adaptive stopping, budget allocation, and voting were complete.
