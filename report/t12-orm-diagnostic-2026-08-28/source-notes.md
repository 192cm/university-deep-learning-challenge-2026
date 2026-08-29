# T12 ORM diagnostic source notes

## Reporting job

- Question: why did a candidate-level ORM with useful ROC-AUC produce only a +0.2pp final voting gain?
- Audience: product stakeholders / competition decision-makers.
- Scope: the frozen T12 fresh validation of 1,000 questions; reused T8 is corroborating diagnostics only.
- Baseline: raw majority@32, tied at 87.4% with the frozen T8-3 filter.
- Decision criterion: explain the HOLD outcome without changing the preregistered T12 formula or decision.

## Required-structure mapping

- Title: `T12 ORM이 채택 기준을 넘지 못한 이유`.
- Executive Summary: preserved verbatim as the first section after the title.
- Key findings with visual evidence: accuracy comparison, raw-error decomposition, selector failure mechanisms, calibration/ranking evidence, and strata evidence.
- Recommended next steps: a new preregistered question-local ranking task; no post-hoc T12 tuning.
- Further questions: transfer and failure-mechanism checks that would change the next experiment.
- Caveats and assumptions: same fresh set is used for post-hoc diagnosis; subgroup overlap; reused T8 is non-primary.

## Source inventory

- `artifacts/t12_cmu_orm/fresh-validation/evaluation.json`: frozen top-line metrics, statistical tests, candidate metrics, folds, strata, runtime, and official basis.
- `artifacts/t12_cmu_orm/fresh-validation/group-weights.jsonl`: answer groups, support counts, geometric means, and weighted predictions.
- `artifacts/t12_cmu_orm/fresh-validation/predictions.jsonl`: ORM weighted and argmax predictions.
- `artifacts/t12_cmu_orm/fresh-validation/raw-majority-predictions.jsonl`: raw baseline predictions.
- `data/canonical/train.csv`: gold answers, joined only after the label-blind freeze.
- `data/cmu_orm/validation.csv`: frozen folds and hard/format strata.
- `data/cmu_orm/train-manifest.json`: local corpus size and 1:1 balance.
- `artifacts/t12_cmu_orm/reused-t8-diagnostic.json`: non-primary corroborating replay.
- `analyze.py`: deterministic transformation producing `diagnostic-summary.json`.
- `diagnostic-summary.sql`: SQLite-compatible materialization of the exact bounded widget rows; this is a portable-report provenance adapter, while `analyze.py` remains the primary computation.

## Chart map

1. `accuracy_comparison`: Comparison & Ranking / vertical bar. Fields: method, accuracy. Claim: weighted ORM stayed next to majority while the oracle ceiling remained 9.5pp higher. Single-root palette; category labels provide non-color identity.
2. `baseline_error_decomposition`: Comparison & Ranking / vertical bar. Fields: category, questions. Claim: only 8 of 95 selectable raw-majority errors were rescued. Single-root palette; exact values remain available in the semantic fallback.

No trend chart is used because the evidence is one frozen evaluation, not a time series. Exact mechanism and subgroup values use tables because lookup and auditability matter more than shape.

## Validation and caveats

- The global ROC-AUC is reproduced from the frozen evaluation. The question-local macro AUC is recomputed only on the 509 questions containing both correct and incorrect valid candidates.
- Oracle pass@32 means the gold integer appeared in at least one valid answer group; it is a selector ceiling, not a deployable method.
- Hard and format strata overlap, so their net-correct changes are not additive.
- The 42/45 mechanism split describes the 87 baseline errors where a gold group existed but weighted ORM remained wrong. `support lost` means the gold group's geometric-mean score was higher, yet `n × geometric mean` still selected another answer.
- Causal claims about 3B LoRA capacity, class-prior shift, and negative diversity are labeled as likely explanations rather than proven causes.
- Post-hoc diagnostics cannot alter the T12 HOLD decision. Formula, calibration, or training changes require a new preregistered task and fresh validation.
