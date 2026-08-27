# T8 pass@32–plurality gap: source and method notes

## Scope

- Frozen evaluation union: 3,737 questions.
- Candidate pool: 32 generations per question, 119,584 generations total.
- Model: `Qwen/Qwen2.5-3B-Instruct`, revision recorded in the T8 manifest.
- Primary source: `artifacts/t8_self_consistency/generations.jsonl`.
- Labels: `data/canonical/train.csv`.
- The four diagnostic splits overlap. Split counts must not be added together.

## Metric definitions

- `pass@32`: at least one of the 32 extracted answers exactly equals the stored label.
- `plurality accuracy`: choose the valid extracted answer with the largest vote count. If the largest count is tied, choose the answer that appeared first in sample-index order. The repository and prior records call this `majority@32`, but the implemented rule is technically plurality.
- `selection failure`: `pass@32=true` and the plurality-selected answer is wrong.
- `oracle failure`: none of the 32 extracted answers equals the label.
- `selection gap`: `pass@32 − plurality accuracy`. It is an oracle upper bound because computing it requires the ground-truth label.
- `correct-answer rank`: one plus the number of distinct answer candidates with strictly more votes than the correct answer. A correct answer tied at the top is rank 1.

Ground truth is never passed to extraction, vote counting, tie-breaking, or the T8-3 filter. It is attached only after each prediction is frozen.

## Reproduction and integrity checks

`analyze.py` streams all 119,584 rows, calls the repository production extractor in `src/extract.py`, sorts the 32 candidates by sample index, and reproduces:

- plurality correct: 2,590 / 3,737 = 69.3069%;
- pass@32: 3,154 / 3,737 = 84.3993%;
- selection failures: 564 / 3,737 = 15.0923%p;
- no-correct-candidate failures: 583 / 3,737 = 15.6007%.

The reproduced pass and plurality values are asserted against `artifacts/t8_self_consistency/sweep.json`. The canonical label file and four split files match the frozen manifest byte-for-byte. The current generations, sweep, and union-ID files use CRLF line endings; normalizing CRLF to LF reproduces the manifest hashes exactly. This is a newline-format difference, not a content difference.

## T8-3 paired comparison

`artifacts/t8_3_vote_filter/holdout/predictions.jsonl` contains predictions made from the same frozen 32-candidate pool. Its label-blind rule removes weak extraction paths and selected hit-max cases, with an unfiltered fallback if everything is removed. Relative to base plurality it:

- rescues 69 base-wrong questions;
- breaks 14 base-correct questions;
- nets +55 questions, or +1.4718%p;
- recovers 69 / 564 = 12.23% of the oracle selection gap.

This is empirical evidence for the recoverable value of a simple output-quality policy. It is not evidence that the remaining gap is impossible to recover with a semantic verifier.

## Example verification

The visible examples were selected to separate mechanisms, not to estimate their population prevalence. `train-003015`, `train-007230`, `train-008043`, `train-008317`, and `train-013974` were already recorded as manually verified genuine cases in `artifacts/t8_6_base_vote_policy/train_error_analysis/verified_genuine_error_examples.json`. `train-003341` and `train-012155` were checked directly against their question text and all relevant raw T8 outputs for this report.

The full 3,737 labels were not manually audited. Therefore the detailed cases should be read as verified illustrations, while population counts remain exact with respect to the stored labels.

## Chart contracts

### Outcome decomposition

- Decision: determine how much of the current error budget is selection versus generation.
- Comparison: plurality correct, correct generated but wrongly selected, and no correct generation.
- Dimensions: outcome; measure: questions/share of all 3,737.
- Chart: horizontal bars for exact category comparison.
- Ordering: outcome decomposition order, with direct labels.

### Correct support among selection failures

- Decision: assess whether a small tie-break or reranker can plausibly recover the gap.
- Comparison: correct-answer vote bands among the 564 selection failures.
- Dimensions: support band; measure: number/share of selection failures.
- Chart: horizontal bars, semantically ordered from 1–2 to 9–16 votes.

### Winning margin among selection failures

- Decision: distinguish near-misses from strongly correlated wrong modes.
- Comparison: winning wrong vote count minus correct vote count.
- Dimensions: margin band; measure: questions/share of selection failures.
- Chart: horizontal bars, ordered from tie to nine-or-more-vote deficit.

### Problem-type selection gap

- Decision: prioritize where a semantic selector or targeted SFT should focus.
- Comparison: selection-gap percentage points by deterministic repository classifier.
- Dimensions: problem type; measure: selection gap; tooltip: count and share of all selection failures.
- Chart: horizontal bars sorted by gap magnitude.

## Interpretation guardrails

- The 15.09%p gap is a gold-label oracle ceiling, not a forecast of achievable improvement.
- Only 29 questions (0.78%p) have the correct answer tied for the top vote count and can be recovered by perfect tie-breaking alone.
- A perfect semantic chooser over the top two vote-ranked answers would reach 76.85%, but this assumes label-like verification capability.
- Candidate samples are correlated because they come from the same model and prompt family. Vote counts should not be treated as independent Bernoulli evidence.
- Do not train on these 3,737 evaluation examples. Use their failure patterns to mine analogous examples from a disjoint training pool, then evaluate once on the frozen union and preferably a fresh holdout.
