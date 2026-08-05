# AGENTS.md

## Scope

These instructions apply to the entire repository. Future agents must follow them before reading or changing project files.

## Project objective

Build a reproducible competition system that fine-tunes only `Qwen/Qwen2.5-3B-Instruct` to solve unseen math problems and submit exact final answers. Optimize for Private Test accuracy while preserving competition compliance, generalization, runtime safety, and reproducibility.

## Start every task here

1. Read this file completely.
2. Read `docs/information/rules.md` completely.
3. Read `docs/strategy/roadmap.md` and identify the current stage and exit gate.
4. Read the relevant part of `docs/strategy/winning-strategy.md`.
5. Run `git status --short` and preserve all unrelated user changes and untracked files.
6. Inspect the current implementation and artifacts before proposing or applying changes.

Do not redo completed work. Do not mark a roadmap item complete without an artifact, test result, or reproducible log.

## Source-of-truth order

When instructions conflict, use this order:

1. Latest Kaggle foundational and competition-specific rules
2. Dated organizer announcements and Discord clarifications
3. `docs/information/rules.md`
4. `docs/strategy/roadmap.md`
5. `docs/strategy/winning-strategy.md`
6. Experiment notes and informal discussion

If a new official clarification changes the interpretation of a rule, stop any conflicting implementation, record the date and source in `docs/information/rules.md`, then update the roadmap and strategy.

## Non-negotiable competition constraints

### Model

- Use only `Qwen/Qwen2.5-3B-Instruct` as the base model.
- Do not load, merge, ensemble, or derive inference weights from another base model.
- Pin the exact model and tokenizer revisions in every reproducible experiment.
- Training methods such as full fine-tuning, LoRA/QLoRA, SFT, DPO, GRPO and quantization are allowed only within the latest official rules.

### Test-time inference

The final answer must be derived from model outputs only.

Never implement or use the following on leaderboard or final-test questions:

- execution of model-generated Python or other code;
- Tool-Integrated Reasoning or Program-of-Thought with execution feedback;
- Python, SymPy, SAT/SMT, numerical solvers, calculators, or participant-written math utilities;
- deterministic mathematical verifiers that calculate, correct, or rerank candidate answers;
- feeding external calculation results back into the model;
- internet access, web search, external APIs, or external models;
- per-question dynamic BM25 or embedding retrieval unless a later organizer clarification explicitly permits it.

The following are allowed under the current documented interpretation:

- multiple generations from the competition model;
- Majority Voting and Self-Consistency;
- adaptive sampling and early stopping based only on extracted model answers, vote counts, elapsed time, and remaining time;
- model-output-only Best-of-N, subject to the documented rules;
- syntactic final-answer extraction and normalization;
- non-mathematical submission checks such as schema, row count, IDs, duplicates, missing values, and file integrity.

Answer extraction may read what the model wrote. It must not solve the problem, perform new arithmetic, or repair an answer mathematically.

### Training-time tools and external data

- Code and external teacher APIs may be used only for permitted training-data creation, filtering, and verification.
- Never send leaderboard or final-test questions to an API, search engine, external model, or external service.
- Strip tool calls and execution feedback from final training targets; train tool-free natural-language reasoning that can complete at inference without tools.
- Use only public, free, equally accessible external data with compatible usage rights.
- Record dataset name, exact revision, URL, license, retrieval date, filtering steps, and hashes.
- Decontaminate every external dataset against all provided evaluation questions using exact, normalized-template, and near-duplicate checks.

## Data protection and provenance

Treat user-provided and competition data as immutable source assets.

Do not overwrite these files:

- `data/deep_chal_math_train.csv`
- `data/deep_chal_math_leaderboard.csv`

Derived files must be reproducible from a script and must document:

- source path and SHA-256;
- generation timestamp;
- script and configuration;
- input and output row counts;
- removed or transformed IDs and reasons;
- output SHA-256.

Preserve existing filtered datasets and audit tables unless the user explicitly requests regeneration. If regeneration is needed, verify the resolved paths and write to a new or clearly versioned derived file before replacing anything.

Use UTF-8 for text and CSV files. Preserve IDs exactly. Never infer or fabricate labels for leaderboard or final-test rows.

## Repository organization

- `docs/information/`: sourced competition facts, rules, data, evaluation, and schedule
- `docs/strategy/`: strategy, roadmap, decisions, and runbooks
- `scripts/`: canonical reproducible data and artifact generation code
- `data/`: immutable competition inputs and documented derived datasets
- `report/`: analytical reports and executed QA notebooks
- `configs/`: create only when real training or inference configurations exist
- `artifacts/experiments/`: create only when experiment registry artifacts exist
- `tests/`: create only when executable tests exist

Notebooks may analyze and present results, but reusable logic must live in scripts or importable modules. Do not make a notebook the sole implementation of a critical pipeline.

Do not commit model weights, caches, downloaded package archives, credentials, API keys, or large generated artifacts unless the user explicitly requests it and repository policy permits it.

## Experiment contract

Every model experiment must record at least:

- experiment ID and objective;
- code commit or exact working-tree diff;
- model and tokenizer revision;
- dataset paths, revisions, and SHA-256 values;
- split version;
- complete training and inference configuration;
- random seeds;
- hardware and library versions;
- wall-clock runtime;
- checkpoint and output paths;
- all required metrics;
- keep/reject decision and evidence.

Required model-selection metrics:

- greedy accuracy;
- average sample accuracy;
- pass@k and majority@k;
- answer agreement;
- invalid-output rate;
- median output tokens;
- p95 per-question latency;
- estimated total final-test runtime;
- separate Random and Template-group holdout results.

Change one primary variable per ablation. Save raw generations so regressions can be investigated. Do not select a model from Public Leaderboard score alone.

## Evaluation and decision rules

- If pass@k is low, improve data and SFT before increasing test-time samples.
- If pass@k is high but majority@k is low, calibrate sampling or test a model-output-only selector.
- If Random holdout improves while Template holdout degrades, investigate duplication and template overfitting.
- If DPO shortens outputs but materially reduces accuracy, reject it and return to the SFT checkpoint.
- If quantization improves speed but causes unacceptable accuracy loss, use higher precision.
- Reserve at least 30% of the final 24-hour window for validation, restart, failure recovery, and submission.
- Stop and ask the organizers when a proposed inference feature has ambiguous rule status.

## Implementation standards

- Search with `rg` or `rg --files` first.
- Use `apply_patch` for hand-authored edits.
- Preserve unrelated work in a dirty worktree.
- Prefer `pathlib`, explicit encodings, typed functions, and deterministic seeds in Python.
- Add command-line arguments for reusable scripts instead of hard-coded local paths.
- Validate input columns, ID uniqueness, row counts, and output paths before writing.
- Keep source-to-output transformations idempotent where practical.
- Fail loudly on schema mismatch, missing required files, duplicate IDs, or unexpected answer formats.
- Do not install dependencies or access the network unless the task requires it and authorization is available.
- Do not delete or move material files without explicit scope, resolved-path verification, and user authorization.

## Verification before handoff

For documentation-only changes:

- run `git diff --check`;
- verify relative links and referenced paths;
- search for statements that conflict with the latest rules.

For data-pipeline changes:

- run the transformation on a temporary or versioned output first;
- verify source hashes and source immutability;
- check schema, row counts, unique IDs, and audit completeness;
- rerun to confirm deterministic output when expected.

For model or inference changes:

- run focused tests or a representative smoke evaluation;
- confirm no prohibited test-time tool execution or retrieval exists;
- report accuracy, parse failures, runtime, hardware, and seeds;
- verify offline model loading and restart behavior;
- keep a known-good fallback checkpoint and configuration.

## Final-test guardrail

On 2026-08-31, agents must follow the frozen runbook and may make only schema compatibility fixes or clearly necessary reliability fixes. Do not perform new model research on final-test questions. Do not inspect them for manual solving, query external services, generate new training labels, or build question-specific tools.

Before writing `submission.csv`, verify the official sample submission. Before submission, verify row count, exact IDs, uniqueness, missing values, answer format, generation timestamp, and file hash.

## Documentation maintenance

- Update `docs/strategy/roadmap.md` checkboxes only with evidence.
- Record important failed experiments, not only successful ones.
- Keep competition facts in `docs/information/` and recommendations in `docs/strategy/`.
- Cite external strategies using primary sources such as official repositories, papers, model cards, or competition write-ups.
- State clearly when a recommendation is an inference rather than an official rule.

## Git behavior

- Do not stage, commit, branch, push, or open a pull request unless explicitly requested.
- Never use destructive Git commands to discard user work.
- If asked to commit, inspect the actual diff first and use a Conventional Commit message.
- Report unrelated modified or untracked files without changing them.

## Definition of done

A task is complete only when the requested artifact exists, relevant checks pass, results and limitations are documented, competition compliance is preserved, and no required work remains hidden behind an unverified assumption.

