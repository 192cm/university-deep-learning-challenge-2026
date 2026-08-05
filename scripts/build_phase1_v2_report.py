#!/usr/bin/env python3
"""Build the evidence-backed Phase 1 v2 Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase1_common import atomic_write_text, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-comparison", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--historical-metrics", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--test-summary", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    comparison = json.loads(args.split_comparison.read_text(encoding="utf-8"))
    environment = json.loads(args.environment.read_text(encoding="utf-8"))
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    historical = json.loads(args.historical_metrics.read_text(encoding="utf-8"))
    verification = json.loads(args.verification.read_text(encoding="utf-8"))
    test_summary = json.loads(args.test_summary.read_text(encoding="utf-8"))

    metric_lines = []
    for row in metrics["metrics"]:
        metric_lines.append(
            "| {baseline_id} | {scope} | {questions} | {greedy} | {sample} | {passk} | "
            "{majority} | {agreement} | {invalid} | {tokens} | {latency} | {runtime} |".format(
                baseline_id=row["baseline_id"],
                scope=row["scope"],
                questions=row["questions"],
                greedy=fmt(row["greedy_accuracy"]),
                sample=fmt(row["sample_accuracy"]),
                passk=fmt(row["pass_at_k"]),
                majority=fmt(row["majority_at_k"]),
                agreement=fmt(row["agreement_at_k"]),
                invalid=fmt(row["invalid_output_rate"]),
                tokens=fmt(row["median_output_tokens"], 1),
                latency=fmt(row["p95_latency_seconds"], 3),
                runtime=fmt(row["estimated_1000_question_runtime_seconds"] / 3600.0, 3),
            )
        )

    historical_by_key = {
        (row["baseline_id"], row["scope"]): row for row in historical["metrics"]
    }
    delta_lines = []
    for row in metrics["metrics"]:
        old = historical_by_key[(row["baseline_id"], row["scope"])]
        score_key = "sample_accuracy" if row["greedy_accuracy"] is None else "greedy_accuracy"
        delta_lines.append(
            f"| {row['baseline_id']} | {row['scope']} | `{score_key}` | "
            f"{fmt(old[score_key])} | {fmt(row[score_key])} | "
            f"{fmt(row[score_key] - old[score_key])} | "
            f"{fmt(row['invalid_output_rate'] - old['invalid_output_rate'])} |"
        )

    comparison_lines = [
        f"| {row['id_file']} | {row['historical_count']} | {row['current_count']} | "
        f"{row['common_count']} | {row['added_count']} | {row['removed_count']} | "
        f"{row['symmetric_difference_count']} |"
        for row in comparison["rows"]
    ]
    split_hash_lines = [
        f"| {name} | {asset['sha256']} | {asset['bytes']} |"
        for name, asset in sorted(split["outputs"].items())
    ]
    verification_lines = [
        f"| {item['name']} | {'PASS' if item['passed'] else 'FAIL'} |"
        for item in verification["checks"]
    ]
    package_lines = [
        f"| {name} | {version} |" for name, version in sorted(environment["packages"].items())
    ]
    baseline_lines = []
    for name, baseline in config["baselines"].items():
        generation = baseline["generation"]
        baseline_lines.append(
            f"| {name} | {str(baseline['do_sample']).lower()} | `{baseline['seeds']}` | "
            f"{generation['max_new_tokens']} | {fmt(generation['temperature'])} | "
            f"{fmt(generation['top_p'])} |"
        )

    model = config["model"]
    counts = provenance["counts"]
    gpu = environment["gpu"]["nvidia_smi_query"]["stdout"].strip().splitlines()[0]
    text = f"""# Phase 1 v2 evaluation on the final_v1 canonical dataset

## Conclusion

The `{config['data']['train_filtered']['path']}` canonical dataset ({counts['train_filtered']:,} rows)
was used to reproduce `{split['split_version']}` splits and the B0/B1/B2 base-model baselines
in a fresh remote workspace. Provenance and final Phase 1 verification are
`{'PASS' if provenance['passed'] else 'FAIL'}` and
`{'PASS' if verification['passed'] else 'FAIL'}`, respectively. Historical generations from
the prior 16,528-row dataset were used only for the explicit comparison below and were not
recycled into this evaluation.

## Canonical data and provenance

| Item | Value |
|---|---|
| canonical path | `{config['data']['train_filtered']['path']}` |
| canonical rows | {counts['train_filtered']} |
| canonical SHA-256 | `{config['data']['train_filtered']['sha256']}` |
| source rows | {counts['train']} |
| organizer exclusions | {counts['organizer_exclusions']} |
| supplemental exclusions | {counts['supplemental_exclusions']} |
| exclusion union / overlap | {counts['exclusion_union']} / {counts['exclusion_overlap']} |
| unexpected additional removals | {counts['unexpected_additional_removals']} |
| canonical manifest SHA-256 | `{config['data']['train_filter_manifest']['sha256']}` |
| provenance report SHA-256 | `{sha256_file(args.provenance)}` |

## Splits

The Random and Template-group validation sets contain
{split['splits']['random']['validation_rows']} and
{split['splits']['template']['validation_rows']} rows. The Hard and Format diagnostics contain
{split['splits']['hard_diagnostic']['rows']} and
{split['splits']['format_diagnostic']['rows']} rows.

| split file | SHA-256 | bytes |
|---|---|---:|
{chr(10).join(split_hash_lines)}

### ID changes relative to historical phase1_v1

| ID file | old | new | common | added | removed | symmetric diff |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(comparison_lines)}

Comparison CSV: `{comparison['output_csv']['path']}` (`{comparison['output_csv']['sha256']}`)

## Environment and pinned model

| Item | Value |
|---|---|
| GPU | `{gpu}` |
| Python | `{environment['python']['version'].split()[0]}` |
| model / revision | `{model['id']}` / `{model['revision']}` |
| tokenizer revision | `{model['tokenizer_revision']}` |
| dtype | `{model['dtype']}` |
| local-files-only generation | `true` |

| package | version |
|---|---|
{chr(10).join(package_lines)}

## Baseline contract

| baseline | sampling | seeds | max new tokens | temperature | top-p |
|---|---|---|---:|---:|---:|
{chr(10).join(baseline_lines)}

B3 was not part of the completed historical Phase 1 implementation, so the reproducible
comparison scope remains B0/B1/B2.

## Phase 1 v2 metrics

| baseline | scope | questions | greedy | sample | pass@k | majority@k | agreement@k | invalid | median tokens | p95 latency(s) | est. 1,000q(h) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(metric_lines)}

## Change relative to the historical 16,528-row evaluation

| baseline | scope | score | old | new | delta | invalid-rate delta |
|---|---|---|---:|---:|---:|---:|
{chr(10).join(delta_lines)}

## Verification

| check | result |
|---|---|
{chr(10).join(verification_lines)}

Independent greedy and seeded-sampling reproduction runs were executed in separate output
directories. The verification artifact records exact text/answer agreement and accuracy tolerance.

## Test-suite result

The full repository suite ran {test_summary['full_suite']['tests_run']} tests:
{test_summary['full_suite']['passed']} passed and {test_summary['full_suite']['failed']} failed.
The single failure was `{test_summary['environment_deviation']['test']}` because the historical
Phase 0 image expected PyTorch `{test_summary['environment_deviation']['expected_torch']}`, while
this fresh server supplied and the experiment recorded `{test_summary['environment_deviation']['actual_torch']}`.
The focused suite excluding that image-specific test module ran
{test_summary['focused_suite']['tests_run']} tests and
`{'PASS' if test_summary['focused_suite']['passed'] else 'FAIL'}`.

## Compliance and limitations

- Generation loaded only `Qwen/Qwen2.5-3B-Instruct` at the pinned revision in offline mode.
- Candidate selection uses extracted model text and vote counts only, with early stopping only
  after model generation finishes.
- No Python/SymPy/solver/calculating verifier/external API/dynamic retrieval is used at inference.
- The remote workspace and files were retained; no recycle or destroy action was issued.
- The PyTorch/CUDA image differs from the historical Phase 0 image. This is recorded as an
  environment deviation; results should not be interpreted as a one-variable data-only ablation.
- Artifact directory: `{args.artifact_dir.as_posix()}`
"""
    atomic_write_text(args.output, text)
    print(args.output.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
