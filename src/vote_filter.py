#!/usr/bin/env python3
"""Reaggregate immutable T8 pools with the frozen T8-3 vote-quality filter."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

if __package__:
    from .evaluate import Generation, Label, evaluate, load_generations, load_labels
    from .self_consistency import exact_mcnemar, group_generations
    from .submit import (
        LOW_QUALITY_VOTE_POLICY,
        build_submission_payload,
        low_quality_vote_reasons,
        select_majority_vote,
    )
else:
    from evaluate import Generation, Label, evaluate, load_generations, load_labels  # type: ignore[no-redef]
    from self_consistency import exact_mcnemar, group_generations  # type: ignore[no-redef]
    from submit import (  # type: ignore[no-redef]
        LOW_QUALITY_VOTE_POLICY,
        build_submission_payload,
        low_quality_vote_reasons,
        select_majority_vote,
    )


EXPECTED_MODEL = "Qwen/Qwen2.5-3B-Instruct"
EXPECTED_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
EXPECTED_SPLITS = (
    "random_holdout",
    "template_holdout",
    "hard_diagnostic",
    "format_diagnostic",
)
POLICY_NAME = "drop-low-quality-votes-v1"
REFERENCE_POLICY_NAME = "unfiltered_majority_k32"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)


def file_record(path: Path, *, rows: int | None = None) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"Required file is missing: {path}")
    record: dict[str, object] = {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        record["rows"] = rows
    return record


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def nested_dict(value: Mapping[str, object], key: str) -> dict[str, object]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise ValueError(f"Expected object field {key!r}")
    return dict(result)


def load_ids(path: Path) -> list[str]:
    ids = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"ID file is empty or contains duplicates: {path}")
    return ids


def validate_config(path: Path) -> dict[str, object]:
    config = load_json(path)
    if config.get("task") != "T8-3" or config.get("policy_name") != POLICY_NAME:
        raise ValueError("Config must identify the frozen T8-3 policy")
    if config.get("vote_filter") != LOW_QUALITY_VOTE_POLICY:
        raise ValueError("Config vote filter differs from the byte-frozen implementation")
    contract = nested_dict(config, "generation_contract")
    expected = {
        "k": 32,
        "new_generations": 0,
        "model_id": EXPECTED_MODEL,
        "model_revision": EXPECTED_REVISION,
        "tokenizer_revision": EXPECTED_REVISION,
        "adapter": None,
    }
    if contract != expected:
        raise ValueError("T8-3 generation contract changed")
    cross_validation = nested_dict(config, "cross_validation")
    if (
        int(cross_validation.get("folds", -1)) != 5
        or cross_validation.get("fold_hash_prefix") != "t8-vote-cv-v1:"
        or cross_validation.get("candidate_policies")
        != [REFERENCE_POLICY_NAME, POLICY_NAME]
    ):
        raise ValueError("T8-3 cross-validation contract changed")
    return config


def protected_snapshot(config: Mapping[str, object]) -> dict[str, object]:
    protected = nested_dict(config, "protected_inputs")
    paths: dict[str, Path] = {}
    for raw_path in protected.get("files", []):
        path = Path(str(raw_path))
        if not path.is_file():
            raise ValueError(f"Protected input is missing: {path}")
        paths[path.as_posix()] = path
    for raw_root in protected.get("trees", []):
        root = Path(str(raw_root))
        if not root.is_dir():
            raise ValueError(f"Protected artifact tree is missing: {root}")
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            paths[path.as_posix()] = path
    return {
        "files": {
            name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for name, path in sorted(paths.items())
        },
        "file_count": len(paths),
        "total_bytes": sum(path.stat().st_size for path in paths.values()),
    }


def verify_protected_snapshot(snapshot: Mapping[str, object]) -> dict[str, object]:
    raw_files = snapshot.get("files")
    if not isinstance(raw_files, dict) or not raw_files:
        raise ValueError("Protected snapshot contains no files")
    mismatches: list[dict[str, object]] = []
    for raw_path, raw_record in raw_files.items():
        path = Path(str(raw_path))
        record = raw_record if isinstance(raw_record, Mapping) else {}
        if not path.is_file():
            mismatches.append({"path": str(raw_path), "reason": "missing"})
            continue
        current_size = path.stat().st_size
        current_hash = sha256_file(path)
        if current_size != int(record.get("bytes", -1)):
            mismatches.append({"path": str(raw_path), "reason": "bytes_changed"})
        elif current_hash != record.get("sha256"):
            mismatches.append({"path": str(raw_path), "reason": "sha256_changed"})
    return {
        "verified": not mismatches,
        "file_count": len(raw_files),
        "mismatches": mismatches,
    }


def validate_generation_metadata(
    metadata_path: Path,
    generations_path: Path,
    *,
    expected_selected_rows: int | None = None,
) -> dict[str, object]:
    metadata = load_json(metadata_path)
    if metadata.get("status") != "complete" or metadata.get("task") != "T8":
        raise ValueError(f"Expected complete T8 metadata: {metadata_path}")
    effective = nested_dict(metadata, "effective_config")
    model = nested_dict(effective, "model")
    if (
        model.get("id") != EXPECTED_MODEL
        or model.get("revision") != EXPECTED_REVISION
        or model.get("tokenizer_revision") != EXPECTED_REVISION
        or effective.get("adapter") is not None
    ):
        raise ValueError(f"Unexpected T8 model identity: {metadata_path}")
    generation = nested_dict(effective, "generation")
    if int(generation.get("n", -1)) != 32:
        raise ValueError(f"Expected immutable k=32 pool: {metadata_path}")
    output = nested_dict(metadata, "output")
    if output.get("sha256") != sha256_file(generations_path):
        raise ValueError(f"Generation SHA-256 differs from metadata: {generations_path}")
    sources = nested_dict(metadata, "sources")
    if expected_selected_rows is not None and int(sources.get("selected_rows", -1)) != expected_selected_rows:
        raise ValueError(f"Unexpected selected row count: {metadata_path}")
    return metadata


def ensure_coverage(
    grouped: Mapping[str, Sequence[Generation]], ids: Sequence[str], *, k: int
) -> None:
    if set(grouped) != set(ids):
        raise ValueError("Holdout generation ID coverage differs from frozen union")
    for row_id in ids:
        indices = [candidate.sample_index for candidate in grouped[row_id]]
        if indices != list(range(k)):
            raise ValueError(f"Incomplete k={k} generation group for {row_id}")


def build_policy_predictions(
    grouped: Mapping[str, Sequence[Generation]], ids: Sequence[str]
) -> tuple[
    dict[str, str | None],
    dict[str, str | None],
    dict[str, list[Generation]],
    list[dict[str, object]],
    dict[str, object],
]:
    reference_predictions: dict[str, str | None] = {}
    filtered_predictions: dict[str, str | None] = {}
    filtered_selection: dict[str, list[Generation]] = {}
    rows: list[dict[str, object]] = []
    condition_candidates: Counter[str] = Counter()
    condition_votes: Counter[str] = Counter()
    path_counts: Counter[str] = Counter()
    filtered_candidate_count = 0
    removed_vote_count = 0
    fallback_ids: list[str] = []
    changed_ids: list[str] = []

    for row_id in ids:
        candidates = list(grouped[row_id])
        extractions = [candidate.extraction for candidate in candidates]
        hit_flags = [candidate.hit_max_new_tokens for candidate in candidates]
        selection = select_majority_vote(
            extractions,
            hit_flags,
            filter_low_quality_votes=True,
        )
        unfiltered_vote = selection["unfiltered_vote"]
        filtered_vote = selection["filtered_vote_before_fallback"]
        assert isinstance(unfiltered_vote, Mapping)
        assert isinstance(filtered_vote, Mapping)
        reasons_by_candidate = selection["filter_reasons"]
        assert isinstance(reasons_by_candidate, list)

        kept: list[Generation] = []
        removed_indices: list[int] = []
        reason_indices: defaultdict[str, list[int]] = defaultdict(list)
        for candidate, reasons in zip(candidates, reasons_by_candidate, strict=True):
            assert isinstance(reasons, tuple)
            path_counts[candidate.extraction.path] += 1
            if reasons:
                filtered_candidate_count += 1
            else:
                kept.append(candidate)
            for reason in reasons:
                condition_candidates[reason] += 1
                reason_indices[reason].append(candidate.sample_index)
            if reasons and candidate.extraction.answer is not None:
                removed_vote_count += 1
                removed_indices.append(candidate.sample_index)
                for reason in reasons:
                    condition_votes[reason] += 1

        fallback = bool(selection["fallback_to_unfiltered"]) or not kept
        if fallback:
            fallback_ids.append(row_id)
            filtered_selection[row_id] = candidates
        else:
            filtered_selection[row_id] = kept
        reference_answer = unfiltered_vote["answer"]
        filtered_answer = selection["answer"]
        reference_predictions[row_id] = (
            None if reference_answer is None else str(reference_answer)
        )
        filtered_predictions[row_id] = (
            None if filtered_answer is None else str(filtered_answer)
        )
        if reference_answer != filtered_answer:
            changed_ids.append(row_id)
        rows.append(
            {
                "id": row_id,
                "unfiltered_answer": reference_answer,
                "filtered_answer": filtered_answer,
                "unfiltered_vote_counts": unfiltered_vote["vote_counts"],
                "filtered_vote_counts_before_fallback": filtered_vote["vote_counts"],
                "unfiltered_agreement": unfiltered_vote["agreement"],
                "filtered_agreement_before_fallback": filtered_vote["agreement"],
                "removed_sample_indices": removed_indices,
                "condition_sample_indices": dict(sorted(reason_indices.items())),
                "fallback_to_unfiltered": fallback,
                "prediction_frozen_without_ground_truth": True,
            }
        )

    diagnostics: dict[str, object] = {
        "questions": len(ids),
        "generations": sum(len(grouped[row_id]) for row_id in ids),
        "condition_candidate_counts": dict(sorted(condition_candidates.items())),
        "condition_removed_vote_counts": dict(sorted(condition_votes.items())),
        "condition_candidate_count_unique": filtered_candidate_count,
        "removed_vote_count_unique": removed_vote_count,
        "extraction_path_counts": dict(sorted(path_counts.items())),
        "fallback_count": len(fallback_ids),
        "fallback_ids": fallback_ids,
        "changed_answer_count": len(changed_ids),
        "changed_answer_ids": changed_ids,
        "ground_truth_consumed": False,
    }
    return (
        reference_predictions,
        filtered_predictions,
        filtered_selection,
        rows,
        diagnostics,
    )


def flatten_selection(
    selected: Mapping[str, Sequence[Generation]], ids: Sequence[str]
) -> list[Generation]:
    return [candidate for row_id in ids for candidate in selected[row_id]]


def accuracy(
    predictions: Mapping[str, str | None], labels: Mapping[str, Label], ids: Sequence[str]
) -> dict[str, object]:
    correct = sum(predictions[row_id] == labels[row_id].answer for row_id in ids)
    invalid = sum(predictions[row_id] is None for row_id in ids)
    return {
        "questions": len(ids),
        "correct": correct,
        "accuracy": correct / len(ids),
        "invalid_predictions": invalid,
        "invalid_prediction_rate": invalid / len(ids),
    }


def fold_for_group(group: str, *, prefix: str, folds: int) -> int:
    digest = hashlib.sha256(f"{prefix}{group}".encode("utf-8")).hexdigest()
    return int(digest, 16) % folds


def load_template_groups(path: Path, ids: Sequence[str]) -> dict[str, str]:
    wanted = set(ids)
    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Template audit has no header: {path}")
        for row in reader:
            row_id = str(row.get("id", "")).strip()
            if row_id not in wanted:
                continue
            group = str(row.get("template_group_id", "")).strip()
            if not group:
                raise ValueError(f"Missing template group for {row_id}")
            result[row_id] = group
    if set(result) != wanted:
        raise ValueError("Template audit does not cover the holdout union")
    return result


def cross_validate(
    *,
    reference_predictions: Mapping[str, str | None],
    filtered_predictions: Mapping[str, str | None],
    labels: Mapping[str, Label],
    ids: Sequence[str],
    groups: Mapping[str, str],
    config: Mapping[str, object],
) -> dict[str, object]:
    cv_config = nested_dict(config, "cross_validation")
    fold_count = int(cv_config["folds"])
    prefix = str(cv_config["fold_hash_prefix"])
    fold_ids: dict[int, list[str]] = {fold: [] for fold in range(fold_count)}
    for row_id in ids:
        fold_ids[fold_for_group(groups[row_id], prefix=prefix, folds=fold_count)].append(
            row_id
        )
    if any(not values for values in fold_ids.values()):
        raise ValueError("Cross-validation produced an empty fold")

    oof_predictions: dict[str, str | None] = {}
    fold_reports: list[dict[str, object]] = []
    all_ids = set(ids)
    for fold in range(fold_count):
        validation_ids = fold_ids[fold]
        training_ids = [row_id for row_id in ids if row_id not in set(validation_ids)]
        reference_train = accuracy(reference_predictions, labels, training_ids)
        filtered_train = accuracy(filtered_predictions, labels, training_ids)
        if float(filtered_train["accuracy"]) > float(reference_train["accuracy"]):
            selected_policy = POLICY_NAME
            selected_predictions = filtered_predictions
        else:
            selected_policy = REFERENCE_POLICY_NAME
            selected_predictions = reference_predictions
        for row_id in validation_ids:
            if row_id in oof_predictions:
                raise AssertionError("OOF prediction assigned twice")
            oof_predictions[row_id] = selected_predictions[row_id]
        validation_comparison = exact_mcnemar(
            selected_predictions,
            reference_predictions,
            labels,
            validation_ids,
        )
        frozen_validation = exact_mcnemar(
            filtered_predictions,
            reference_predictions,
            labels,
            validation_ids,
        )
        fold_reports.append(
            {
                "fold": fold,
                "training_questions": len(training_ids),
                "validation_questions": len(validation_ids),
                "template_groups": len({groups[row_id] for row_id in validation_ids}),
                "training_accuracy": {
                    REFERENCE_POLICY_NAME: reference_train["accuracy"],
                    POLICY_NAME: filtered_train["accuracy"],
                },
                "selected_policy": selected_policy,
                "validation_selected_vs_unfiltered": validation_comparison,
                "validation_frozen_filter_vs_unfiltered": frozen_validation,
            }
        )
    if set(oof_predictions) != all_ids:
        raise AssertionError("OOF predictions do not cover the holdout union")
    oof_comparison = exact_mcnemar(
        oof_predictions,
        reference_predictions,
        labels,
        ids,
    )
    return {
        "schema_version": 1,
        "task": "T8-3",
        "method": {
            "grouping": "template_group_id",
            "fold_assignment": "sha256('t8-vote-cv-v1:' + group) interpreted as an integer modulo 5",
            "candidate_policies": [REFERENCE_POLICY_NAME, POLICY_NAME],
            "selection": "higher training-fold union accuracy; exact tie chooses unfiltered",
        },
        "folds": fold_reports,
        "all_folds_selected_frozen_policy": all(
            report["selected_policy"] == POLICY_NAME for report in fold_reports
        ),
        "all_frozen_validation_deltas_positive": all(
            float(
                nested_dict(report, "validation_frozen_filter_vs_unfiltered")[
                    "delta_pp"
                ]
            )
            > 0
            for report in fold_reports
        ),
        "out_of_fold": oof_comparison,
    }


def support_band(top_count: int) -> str:
    if top_count <= 2:
        return "01_02"
    if top_count <= 4:
        return "03_04"
    if top_count <= 8:
        return "05_08"
    if top_count <= 16:
        return "09_16"
    if top_count <= 24:
        return "17_24"
    return "25_32"


def posthoc_diagnostics(
    *,
    prediction_rows: Sequence[Mapping[str, object]],
    grouped: Mapping[str, Sequence[Generation]],
    labels: Mapping[str, Label],
) -> dict[str, object]:
    path_total: Counter[str] = Counter()
    path_correct: Counter[str] = Counter()
    condition_total: Counter[str] = Counter()
    condition_correct: Counter[str] = Counter()
    for row_id, candidates in grouped.items():
        label = labels[row_id].answer
        for candidate in candidates:
            path = candidate.extraction.path
            path_total[path] += 1
            path_correct[path] += int(candidate.extraction.answer == label)
            reasons = low_quality_vote_reasons(
                candidate.extraction,
                hit_max_new_tokens=candidate.hit_max_new_tokens,
            )
            for reason in reasons:
                condition_total[reason] += 1
                condition_correct[reason] += int(candidate.extraction.answer == label)

    bands: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in prediction_rows:
        row_id = str(row["id"])
        vote_counts = row["unfiltered_vote_counts"]
        assert isinstance(vote_counts, Mapping)
        top_count = max((int(value) for value in vote_counts.values()), default=0)
        band = support_band(top_count)
        reference_correct = row["unfiltered_answer"] == labels[row_id].answer
        filtered_correct = row["filtered_answer"] == labels[row_id].answer
        bands[band]["questions"] += 1
        bands[band]["changed"] += int(row["unfiltered_answer"] != row["filtered_answer"])
        bands[band]["improved"] += int(filtered_correct and not reference_correct)
        bands[band]["regressed"] += int(reference_correct and not filtered_correct)

    return {
        "marginal_extraction_path_accuracy": {
            path: {
                "generations": path_total[path],
                "correct": path_correct[path],
                "accuracy": path_correct[path] / path_total[path],
            }
            for path in sorted(path_total)
        },
        "filter_condition_marginal_accuracy": {
            reason: {
                "generations": condition_total[reason],
                "correct": condition_correct[reason],
                "accuracy": condition_correct[reason] / condition_total[reason],
            }
            for reason in sorted(condition_total)
        },
        "winner_support_bands": {
            band: dict(sorted(counts.items())) for band, counts in sorted(bands.items())
        },
        "labels_used_only_for_posthoc_diagnostics": True,
    }


def submission_csv_bytes(payload: Mapping[str, object]) -> bytes:
    headers = payload.get("headers")
    rows = payload.get("rows")
    if not isinstance(headers, list) or not isinstance(rows, list):
        raise ValueError("Invalid submission payload")
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\r\n")
    writer.writerow([str(value) for value in headers])
    for row in rows:
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError("Submission payload row must contain two cells")
        writer.writerow([str(value) for value in row])
    return stream.getvalue().encode("utf-8")


def junit_passed(path: Path) -> dict[str, object]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    tests = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
    failures = sum(int(suite.attrib.get("failures", 0)) for suite in suites)
    errors = sum(int(suite.attrib.get("errors", 0)) for suite in suites)
    skipped = sum(int(suite.attrib.get("skipped", 0)) for suite in suites)
    return {
        "tests": tests,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "passed": tests > 0 and failures == 0 and errors == 0,
    }


def build_comparison_markdown(comparison: Mapping[str, object]) -> str:
    union = nested_dict(comparison, "union")
    decision = nested_dict(comparison, "preregistered_decision")
    lines = [
        "# T8-3 vote-filter comparison",
        "",
        "The frozen filter removes weak-path, truncated, or conflicting votes and uses",
        "the original per-question majority only when every usable vote was removed.",
        "Ground truth was loaded only after both prediction maps were frozen.",
        "",
        "| Scope | T8 majority@32 | T8-3 filtered | Delta | McNemar p |",
        "|---|---:|---:|---:|---:|",
        (
            f"| union ({union['questions']}) | {float(union['reference_accuracy']):.2%} | "
            f"{float(union['candidate_accuracy']):.2%} | {float(union['delta_pp']):+.2f}pp | "
            f"{float(union['two_sided_exact_p']):.3g} |"
        ),
    ]
    splits = comparison.get("splits")
    assert isinstance(splits, Mapping)
    for name in EXPECTED_SPLITS:
        report = splits[name]
        assert isinstance(report, Mapping)
        lines.append(
            f"| {name} ({report['questions']}) | {float(report['reference_accuracy']):.2%} | "
            f"{float(report['candidate_accuracy']):.2%} | {float(report['delta_pp']):+.2f}pp | "
            f"{float(report['two_sided_exact_p']):.3g} |"
        )
    lines.extend(
        [
            "",
            "## Preregistered decision",
            "",
            f"**{str(decision['status']).upper()}** — {decision['reason']}",
            "",
            "The holdout policy was a post-hoc discovery. The filtered 831-row leaderboard",
            "artifact is retained as a label-blind candidate, but the frozen final strategy",
            "remains unfiltered T8 fixed majority@32 unless the adoption gate is changed in a",
            "separate documented decision.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    config = validate_config(args.config)
    protected_before = protected_snapshot(config)
    sources = nested_dict(config, "sources")
    outputs = nested_dict(config, "outputs")
    artifact_dir = Path(str(outputs["artifact_dir"]))
    submission_dir = Path(str(outputs["submission_dir"]))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)

    tests_path = artifact_dir / "tests.xml"
    tests_result = junit_passed(tests_path)
    if not tests_result["passed"]:
        raise ValueError("Focused T8-3 tests did not pass")

    union_ids_path = Path(str(sources["union_ids"]))
    union_ids = load_ids(union_ids_path)
    holdout_generations_path = Path(str(sources["holdout_generations"]))
    holdout_metadata_path = Path(str(sources["holdout_metadata"]))
    holdout_pool_sha_before = sha256_file(holdout_generations_path)
    holdout_metadata = validate_generation_metadata(
        holdout_metadata_path,
        holdout_generations_path,
        expected_selected_rows=len(union_ids),
    )
    holdout_generations = load_generations(holdout_generations_path)
    holdout_grouped = group_generations(holdout_generations)
    ensure_coverage(holdout_grouped, union_ids, k=32)
    (
        reference_predictions,
        filtered_predictions,
        filtered_selection,
        prediction_rows,
        holdout_filter_diagnostics,
    ) = build_policy_predictions(holdout_grouped, union_ids)

    leaderboard_input = Path(str(sources["leaderboard_input"]))
    leaderboard_generations = Path(str(sources["leaderboard_generations"]))
    leaderboard_metadata = Path(str(sources["leaderboard_metadata"]))
    leaderboard_pool_sha_before = sha256_file(leaderboard_generations)
    validate_generation_metadata(leaderboard_metadata, leaderboard_generations)
    unfiltered_payload = build_submission_payload(
        input_path=leaderboard_input,
        generations_path=leaderboard_generations,
        k=32,
        config_path=Path("configs/t8_self_consistency.json"),
        metadata_path=leaderboard_metadata,
        allow_generation_superset=True,
        filter_low_quality_votes=False,
    )
    filtered_payload = build_submission_payload(
        input_path=leaderboard_input,
        generations_path=leaderboard_generations,
        k=32,
        config_path=Path("configs/t8_self_consistency.json"),
        metadata_path=leaderboard_metadata,
        allow_generation_superset=True,
        filter_low_quality_votes=True,
    )
    frozen_at = utc_now()

    canonical_path = Path(str(sources["canonical"]))
    canonical_labels = load_labels(canonical_path)
    union_labels = {row_id: canonical_labels[row_id] for row_id in union_ids}
    split_paths = nested_dict(sources, "splits")
    if set(split_paths) != set(EXPECTED_SPLITS):
        raise ValueError("Config must contain all four fixed holdout splits")
    split_labels = {
        name: load_labels(Path(str(split_paths[name]))) for name in EXPECTED_SPLITS
    }

    results = nested_dict(holdout_metadata, "results")
    generation_wall = float(results["generation_wall_seconds"])
    reference_selection = {
        row_id: list(holdout_grouped[row_id]) for row_id in union_ids
    }
    reference_metrics = evaluate(
        flatten_selection(reference_selection, union_ids),
        union_labels,
        wall_seconds=generation_wall,
    )
    filtered_metrics = evaluate(
        flatten_selection(filtered_selection, union_ids),
        union_labels,
        wall_seconds=generation_wall,
    )
    union_comparison = exact_mcnemar(
        filtered_predictions,
        reference_predictions,
        union_labels,
        union_ids,
    )
    split_comparisons: dict[str, object] = {}
    split_metric_reports: dict[str, object] = {}
    for name in EXPECTED_SPLITS:
        ids = list(split_labels[name])
        split_comparisons[name] = exact_mcnemar(
            filtered_predictions,
            reference_predictions,
            split_labels[name],
            ids,
        )
        split_wall = generation_wall * len(ids) / len(union_ids)
        split_metric_reports[name] = {
            "unfiltered": evaluate(
                flatten_selection(reference_selection, ids),
                split_labels[name],
                wall_seconds=split_wall,
            ),
            "filtered": evaluate(
                flatten_selection(filtered_selection, ids),
                split_labels[name],
                wall_seconds=split_wall,
            ),
        }

    template_groups = load_template_groups(
        Path(str(sources["template_group_audit"])), union_ids
    )
    cv_report = cross_validate(
        reference_predictions=reference_predictions,
        filtered_predictions=filtered_predictions,
        labels=union_labels,
        ids=union_ids,
        groups=template_groups,
        config=config,
    )

    holdout_posthoc = posthoc_diagnostics(
        prediction_rows=prediction_rows,
        grouped=holdout_grouped,
        labels=union_labels,
    )
    for row in prediction_rows:
        row_id = str(row["id"])
        label = union_labels[row_id].answer
        row["ground_truth"] = label
        row["unfiltered_correct"] = row["unfiltered_answer"] == label
        row["filtered_correct"] = row["filtered_answer"] == label
        row["ground_truth_attached_after_prediction_freeze"] = True

    reference_invalid = float(reference_metrics["invalid_output_rate"])
    filtered_invalid = float(filtered_metrics["invalid_output_rate"])
    invalid_delta_pp = (filtered_invalid - reference_invalid) * 100
    hard_delta = float(nested_dict(split_comparisons, "hard_diagnostic")["delta_pp"])
    format_delta = float(nested_dict(split_comparisons, "format_diagnostic")["delta_pp"])
    gate = nested_dict(config, "decision_gate")
    effect_pass = float(union_comparison["delta_pp"]) >= float(
        gate["minimum_union_delta_pp"]
    )
    significance_pass = float(union_comparison["two_sided_exact_p"]) < float(
        gate["maximum_exact_mcnemar_p"]
    )
    guardrail_pass = (
        hard_delta >= -float(gate["maximum_hard_or_format_drop_pp"])
        and format_delta >= -float(gate["maximum_hard_or_format_drop_pp"])
        and invalid_delta_pp
        <= float(gate["maximum_union_invalid_increase_pp"])
    )
    if effect_pass and significance_pass and guardrail_pass:
        decision_status = "adopt"
        decision_reason = "all preregistered effect, significance, and guardrail gates passed"
    elif float(union_comparison["delta_pp"]) > 0 and guardrail_pass:
        decision_status = "hold"
        decision_reason = (
            "union accuracy improved, but the preregistered +1.5pp effect-size gate was not met"
            if not effect_pass
            else "union accuracy improved, but the preregistered significance gate was not met"
        )
    else:
        decision_status = "reject"
        decision_reason = "union accuracy did not improve or a preregistered guardrail failed"

    comparison: dict[str, object] = {
        "schema_version": 1,
        "task": "T8-3",
        "predictions_frozen_at_utc": frozen_at,
        "reference": "T8 fixed unfiltered majority@32",
        "candidate": "T8-3 frozen low-quality-vote filter at k=32",
        "union": union_comparison,
        "splits": split_comparisons,
        "metrics": {
            "union": {
                "unfiltered": reference_metrics,
                "filtered": filtered_metrics,
            },
            "splits": split_metric_reports,
        },
        "invalid_guardrail": {
            "reference_invalid_output_rate": reference_invalid,
            "filtered_invalid_output_rate": filtered_invalid,
            "delta_pp": invalid_delta_pp,
            "maximum_increase_pp": gate["maximum_union_invalid_increase_pp"],
            "passed": invalid_delta_pp
            <= float(gate["maximum_union_invalid_increase_pp"]),
        },
        "preregistered_decision": {
            "status": decision_status,
            "adopted": decision_status == "adopt",
            "reason": decision_reason,
            "criteria": gate,
            "checks": {
                "effect_size": effect_pass,
                "significance": significance_pass,
                "hard_format_guardrail": guardrail_pass,
            },
        },
        "ground_truth_contract": {
            "used_for_filtering": False,
            "used_for_voting": False,
            "used_for_leaderboard_submission": False,
            "predictions_frozen_before_label_load": True,
            "used_only_for_post_freeze_metrics_and_cross_validation": True,
        },
    }

    unfiltered_bytes = submission_csv_bytes(unfiltered_payload)
    existing_unfiltered_path = Path(str(sources["unfiltered_submission"]))
    existing_unfiltered_bytes = existing_unfiltered_path.read_bytes()
    regression = {
        "expected_path": existing_unfiltered_path.as_posix(),
        "expected_sha256": hashlib.sha256(existing_unfiltered_bytes).hexdigest(),
        "reaggregated_sha256": hashlib.sha256(unfiltered_bytes).hexdigest(),
        "byte_identical": unfiltered_bytes == existing_unfiltered_bytes,
        "row_mismatches": 0,
        "filtered_changed_rows": sum(
            left != right
            for left, right in zip(
                unfiltered_payload["rows"],
                filtered_payload["rows"],
                strict=True,
            )
        ),
    }
    if not regression["byte_identical"]:
        raise ValueError("Filter-off regression did not reproduce the T8 submission bytes")

    prepared_path = submission_dir / "submission-prepared.json"
    submission_path = submission_dir / "submission.csv"
    submission_audit_path = submission_dir / "submission-audit.json"
    submission_diff_path = submission_dir / "diff-vs-t8-unfiltered.json"
    write_json(prepared_path, filtered_payload)
    filtered_csv = submission_csv_bytes(filtered_payload)
    submission_path.write_bytes(filtered_csv)
    unfiltered_rows = {str(row[0]): str(row[1]) for row in unfiltered_payload["rows"]}
    filtered_rows = {str(row[0]): str(row[1]) for row in filtered_payload["rows"]}
    vote_filter_audit = nested_dict(nested_dict(filtered_payload, "audit"), "vote_filter")
    per_question = vote_filter_audit.pop("per_question")
    assert isinstance(per_question, list)
    per_question_map = {str(row["id"]): row for row in per_question}
    changed = [
        {
            "id": row_id,
            "unfiltered_answer": unfiltered_rows[row_id],
            "filtered_answer": filtered_rows[row_id],
            "vote_composition": per_question_map[row_id],
        }
        for row_id in filtered_rows
        if filtered_rows[row_id] != unfiltered_rows[row_id]
    ]
    submission_diff = {
        "schema_version": 1,
        "task": "T8-3",
        "labels_available": False,
        "accuracy_computed": False,
        "rows": len(filtered_rows),
        "changed_count": len(changed),
        "unchanged_count": len(filtered_rows) - len(changed),
        "changes": changed,
        "per_question_vote_composition": per_question,
    }
    write_json(submission_diff_path, submission_diff)
    submission_audit = dict(nested_dict(filtered_payload, "audit"))
    submission_audit["vote_filter"] = vote_filter_audit
    submission_audit.update(
        {
            "output_path": submission_path.as_posix(),
            "output_sha256": hashlib.sha256(filtered_csv).hexdigest(),
            "output_bytes": len(filtered_csv),
            "output_rows": len(filtered_rows),
            "csv_round_trip_verified": True,
            "filter_off_regression": regression,
            "diff_audit": {
                "path": submission_diff_path.as_posix(),
                "changed_count": len(changed),
            },
        }
    )
    write_json(submission_audit_path, submission_audit)

    predictions_path = artifact_dir / "holdout" / "predictions.jsonl"
    comparison_path = artifact_dir / "holdout" / "comparison.json"
    comparison_markdown_path = artifact_dir / "holdout" / "comparison.md"
    cv_path = artifact_dir / "cross-validation.json"
    diagnostics_path = artifact_dir / "vote-filter-diagnostics.json"
    final_config_path = artifact_dir / "final_config.json"
    manifest_path = artifact_dir / "manifest.json"
    write_jsonl(predictions_path, prediction_rows)
    write_json(comparison_path, comparison)
    comparison_markdown_path.write_text(
        build_comparison_markdown(comparison), encoding="utf-8"
    )
    write_json(cv_path, cv_report)
    diagnostics = {
        "schema_version": 1,
        "task": "T8-3",
        "holdout": {
            **holdout_filter_diagnostics,
            **holdout_posthoc,
        },
        "leaderboard_831": {
            "labels_available": False,
            "accuracy_computed": False,
            "vote_filter": vote_filter_audit,
            "changed_answer_count": len(changed),
        },
        "source_pool_sha256": {
            "holdout_before": holdout_pool_sha_before,
            "holdout_after": sha256_file(holdout_generations_path),
            "leaderboard_before": leaderboard_pool_sha_before,
            "leaderboard_after": sha256_file(leaderboard_generations),
        },
    }
    write_json(diagnostics_path, diagnostics)

    final_config = {
        "schema_version": 1,
        "task": "T8-3",
        "status": decision_status,
        "adopted": decision_status == "adopt",
        "candidate_policy": {
            "name": POLICY_NAME,
            "config": args.config.as_posix(),
            "config_sha256": sha256_file(args.config),
            "vote_filter": LOW_QUALITY_VOTE_POLICY,
        },
        "final_strategy": (
            {
                "task": "T8-3",
                "policy": POLICY_NAME,
                "k": 32,
            }
            if decision_status == "adopt"
            else {
                "task": "T8",
                "policy": REFERENCE_POLICY_NAME,
                "k": 32,
                "reason": "T8-3 retained as a candidate but did not pass the +1.5pp gate",
            }
        ),
        "decision": comparison["preregistered_decision"],
        "t10_update_required": decision_status == "adopt",
    }
    write_json(final_config_path, final_config)

    protected_after = verify_protected_snapshot(protected_before)
    if not protected_after["verified"]:
        raise ValueError("A protected T8/T8-1/T8-2/T9 input changed during T8-3")
    completion_checks = {
        "filter_off_submission_byte_identical": regression["byte_identical"],
        "focused_tests_passed": tests_result["passed"],
        "protected_t8_through_t9_hashes_unchanged": protected_after["verified"],
        "filter_config_hash_recorded": bool(sha256_file(args.config)),
        "union_primary_mcnemar_recorded": "two_sided_exact_p" in union_comparison,
        "all_four_split_guardrails_recorded": set(split_comparisons)
        == set(EXPECTED_SPLITS),
        "five_fold_oof_recorded": len(cv_report["folds"]) == 5,
        "ground_truth_free_filter_and_vote": comparison["ground_truth_contract"][
            "predictions_frozen_before_label_load"
        ],
        "filter_fallback_audit_recorded": "fallback_ids"
        in holdout_filter_diagnostics,
        "condition_removal_counts_recorded": "condition_candidate_counts"
        in holdout_filter_diagnostics,
        "preregistered_decision_recorded": decision_status
        in {"adopt", "hold", "reject"},
        "final_strategy_recorded": "final_strategy" in final_config,
        "holdout_generation_pool_unchanged": holdout_pool_sha_before
        == sha256_file(holdout_generations_path),
        "leaderboard_generation_pool_unchanged": leaderboard_pool_sha_before
        == sha256_file(leaderboard_generations),
        "leaderboard_has_no_accuracy_metric": submission_diff["accuracy_computed"]
        is False,
    }
    if not all(completion_checks.values()):
        failed = [name for name, passed in completion_checks.items() if not passed]
        raise ValueError(f"T8-3 completion checks failed: {failed}")

    manifest = {
        "schema_version": 1,
        "task": "T8-3",
        "status": "complete",
        "created_at_utc": utc_now(),
        "objective": "reaggregate immutable T8 k=32 pools with a label-blind low-quality vote filter",
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "model": {
            "id": EXPECTED_MODEL,
            "revision": EXPECTED_REVISION,
            "tokenizer_revision": EXPECTED_REVISION,
            "adapter": None,
            "new_generations": 0,
        },
        "policy": {
            "name": POLICY_NAME,
            "config_sha256": sha256_file(args.config),
            "definition": LOW_QUALITY_VOTE_POLICY,
        },
        "decision": comparison["preregistered_decision"],
        "final_strategy": final_config["final_strategy"],
        "presentation_record": {
            "random_accuracy": nested_dict(split_comparisons, "random_holdout")[
                "candidate_accuracy"
            ],
            "template_accuracy": nested_dict(split_comparisons, "template_holdout")[
                "candidate_accuracy"
            ],
            "hard_accuracy": nested_dict(split_comparisons, "hard_diagnostic")[
                "candidate_accuracy"
            ],
            "format_accuracy": nested_dict(split_comparisons, "format_diagnostic")[
                "candidate_accuracy"
            ],
            "random_invalid_output_rate": nested_dict(
                nested_dict(split_metric_reports, "random_holdout"), "filtered"
            )["invalid_output_rate"],
            "union_delta_vs_t8_pp": union_comparison["delta_pp"],
            "union_mcnemar_p": union_comparison["two_sided_exact_p"],
        },
        "completion_checks": completion_checks,
        "tests": {**tests_result, "record": file_record(tests_path)},
        "protected_inputs": {
            "before": protected_before,
            "after_verification": protected_after,
        },
        "sources": {
            "config": file_record(args.config),
            "canonical": file_record(canonical_path, rows=len(canonical_labels)),
            "union_ids": file_record(union_ids_path, rows=len(union_ids)),
            "template_group_audit": file_record(
                Path(str(sources["template_group_audit"]))
            ),
            "holdout_generations": file_record(
                holdout_generations_path, rows=len(holdout_generations)
            ),
            "holdout_metadata": file_record(holdout_metadata_path),
            "leaderboard_input": file_record(leaderboard_input, rows=831),
            "leaderboard_generations": file_record(
                leaderboard_generations,
                rows=int(
                    nested_dict(load_json(leaderboard_metadata), "output")["rows"]
                ),
            ),
            "leaderboard_metadata": file_record(leaderboard_metadata),
            "unfiltered_submission": file_record(existing_unfiltered_path, rows=831),
            "splits": {
                name: file_record(Path(str(split_paths[name])), rows=len(split_labels[name]))
                for name in EXPECTED_SPLITS
            },
            "implementation": file_record(Path(__file__)),
            "submission_implementation": file_record(Path("src/submit.py")),
        },
        "outputs": {
            "predictions": file_record(predictions_path, rows=len(prediction_rows)),
            "comparison": file_record(comparison_path),
            "comparison_markdown": file_record(comparison_markdown_path),
            "cross_validation": file_record(cv_path),
            "diagnostics": file_record(diagnostics_path),
            "final_config": file_record(final_config_path),
            "submission_prepared": file_record(prepared_path, rows=831),
            "submission": file_record(submission_path, rows=831),
            "submission_audit": file_record(submission_audit_path),
            "submission_diff": file_record(submission_diff_path, rows=len(changed)),
        },
        "raw_generations_deleted": False,
    }
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "event": "t8_3_vote_filter_complete",
                "decision": decision_status,
                "union_delta_pp": union_comparison["delta_pp"],
                "mcnemar_p": union_comparison["two_sided_exact_p"],
                "leaderboard_changes": len(changed),
                "manifest": manifest_path.as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_policy_predictions",
    "cross_validate",
    "fold_for_group",
    "submission_csv_bytes",
]
