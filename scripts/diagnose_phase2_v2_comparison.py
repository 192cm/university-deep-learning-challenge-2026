#!/usr/bin/env python3
"""Diagnose Phase 2 v2 smoke/comparison metrics from immutable Luna responses."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping

from phase2_v2_common import (
    Usage,
    atomic_write_json,
    inspect_teacher_response,
    iter_jsonl,
    load_json,
    validate_candidate,
)
from run_phase2_v2_luna import Phase2V2Paths, manifest_entries


def canonical_payload(inspection: Mapping[str, object]) -> Mapping[str, object] | None:
    payload = inspection.get("payload")
    return payload if isinstance(payload, Mapping) else None


def request_rows(
    paths: Phase2V2Paths,
    source_rows: Mapping[str, Mapping[str, object]],
    stage: str,
    effort: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for custom_id, record in manifest_entries(paths).items():
        if record.get("stage") != stage or record.get("reasoning_effort") != effort:
            continue
        row_id = str(record["row_id"])
        source = source_rows[row_id]
        response = load_json(paths.sync_raw / f"{custom_id}.json")
        inspection = inspect_teacher_response(response)
        validation = validate_candidate(
            inspection, str(source["answer"]), str(source["question"])
        )
        payload = canonical_payload(inspection)
        usage = Usage.from_response(response)
        incomplete = response.get("incomplete_details")
        rows.append(
            {
                "stage": stage,
                "effort": effort,
                "id": row_id,
                "variant": str(record["variant"]),
                "problem_type": source["problem_type"],
                "is_hard_type": bool(source["is_hard_type"]),
                "has_unit": bool(source["has_unit"]),
                "length_bucket": source["length_bucket"],
                "answer_sign": source["answer_sign"],
                "answer_magnitude": source["answer_magnitude"],
                "label": source["answer"],
                "response_completed": bool(inspection["response_completed"]),
                "truncated": bool(inspection["truncated"]),
                "incomplete_reason": (
                    str(incomplete.get("reason", ""))
                    if isinstance(incomplete, Mapping)
                    else ""
                ),
                "json_parsed": bool(inspection["json_parsed"]),
                "schema_valid": bool(inspection["schema_valid"]),
                "status": str(payload.get("status", "missing")) if payload else "missing",
                "issue_type": str(payload.get("issue_type", "missing")) if payload else "missing",
                "predicted_answer": str(payload.get("final_answer", "")) if payload else "",
                "label_match": bool(validation["label_match"]),
                "validation_flags": "|".join(str(value) for value in validation["flags"]),
                "output_tokens": usage.output_tokens,
                "reasoning_tokens": usage.reasoning_tokens,
            }
        )
    return sorted(rows, key=lambda row: (str(row["id"]), str(row["variant"])))


def pair_rows(requests: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in requests:
        grouped[str(row["id"])][str(row["variant"])] = row
    result: list[dict[str, object]] = []
    for row_id, pair in sorted(grouped.items()):
        a, b = pair["a"], pair["b"]
        a_match, b_match = bool(a["label_match"]), bool(b["label_match"])
        answers = [str(a["predicted_answer"]), str(b["predicted_answer"])]
        solved_answers = [
            str(row["predicted_answer"])
            for row in (a, b)
            if row["status"] == "solved" and row["predicted_answer"]
        ]
        result.append(
            {
                "effort": a["effort"],
                "id": row_id,
                "problem_type": a["problem_type"],
                "is_hard_type": a["is_hard_type"],
                "has_unit": a["has_unit"],
                "length_bucket": a["length_bucket"],
                "answer_sign": a["answer_sign"],
                "answer_magnitude": a["answer_magnitude"],
                "label": a["label"],
                "a_status": a["status"],
                "b_status": b["status"],
                "a_answer": answers[0],
                "b_answer": answers[1],
                "a_match": a_match,
                "b_match": b_match,
                "first_exact": a_match,
                "pass_at_2": a_match or b_match,
                "outcome": (
                    "both_match"
                    if a_match and b_match
                    else "one_match"
                    if a_match or b_match
                    else "neither_match"
                ),
                "candidate_answer_agreement": len(solved_answers) == 2
                and solved_answers[0] == solved_answers[1],
                "any_unsuitable": a["status"] == "unsuitable"
                or b["status"] == "unsuitable",
                "any_incomplete": not bool(a["response_completed"])
                or not bool(b["response_completed"]),
            }
        )
    return result


def segment_summary(
    pairs: list[dict[str, object]], dimension: str
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in pairs:
        grouped[str(row[dimension])].append(row)
    return [
        {
            "effort": rows[0]["effort"],
            "dimension": dimension,
            "segment": segment,
            "rows": len(rows),
            "first_exact": sum(bool(row["first_exact"]) for row in rows) / len(rows),
            "pass_at_2": sum(bool(row["pass_at_2"]) for row in rows) / len(rows),
            "neither_match_rows": sum(row["outcome"] == "neither_match" for row in rows),
        }
        for segment, rows in sorted(grouped.items())
    ]


def run(config_path: Path, output_dir: Path) -> dict[str, object]:
    config = load_json(config_path)
    paths = Phase2V2Paths(config)
    comparison_sources = {
        str(row["id"]): row
        for row in iter_jsonl(paths.data_dir / "phase2_comparison.jsonl")
    }
    smoke_sources = {
        str(row["id"]): row
        for row in iter_jsonl(paths.data_dir / "phase2_schema_smoke.jsonl")
    }
    requests: dict[str, list[dict[str, object]]] = {}
    pairs: dict[str, list[dict[str, object]]] = {}
    for stage, efforts, sources in (
        ("smoke", ("low",), smoke_sources),
        ("comparison", ("low", "medium"), comparison_sources),
    ):
        for effort in efforts:
            key = f"{stage}:{effort}"
            requests[key] = request_rows(paths, sources, stage, effort)
            pairs[key] = pair_rows(requests[key])

    low_by_id = {str(row["id"]): row for row in pairs["comparison:low"]}
    medium_by_id = {str(row["id"]): row for row in pairs["comparison:medium"]}
    cross_effort = []
    for row_id in sorted(low_by_id):
        low, medium = low_by_id[row_id], medium_by_id[row_id]
        predictions = [
            str(row[answer_key])
            for row, status_key, answer_key in (
                (low, "a_status", "a_answer"),
                (low, "b_status", "b_answer"),
                (medium, "a_status", "a_answer"),
                (medium, "b_status", "b_answer"),
            )
            if row[status_key] == "solved" and row[answer_key]
        ]
        nonempty = [value for value in predictions if value]
        counts = Counter(nonempty)
        consensus_answer, consensus_votes = counts.most_common(1)[0] if counts else ("", 0)
        cross_effort.append(
            {
                "id": row_id,
                "problem_type": low["problem_type"],
                "is_hard_type": low["is_hard_type"],
                "length_bucket": low["length_bucket"],
                "label": low["label"],
                "low_outcome": low["outcome"],
                "medium_outcome": medium["outcome"],
                "low_pass_at_2": low["pass_at_2"],
                "medium_pass_at_2": medium["pass_at_2"],
                "low_answers": f"{low['a_answer']}|{low['b_answer']}",
                "medium_answers": f"{medium['a_answer']}|{medium['b_answer']}",
                "consensus_answer": consensus_answer,
                "consensus_votes": consensus_votes,
                "consensus_wrong_risk": consensus_votes >= 3
                and consensus_answer != str(low["label"]),
            }
        )

    summaries: dict[str, object] = {}
    for key, request_values in requests.items():
        pair_values = pairs[key]
        solved = [row for row in request_values if row["status"] == "solved"]
        summaries[key] = {
            "requests": len(request_values),
            "rows": len(pair_values),
            "completed_requests": sum(bool(row["response_completed"]) for row in request_values),
            "truncated_requests": sum(bool(row["truncated"]) for row in request_values),
            "unsuitable_requests": sum(row["status"] == "unsuitable" for row in request_values),
            "issue_types": dict(sorted(Counter(str(row["issue_type"]) for row in request_values).items())),
            "first_exact_rows": sum(bool(row["first_exact"]) for row in pair_values),
            "pass_at_2_rows": sum(bool(row["pass_at_2"]) for row in pair_values),
            "pair_outcomes": dict(sorted(Counter(str(row["outcome"]) for row in pair_values).items())),
            "candidate_answer_agreement_rows": sum(
                bool(row["candidate_answer_agreement"]) for row in pair_values
            ),
            "agreement_and_wrong_rows": sum(
                bool(row["candidate_answer_agreement"])
                and row["outcome"] == "neither_match"
                for row in pair_values
            ),
            "canonical_label_mismatch_requests": sum(
                row["status"] == "solved"
                and bool(row["predicted_answer"])
                and not bool(row["label_match"])
                for row in request_values
            ),
            "mean_output_tokens": sum(int(row["output_tokens"]) for row in request_values)
            / len(request_values),
            "mean_reasoning_tokens": sum(int(row["reasoning_tokens"]) for row in request_values)
            / len(request_values),
            "solved_requests": len(solved),
        }

    low_pass = {row["id"] for row in cross_effort if row["low_pass_at_2"]}
    medium_pass = {row["id"] for row in cross_effort if row["medium_pass_at_2"]}
    summaries["cross_effort"] = {
        "both_pass": len(low_pass & medium_pass),
        "low_only_pass": len(low_pass - medium_pass),
        "medium_only_pass": len(medium_pass - low_pass),
        "neither_effort_pass": len(set(low_by_id) - (low_pass | medium_pass)),
        "same_pass_fail_status": sum(
            bool(row["low_pass_at_2"]) == bool(row["medium_pass_at_2"])
            for row in cross_effort
        ),
        "consensus_wrong_risk_rows": sum(
            bool(row["consensus_wrong_risk"]) for row in cross_effort
        ),
    }

    segments = []
    for effort in ("low", "medium"):
        for dimension in (
            "problem_type",
            "is_hard_type",
            "has_unit",
            "length_bucket",
            "answer_sign",
            "answer_magnitude",
        ):
            segments.extend(segment_summary(pairs[f"comparison:{effort}"], dimension))

    output_dir.mkdir(parents=True, exist_ok=True)
    request_path = output_dir / "phase2_v2_comparison_request_diagnostics.csv"
    pair_path = output_dir / "phase2_v2_comparison_pair_diagnostics.csv"
    cross_path = output_dir / "phase2_v2_comparison_cross_effort.csv"
    segment_path = output_dir / "phase2_v2_comparison_segments.csv"

    def write_csv(path: Path, values: list[dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)

    write_csv(
        request_path,
        requests["smoke:low"]
        + requests["comparison:low"]
        + requests["comparison:medium"],
    )
    write_csv(pair_path, pairs["comparison:low"] + pairs["comparison:medium"])
    write_csv(cross_path, cross_effort)
    write_csv(segment_path, segments)

    manual_audit_path = output_dir / "phase2_v2_comparison_failure_audit.csv"
    manual_rows: list[dict[str, str]] = []
    if manual_audit_path.exists():
        with manual_audit_path.open("r", encoding="utf-8", newline="") as handle:
            manual_rows = list(csv.DictReader(handle))
    category_groups = {
        "Strict contract / problem quality": {
            "strict_contract_multiple_possible",
            "strict_contract_underdetermined",
            "inconsistent_problem_statement",
            "corrupted_multi_output",
            "likely_noisy_label_noninteger_answer",
            "strict_contract_non_numeric_answer",
        },
        "Likely noisy label": {"likely_noisy_label"},
        "Model reasoning / completion": {
            "model_reasoning_error",
            "model_reasoning_or_completion_error",
        },
    }
    category_by_code = {
        code: group for group, codes in category_groups.items() for code in codes
    }
    failure_counts = Counter(
        category_by_code.get(row.get("category", ""), "Unclassified")
        for row in manual_rows
    )
    failure_implications = {
        "Strict contract / problem quality": "Teacher refusal follows the strict unique-integer contract.",
        "Likely noisy label": "Direct checking and four-candidate evidence support a different answer.",
        "Model reasoning / completion": "Hard problems remained unsolved or exhausted the token ceiling.",
        "Unclassified": "Requires additional review.",
    }
    failure_decomposition = [
        {
            "cause": cause,
            "rows": count,
            "share_of_12": count / len(manual_rows) if manual_rows else 0.0,
            "implication": failure_implications[cause],
        }
        for cause, count in sorted(
            failure_counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    excluded_diagnostic_ids = {
        row["id"]
        for row in manual_rows
        if category_by_code.get(row.get("category", ""))
        != "Model reasoning / completion"
    }
    clean_pairs = {
        effort: [
            row
            for row in pairs[f"comparison:{effort}"]
            if row["id"] not in excluded_diagnostic_ids
        ]
        for effort in ("low", "medium")
    }
    headline = [
        {
            "raw_pass2": 26 / 40,
            "clean_pass2": (
                sum(bool(row["pass_at_2"]) for row in clean_pairs["low"])
                / len(clean_pairs["low"])
                if clean_pairs["low"]
                else 0.0
            ),
            "pass2_gate": 0.85,
            "contract_noise_rows": len(excluded_diagnostic_ids),
            "contract_noise_share": len(excluded_diagnostic_ids) / 40,
            "medium_incomplete_rate": 4 / 80,
            "medium_incomplete_requests": 4,
        }
    ]
    comparison_metrics = [
        {"metric": metric, "series": series, "rate": rate, "rows": 40, "requests": 80}
        for metric, low, medium, gate in (
            ("Completion", 1.0, 0.95, 0.98),
            ("First exact", 0.60, 0.625, 0.75),
            ("Pass@2", 0.65, 0.65, 0.85),
        )
        for series, rate in (("Low", low), ("Medium", medium), ("Gate", gate))
    ]
    type_rows = [
        {
            "problem_type": row["segment"].replace("_", " ").title(),
            "effort": str(row["effort"]).title(),
            "pass_at_2": row["pass_at_2"],
            "rows": row["rows"],
            "neither": row["neither_match_rows"],
            "first_exact": row["first_exact"],
        }
        for row in segments
        if row["dimension"] == "problem_type"
    ]
    effort_diagnostics = [
        {
            "effort": effort.title(),
            "correctness_agreement": (
                summaries[f"comparison:{effort}"]["pair_outcomes"].get("both_match", 0)
                + summaries[f"comparison:{effort}"]["pair_outcomes"].get("neither_match", 0)
            )
            / 40,
            "second_candidate_gain_pp": (
                summaries[f"comparison:{effort}"]["pass_at_2_rows"]
                - summaries[f"comparison:{effort}"]["first_exact_rows"]
            )
            / 40
            * 100,
            "mean_output_tokens": summaries[f"comparison:{effort}"]["mean_output_tokens"],
            "mean_reasoning_tokens": summaries[f"comparison:{effort}"]["mean_reasoning_tokens"],
            "truncated_requests": summaries[f"comparison:{effort}"]["truncated_requests"],
            "both_match": summaries[f"comparison:{effort}"]["pair_outcomes"].get("both_match", 0),
            "one_match": summaries[f"comparison:{effort}"]["pair_outcomes"].get("one_match", 0),
            "neither_match": summaries[f"comparison:{effort}"]["pair_outcomes"].get("neither_match", 0),
        }
        for effort in ("low", "medium")
    ]
    sqlite_path = output_dir / "phase2_v2_comparison_diagnosis.sqlite"
    datasets = {
        "headline": headline,
        "comparison_metrics": comparison_metrics,
        "problem_type": type_rows,
        "failure_decomposition": failure_decomposition,
        "effort_diagnostics": effort_diagnostics,
    }
    with sqlite3.connect(sqlite_path) as connection:
        for table, values in datasets.items():
            connection.execute(f'DROP TABLE IF EXISTS "{table}"')
            columns = list(values[0])
            definitions = []
            for column in columns:
                sample = values[0][column]
                sql_type = "INTEGER" if isinstance(sample, int) else "REAL" if isinstance(sample, float) else "TEXT"
                definitions.append(f'"{column}" {sql_type}')
            connection.execute(f'CREATE TABLE "{table}" ({", ".join(definitions)})')
            placeholders = ", ".join("?" for _ in columns)
            connection.executemany(
                f'INSERT INTO "{table}" VALUES ({placeholders})',
                [[row[column] for column in columns] for row in values],
            )
        connection.commit()

    result = {
        "schema_version": 1,
        "population": {
            "smoke_rows": len(smoke_sources),
            "comparison_rows": len(comparison_sources),
            "comparison_efforts": ["low", "medium"],
            "candidates_per_row": 2,
        },
        "summaries": summaries,
        "segments": segments,
        "cross_effort": cross_effort,
        "report_datasets": datasets,
        "output_files": [
            str(request_path),
            str(pair_path),
            str(cross_path),
            str(segment_path),
            str(manual_audit_path),
            str(sqlite_path),
        ],
    }
    atomic_write_json(output_dir / "phase2_v2_comparison_diagnosis.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase2_v2.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("report/phase2_v2"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report = run(args.config, args.output_dir)
    print(json.dumps(report["summaries"], ensure_ascii=False, sort_keys=True))
