#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent consistency checks for the T8 diagnostic deliverable."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


REPORT_DIR = Path(__file__).resolve().parent


def rows(name: str) -> list[dict[str, str]]:
    with (REPORT_DIR / name).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    summary = json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))
    artifact = json.loads((REPORT_DIR / "artifact.json").read_text(encoding="utf-8"))
    html = (REPORT_DIR / "report.html").read_text(encoding="utf-8")

    checks: dict[str, bool] = {}

    outcomes = rows("outcome_decomposition.csv")
    checks["outcomes_sum_to_3737"] = sum(int(row["questions"]) for row in outcomes) == 3737
    checks["outcome_counts_exact"] = [int(row["questions"]) for row in outcomes] == [2590, 564, 583]

    support = rows("support_bands.csv")
    margins = rows("margin_bands.csv")
    ranks = rows("correct_rank.csv")
    checks["support_bands_sum_to_564"] = sum(int(row["selection_failures"]) for row in support) == 564
    checks["margin_bands_sum_to_564"] = sum(int(row["selection_failures"]) for row in margins) == 564
    checks["correct_ranks_sum_to_564"] = sum(int(row["selection_failures"]) for row in ranks) == 564
    checks["rank_distribution_exact"] = {
        int(row["correct_rank"]): int(row["selection_failures"]) for row in ranks
    } == {1: 29, 2: 253, 3: 99, 4: 68, 5: 40, 6: 29, 7: 24, 8: 15, 9: 5, 10: 2}

    ceiling = rows("oracle_rank_ceiling.csv")
    checks["oracle_ceiling_starts_at_majority"] = int(ceiling[0]["oracle_correct"]) == 2590
    checks["oracle_ceiling_ends_at_pass32"] = int(ceiling[-1]["oracle_correct"]) == 3154

    problem_types = rows("problem_type_segments.csv")
    lengths = rows("question_length_segments.csv")
    checks["problem_types_partition_union"] = sum(int(row["questions"]) for row in problem_types) == 3737
    checks["length_buckets_partition_union"] = sum(int(row["questions"]) for row in lengths) == 3737
    checks["segment_selection_failures_sum"] = (
        sum(int(row["selection_failure_count"]) for row in problem_types) == 564
        and sum(int(row["selection_failure_count"]) for row in lengths) == 564
    )

    filters = summary["vote_filter_comparison"]
    checks["filter_paired_identity"] = (
        filters["filtered_correct"] == 2645
        and filters["selection_failures_recovered"] == 69
        and filters["base_correct_broken"] == 14
        and filters["net_gain_questions"] == 55
        and 2590 + 69 - 14 == 2645
    )

    examples = rows("example_cases.csv")
    example_by_id = {row["id"]: row for row in examples}
    checks["seven_verified_examples"] = len(examples) == 7
    checks["parser_example_exact"] = (
        example_by_id["train-012155"]["vote_summary"] == "50=16, 400=16"
        and example_by_id["train-012155"]["t8_3_result"] == "회수"
    )
    checks["no_correct_example_exact"] = (
        example_by_id["train-008043"]["correct_votes"] == "0"
        and example_by_id["train-008043"]["vote_summary"] == "252=31, 32=1"
    )

    manifest = artifact["manifest"]
    datasets = artifact["snapshot"]["datasets"]
    source_map = {source["id"]: source for source in artifact["sources"]}
    checks["title_block_matches_manifest"] = manifest["blocks"][0]["body"].splitlines()[0] == f"# {manifest['title']}"
    checks["all_widget_datasets_exist"] = all(
        item["dataset"] in datasets
        for item in [*manifest["cards"], *manifest["charts"], *manifest["tables"]]
    )
    checks["all_widget_sources_exist"] = all(
        item["sourceId"] in source_map
        for item in [*manifest["cards"], *manifest["charts"], *manifest["tables"]]
    )
    checks["native_widget_sources_have_sql"] = all(
        bool(source_map[item["sourceId"]].get("query", {}).get("sql", "").strip())
        for item in [*manifest["cards"], *manifest["charts"], *manifest["tables"]]
    )

    checks["all_blocks_packaged"] = all(
        f'data-artifact-block-id="{block["id"]}"' in html for block in manifest["blocks"]
    )
    checks["all_charts_packaged"] = all(
        f'data-chart-id="{chart["id"]}"' in html for chart in manifest["charts"]
    )
    checks["all_tables_packaged"] = all(
        f'data-table-id="{table["id"]}"' in html for table in manifest["tables"]
    )
    checks["critical_text_packaged"] = all(
        text in html
        for text in ["현재 오답 1,147건", "train-012155", "T8-3 출력 품질 필터", "Recommended Next Experiments"]
    )

    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"Validation failures: {failed}")

    result = {
        "schema_version": 1,
        "validated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "passed",
        "checks": checks,
        "note": "Portable builder validation and packaging passed. Browser screenshot verification was unavailable; HTML structure and packaged chart/table fallbacks were checked independently.",
    }
    (REPORT_DIR / "validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
