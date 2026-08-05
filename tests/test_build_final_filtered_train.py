from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from scripts.build_final_filtered_train import build_dataset


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_dataset_combines_exclusion_sources_and_preserves_order(
    tmp_path: Path,
) -> None:
    source = tmp_path / "data" / "source.csv"
    organizer = tmp_path / "data" / "organizer.csv"
    supplemental = tmp_path / "data" / "supplemental.csv"
    origin = tmp_path / "message.txt"
    origin.write_text("review evidence\n", encoding="utf-8")
    origin_hash = _sha256(origin)

    _write_csv(
        source,
        ["id", "question", "answer"],
        [
            {"id": "train-000000", "question": "keep first", "answer": "1"},
            {"id": "train-000001", "question": "line one\nline two", "answer": "2"},
            {"id": "train-000002", "question": "reviewed mismatch", "answer": "3"},
            {"id": "train-000003", "question": "keep last", "answer": "4"},
        ],
    )
    _write_csv(
        organizer,
        ["id", "answer", "question"],
        [{"id": "train-000001", "answer": "2", "question": "line one line two"}],
    )
    _write_csv(
        supplemental,
        [
            "id",
            "category",
            "source_answer",
            "reviewed_expected_answer",
            "reason",
            "source_reference",
            "source_sha256",
        ],
        [
            {
                "id": "train-000002",
                "category": "verified_label_mismatch",
                "source_answer": "3",
                "reviewed_expected_answer": "30",
                "reason": "independent calculation",
                "source_reference": origin.as_posix(),
                "source_sha256": origin_hash,
            }
        ],
    )

    config = {
        "schema_version": 1,
        "policy_version": "test-v1",
        "repository_root": ".",
        "inputs": {
            "source": {"path": source.as_posix(), "sha256": _sha256(source)},
            "organizer_exclusions": {
                "path": organizer.as_posix(),
                "sha256": _sha256(organizer),
            },
            "supplemental_exclusions": {
                "path": supplemental.as_posix(),
                "sha256": _sha256(supplemental),
            },
            "supplemental_origin": {
                "path_as_received": origin.as_posix(),
                "sha256": origin_hash,
                "required_for_reproduction": False,
            },
        },
        "outputs": {
            "dataset": (tmp_path / "out" / "final.csv").as_posix(),
            "audit": (tmp_path / "out" / "audit.csv").as_posix(),
            "manifest": (tmp_path / "out" / "manifest.json").as_posix(),
        },
        "expected_counts": {
            "source_rows": 4,
            "organizer_exclusions": 1,
            "supplemental_exclusions": 1,
            "exclusion_overlap": 0,
            "total_exclusions": 2,
            "output_rows": 2,
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    manifest = build_dataset(
        config_path,
        generated_at_utc="2026-08-04T00:00:00+00:00",
    )

    with (tmp_path / "out" / "final.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        output_rows = list(csv.DictReader(handle))
    assert [row["id"] for row in output_rows] == ["train-000000", "train-000003"]
    assert manifest["decision_counts"] == {
        "keep": 2,
        "remove": 2,
        "removal_rate": 0.5,
        "organizer_only": 1,
        "supplemental_only": 1,
        "both_sources": 0,
    }
    assert all(manifest["quality_checks"].values())
