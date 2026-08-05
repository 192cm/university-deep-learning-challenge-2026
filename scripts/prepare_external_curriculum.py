#!/usr/bin/env python3
"""Download, filter, and locally decontaminate OpenMathInstruct-2 curriculum rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping

from phase2_common import (
    atomic_write_json,
    atomic_write_jsonl,
    exact_question_key,
    json_dumps,
    load_json,
    normalize_teacher_answer,
    normalize_template,
    sha256_file,
    stable_hash,
    token_shingles,
    utc_now,
)


FORBIDDEN_RE = re.compile(
    r"(?i)(?:```|\bpython\b|\bsympy\b|\bcode interpreter\b|"
    r"\bcalculator\b|\bwolfram\b|\bweb search\b|\btool call\b)"
)
NON_ENGLISH_RE = re.compile(r"[^\x00-\x7f]")


def get_json(url: str, retries: int = 5) -> dict[str, object]:
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "phase2-curriculum/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Dataset Viewer returned non-object JSON")
            return payload
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt >= retries:
                raise
            time.sleep(min(20, 2**attempt))
    raise AssertionError("unreachable")


def download_file(url: str, target: Path, retries: int = 5) -> dict[str, object]:
    """Download a revision-pinned source shard with resumable Range requests."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return {
            "url": url,
            "path": str(target),
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
            "reused_local_cache": True,
        }
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(retries + 1):
        existing = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "phase2-curriculum/1.0"}
        if existing:
            headers["Range"] = f"bytes={existing}-"
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                append = existing > 0 and getattr(response, "status", None) == 206
                mode = "ab" if append else "wb"
                with partial.open(mode) as handle:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
            os.replace(partial, target)
            return {
                "url": url,
                "path": str(target),
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
                "reused_local_cache": False,
            }
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt >= retries:
                raise
            time.sleep(min(30, 2**attempt))
    raise AssertionError("unreachable")


def import_parquet(cache_dir: Path):
    dependency_dir = cache_dir / "pydeps"
    if dependency_dir.exists():
        sys.path.insert(0, str(dependency_dir.resolve()))
    try:
        import pyarrow.parquet as parquet
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pyarrow is required; install it with "
            f"python -m pip install --target {dependency_dir} pyarrow"
        ) from exc
    return parquet


def load_leaderboard_questions(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{"id": row["id"], "question": row["question"]} for row in reader]


class LocalContaminationIndex:
    def __init__(self, rows: list[dict[str, str]], threshold: float) -> None:
        self.threshold = threshold
        self.exact = {exact_question_key(row["question"]): row["id"] for row in rows}
        self.templates = {normalize_template(row["question"]): row["id"] for row in rows}
        self.shingles = [token_shingles(row["question"]) for row in rows]
        self.ids = [row["id"] for row in rows]
        self.inverted: dict[str, set[int]] = defaultdict(set)
        for index, shingles in enumerate(self.shingles):
            for shingle in shingles:
                self.inverted[shingle].add(index)

    def match(self, question: str) -> tuple[str | None, str | None, float]:
        exact = self.exact.get(exact_question_key(question))
        if exact is not None:
            return "exact", exact, 1.0
        template = self.templates.get(normalize_template(question))
        if template is not None:
            return "template", template, 1.0
        shingles = token_shingles(question)
        if not shingles:
            return None, None, 0.0
        # At Jaccard >= t, at most floor((1-t)*|A|) query shingles may be absent.
        # Therefore any true match must contain at least one of the next N rarest
        # query shingles. This preserves recall while avoiding common-shingle scans.
        probe_count = math.floor((1.0 - self.threshold) * len(shingles)) + 1
        probes = sorted(
            shingles,
            key=lambda shingle: (len(self.inverted.get(shingle, ())), shingle),
        )[:probe_count]
        candidates: set[int] = set()
        for shingle in probes:
            candidates.update(self.inverted.get(shingle, ()))
        best_score = 0.0
        best_index: int | None = None
        minimum_size = math.ceil(self.threshold * len(shingles))
        maximum_size = math.floor(len(shingles) / self.threshold)
        for index in candidates:
            other_size = len(self.shingles[index])
            if not minimum_size <= other_size <= maximum_size:
                continue
            intersection = len(shingles & self.shingles[index])
            union = len(shingles) + other_size - intersection
            score = intersection / union if union else 1.0
            if score > best_score:
                best_score = score
                best_index = index
        if best_index is not None and best_score >= self.threshold:
            return "near", self.ids[best_index], best_score
        return None, None, best_score


def english_ratio(text: str) -> float:
    if not text:
        return 0.0
    return 1.0 - len(NON_ENGLISH_RE.findall(text)) / len(text)


def page_order(total_rows: int, page_size: int, seed: int) -> list[int]:
    pages = list(range((total_rows + page_size - 1) // page_size))
    return sorted(pages, key=lambda page: stable_hash("openmathinstruct2", seed, page))


def run(config_path: Path) -> dict[str, object]:
    config = load_json(config_path)
    external = config["external_curriculum"]
    data = config["data"]
    if not isinstance(external, Mapping) or not isinstance(data, Mapping):
        raise ValueError("Invalid Phase 2 config")
    output_dir = Path(str(data["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "openmathinstruct2_curriculum_v1.jsonl"
    manifest_path = output_dir / "openmathinstruct2_provenance.json"

    api_url = (
        f"https://huggingface.co/api/datasets/{external['dataset']}"
        f"/revision/{external['revision']}"
    )
    metadata = get_json(api_url)
    if metadata.get("sha") != external["revision"]:
        raise ValueError("OpenMathInstruct-2 revision changed; update config explicitly")
    tags = metadata.get("tags") or []
    if f"license:{str(external['license']).lower()}" not in tags:
        raise ValueError("Configured external dataset license was not confirmed")

    phase1_ids: set[str] = set()
    split_dir = Path(str(data["phase1_split_dir"]))
    for name in (
        "random_validation_ids.txt",
        "template_validation_ids.txt",
        "hard_diagnostic_ids.txt",
        "format_diagnostic_ids.txt",
    ):
        phase1_ids.update(
            line.strip()
            for line in (split_dir / name).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    import csv

    with Path(str(data["train_path"])).open("r", encoding="utf-8-sig", newline="") as handle:
        train_rows = list(csv.DictReader(handle))
    protected_rows = [
        {"id": row["id"], "question": row["question"]}
        for row in train_rows
    ]
    protected_rows.extend(load_leaderboard_questions(Path(str(data["leaderboard_path"]))))
    contamination = LocalContaminationIndex(
        protected_rows, float(data["near_duplicate_jaccard_threshold"])
    )

    dataset = str(external["dataset"])
    split = str(external["split"])
    viewer = str(external["viewer_url"])
    source_shard = str(external["source_shard"])
    sibling_names = {
        str(item.get("rfilename"))
        for item in metadata.get("siblings", [])
        if isinstance(item, Mapping)
    }
    if source_shard not in sibling_names:
        raise ValueError("Configured Parquet shard is absent from the pinned dataset revision")
    cache_dir = output_dir / "external_download_cache"
    source_url = (
        f"https://huggingface.co/datasets/{dataset}/resolve/"
        f"{external['revision']}/{urllib.parse.quote(source_shard, safe='/')}"
    )
    source_path = cache_dir / Path(source_shard).name
    download = download_file(source_url, source_path)
    parquet = import_parquet(cache_dir)
    parquet_file = parquet.ParquetFile(source_path)
    total_rows = int(parquet_file.metadata.num_rows)
    target = int(external["target_rows"])
    max_source = int(external["max_source_rows"])
    batch_size = 65536

    accepted: list[dict[str, object]] = []
    reasons: Counter[str] = Counter()
    problem_sources: Counter[str] = Counter()
    seen_exact: set[str] = set()
    seen_template: set[str] = set()
    sampled_digest = hashlib.sha256()
    sampled_rows = 0
    fetched_batches = 0
    contamination_examples: list[dict[str, object]] = []

    source_offset = 0
    columns = ["problem", "generated_solution", "expected_answer", "problem_source"]
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        fetched_batches += 1
        rows = batch.to_pylist()
        row_order = sorted(
            range(len(rows)),
            key=lambda index: stable_hash(
                "openmathinstruct2", config["seed"], source_offset + index
            ),
        )
        for index in row_order:
            if sampled_rows >= max_source or len(accepted) >= target:
                break
            source_index = source_offset + index
            row = rows[index]
            sampled_digest.update((json_dumps({"row_idx": source_index, "row": row}) + "\n").encode("utf-8"))
            sampled_rows += 1
            problem = str(row.get("problem", "")).strip()
            solution = str(row.get("generated_solution", "")).strip()
            expected = str(row.get("expected_answer", "")).strip()
            source = str(row.get("problem_source", "unknown"))
            normalized_answer = normalize_teacher_answer(expected)
            reason: str | None = None
            if not problem or not solution:
                reason = "empty_problem_or_solution"
            elif normalized_answer is None:
                reason = "non_single_numeric_answer"
            elif not 40 <= len(problem) <= 1200:
                reason = "question_length"
            elif not 35 <= len(solution.split()) <= 700:
                reason = "solution_length"
            elif english_ratio(problem + solution) < 0.98:
                reason = "non_english_or_abnormal_unicode"
            elif FORBIDDEN_RE.search(solution):
                reason = "tool_or_code_dependent"
            exact_key = exact_question_key(problem)
            template_key = normalize_template(problem)
            if reason is None and exact_key in seen_exact:
                reason = "external_internal_exact_duplicate"
            if reason is None and template_key in seen_template:
                reason = "external_internal_template_duplicate"
            match_type: str | None = None
            match_id: str | None = None
            match_score = 0.0
            if reason is None:
                match_type, match_id, match_score = contamination.match(problem)
                if match_type is not None:
                    reason = f"competition_{match_type}_duplicate"
                    if len(contamination_examples) < 100:
                        contamination_examples.append(
                            {
                                "external_row_idx": source_index,
                                "match_type": match_type,
                                "protected_id": match_id,
                                "score": round(match_score, 6),
                            }
                        )
            if reason is not None:
                reasons[reason] += 1
                continue
            seen_exact.add(exact_key)
            seen_template.add(template_key)
            problem_sources[source] += 1
            target_text = solution.rstrip() + f"\n\nFINAL_ANSWER: {normalized_answer}"
            accepted.append(
                {
                    "id": f"omi2-{source_index:09d}",
                    "messages": [
                        {"role": "user", "content": problem},
                        {"role": "assistant", "content": target_text},
                    ],
                    "problem": problem,
                    "solution": solution,
                    "final_answer": normalized_answer,
                    "grade": "external_public",
                    "sampling_weight": 1.0,
                    "provenance": {
                        "dataset": dataset,
                        "revision": external["revision"],
                        "split": split,
                        "source_row_idx": source_index,
                        "problem_source": source,
                        "license": external["license"],
                    },
                }
            )
        source_offset += len(rows)
        print(
            json.dumps(
                {
                    "event": "external_curriculum_progress",
                    "source_rows": sampled_rows,
                    "accepted_rows": len(accepted),
                    "fetched_batches": fetched_batches,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if len(accepted) >= target or sampled_rows >= max_source:
            break

    if len(accepted) < target:
        raise RuntimeError(
            f"Only {len(accepted)} external rows passed filters after {sampled_rows} source rows"
        )
    atomic_write_jsonl(output_path, accepted)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "dataset": dataset,
        "revision": external["revision"],
        "dataset_url": external["dataset_url"],
        "viewer_url": viewer,
        "source_download": download,
        "config": external["config"],
        "split": split,
        "license": external["license"],
        "license_verified_from_hugging_face_tags": True,
        "public_free_ungated": not bool(metadata.get("private")) and not bool(metadata.get("gated")),
        "retrieved_at_utc": utc_now(),
        "sampling": {
            "method": "deterministic hash-ranked rows within pinned Parquet batches",
            "seed": config["seed"],
            "batch_size": batch_size,
            "fetched_batches": fetched_batches,
            "source_rows_seen": sampled_rows,
            "source_row_stream_sha256": sampled_digest.hexdigest(),
            "source_shard_rows": total_rows,
        },
        "filters": {
            "single_numeric_answer": True,
            "english_text": True,
            "tool_and_code_mentions_removed": True,
            "length_limits": True,
            "internal_exact_and_template_deduplication": True,
        },
        "contamination": {
            "comparison_is_local_only": True,
            "protected_phase1_ids": len(phase1_ids),
            "competition_train_rows": len(train_rows),
            "leaderboard_original_rows": 1000,
            "method": "exact, normalized-template, normalized token-trigram Jaccard",
            "near_threshold": data["near_duplicate_jaccard_threshold"],
            "removed_examples_without_question_text": contamination_examples,
            "accepted_exact_or_template_matches": 0,
            "accepted_near_matches": 0,
        },
        "counts": {
            "source_rows_seen": sampled_rows,
            "accepted_rows": len(accepted),
            "removed_rows": sampled_rows - len(accepted),
            "removal_reasons": dict(sorted(reasons.items())),
            "problem_sources": dict(sorted(problem_sources.items())),
        },
        "output": {
            "path": str(output_path),
            "rows": len(accepted),
            "sha256": sha256_file(output_path),
            "bytes": output_path.stat().st_size,
        },
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase2.json"))
    return parser.parse_args()


def main() -> int:
    manifest = run(parse_args().config)
    print(json.dumps(manifest["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
