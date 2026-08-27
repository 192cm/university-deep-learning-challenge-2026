#!/usr/bin/env python3
"""Reproduce and decompose the T8 pass@32 versus plurality gap.

The script intentionally uses the repository's production answer extractor.
Ground truth is used only after candidate generation and vote selection have
been frozen.  All outputs are written beside this script.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPORT_DIR = Path(__file__).resolve().parent
ROOT = REPORT_DIR.parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import classify_problem_type, question_length_bucket  # noqa: E402
from src.extract import extract_answer, normalize_integer  # noqa: E402


GEN_PATH = ROOT / "artifacts/t8_self_consistency/generations.jsonl"
IDS_PATH = ROOT / "artifacts/t8_self_consistency/holdout_union_ids.txt"
SWEEP_PATH = ROOT / "artifacts/t8_self_consistency/sweep.json"
MANIFEST_PATH = ROOT / "artifacts/t8_self_consistency/manifest.json"
LABEL_PATH = ROOT / "data/canonical/train.csv"
FILTER_PATH = ROOT / "artifacts/t8_3_vote_filter/holdout/predictions.jsonl"

SPLIT_PATHS = {
    "random_holdout": ROOT / "data/splits/random_holdout.csv",
    "template_holdout": ROOT / "data/splits/template_holdout.csv",
    "hard_diagnostic": ROOT / "data/splits/hard_diagnostic.csv",
    "format_diagnostic": ROOT / "data/splits/format_diagnostic.csv",
}

EXAMPLE_IDS = {
    "train-012155",
    "train-003341",
    "train-008317",
    "train-003015",
    "train-007230",
    "train-013974",
    "train-008043",
}

EXAMPLE_NOTES: dict[str, dict[str, str]] = {
    "train-012155": {
        "mechanism": "파서·출력 형식 오류",
        "evidence": "25야드=75피트, 나무당 1.5피트이므로 50그루, 그루당 $8이라 정답은 $400이다. 32개 답 모두 계산은 400으로 끝났지만, ‘FINAL_ANSWER: $400.00’ 16개는 정수 전용 파서가 거부해 본문의 마지막 정수 50으로 후퇴했다.",
        "implication": "추론이 아니라 정답 표기의 계약 문제다. 금액·소수점 표기를 정규화하거나 약한 추출 경로를 제외하면 회수 가능하다.",
        "recommended_fix": "추출기 계약 보강 + FINAL_ANSWER 정수 형식 강제",
        "verification": "원시 32개 출력과 문제 산술을 직접 대조",
    },
    "train-003341": {
        "mechanism": "근접한 추론 분기",
        "evidence": "Jon의 총시간은 170분이고 10분 차로 이겼으므로 James는 180분이다. 수영 36분과 자전거 85분을 빼면 달리기는 59분이다. 우세 오답 39는 승패 방향을 뒤집어 James의 총시간을 160분으로 둔다.",
        "implication": "정답과 오답 모두 명시적 FINAL_ANSWER를 사용한다. 형식 필터로는 구별할 수 없고 문장 관계를 검증하는 scorer가 필요하다.",
        "recommended_fix": "쌍대 비교 verifier 또는 승패·시간관계 hard-negative SFT",
        "verification": "원시 출력의 두 추론 분기와 문제 산술을 직접 대조",
    },
    "train-008317": {
        "mechanism": "상관된 조합론 오개념",
        "evidence": "6권 중 3권만 동일하고 나머지 3권은 서로 다르므로 6!/3!=120이다. 우세 오답 20은 6!/(3!3!)를 써서 서로 다른 3권까지 동일한 묶음처럼 취급한다.",
        "implication": "23개 표가 같은 오개념에 몰려 있다. 단순 다수결은 독립 표본 가정을 만족하지 않는다.",
        "recommended_fix": "오개념 대조 데이터 + 의미 verifier",
        "verification": "T8-6 수동 검증 사례와 원시 표 분포 교차 확인",
    },
    "train-003015": {
        "mechanism": "희소 정답·계산 연쇄 오류",
        "evidence": "요구되는 수는 gcd(1255-8, 1490-11)=gcd(1247,1479)=29이다. 많은 출력이 gcd 설정까지는 맞지만 유클리드 알고리즘 산술에서 1로 잘못 수렴한다.",
        "implication": "정답은 4개뿐이고 오답 1은 24개다. top-2 reranking조차 완전한 계산 검증 없이는 어렵다.",
        "recommended_fix": "산술 검증형 verifier + 계산 오류 hard-negative SFT",
        "verification": "T8-6 수동 검증 사례와 원시 표 분포 교차 확인",
    },
    "train-007230": {
        "mechanism": "불필요 정보의 중복 차감",
        "evidence": "Dan은 97장을 갖고 있고 Sam이 15장을 사므로 남은 수는 82장이다. 찢어진 8장은 상태 설명인데 우세 오답은 8과 15를 모두 빼 74를 만든다.",
        "implication": "모든 숫자를 연산에 넣는 강한 편향이 26개 표에 반복된다. 출력 형식 가중치로는 해결되지 않는다.",
        "recommended_fix": "관련성 판별 대조학습 + 의미 verifier",
        "verification": "T8-6 수동 검증 사례와 원시 표 분포 교차 확인",
    },
    "train-013974": {
        "mechanism": "엄격 부등식 경계 오류",
        "evidence": "완전제곱을 만들면 (x-5)^2+4(y+7)^2=k+221이다. 비퇴화 조건은 k>-221이므로 최소 정수는 -220인데, 28개 출력이 등호 경계 -221을 택한다.",
        "implication": "유도식은 맞아도 마지막 논리 조건을 놓치는 체계적 오류다. 같은 모델 표본을 더 뽑아도 오답 상관이 유지된다.",
        "recommended_fix": "경계조건 검증 단계 + strict/non-strict 대조 SFT",
        "verification": "T8-6 수동 검증 사례와 원시 표 분포 교차 확인",
    },
    "train-008043": {
        "mechanism": "정답 미생성(대조군)",
        "evidence": "10개 교점을 K5의 변으로 보면, 5개를 골라 어떤 원래 직선에도 3개가 없으려면 각 꼭짓점 차수가 2인 5-cycle이어야 한다. 개수는 (5-1)!/2=12지만 32개 중 정답 표가 없다.",
        "implication": "선택기는 존재하지 않는 정답을 고를 수 없다. 이 583문항 구간은 생성 능력과 학습 데이터가 핵심이다.",
        "recommended_fix": "교사 생성·검증 데이터와 targeted SFT/RFT",
        "verification": "T8-6 수동 검증 사례와 원시 표 분포 교차 확인",
    },
}

TYPE_LABELS = {
    "algebra": "대수",
    "arithmetic_word_problem": "산술 문장제",
    "combinatorics_probability": "조합·확률",
    "geometry": "기하",
    "number_theory": "정수론",
}

LENGTH_LABELS = {
    "le128": "≤128자",
    "129_256": "129–256자",
    "257_512": "257–512자",
    "gt512": ">512자",
}

SPLIT_LABELS = {
    "random_holdout": "random",
    "template_holdout": "template",
    "hard_diagnostic": "hard",
    "format_diagnostic": "format",
}


def load_csv_by_id(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {str(row["id"]).strip(): row for row in csv.DictReader(handle)}


def load_jsonl_map(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["id"])] = row
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_pair(path: Path) -> tuple[str, str]:
    raw_digest = hashlib.sha256()
    normalized_digest = hashlib.sha256()
    with path.open("rb") as handle:
        for line in handle:
            raw_digest.update(line)
            normalized_digest.update(line.replace(b"\r\n", b"\n"))
    return raw_digest.hexdigest(), normalized_digest.hexdigest()


def count_band(value: int) -> str:
    if value <= 2:
        return "1–2표"
    if value <= 4:
        return "3–4표"
    if value <= 8:
        return "5–8표"
    return "9–16표"


def margin_band(value: int) -> str:
    if value == 0:
        return "동률"
    if value == 1:
        return "+1표"
    if value == 2:
        return "+2표"
    if value <= 4:
        return "+3–4표"
    if value <= 8:
        return "+5–8표"
    return "+9표 이상"


def segment_rows(
    diagnostics: list[dict[str, Any]],
    field: str,
    labels: dict[str, str],
    order: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, key in enumerate(order, start=1):
        group = [row for row in diagnostics if row[field] == key]
        n = len(group)
        majority = sum(int(row["majority_correct"]) for row in group)
        passed = sum(int(row["pass_at32"]) for row in group)
        selection = sum(int(row["selection_failure"]) for row in group)
        oracle = sum(int(row["oracle_failure"]) for row in group)
        rows.append(
            {
                "order": index,
                "segment": key,
                "segment_label": labels[key],
                "questions": n,
                "pass_count": passed,
                "majority_correct_count": majority,
                "selection_failure_count": selection,
                "oracle_failure_count": oracle,
                "pass_pct": round(100 * passed / n, 4),
                "majority_accuracy_pct": round(100 * majority / n, 4),
                "selection_gap_pp": round(100 * selection / n, 4),
                "oracle_failure_pct": round(100 * oracle / n, 4),
                "selection_failure_share_pct": round(100 * selection / 564, 4),
            }
        )
    return rows


def main() -> None:
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    labels = load_csv_by_id(LABEL_PATH)
    union_ids = [line.strip() for line in IDS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(union_ids) != 3737 or len(set(union_ids)) != 3737:
        raise AssertionError("Expected 3,737 unique frozen union IDs")
    union_set = set(union_ids)

    splits = {name: set(load_csv_by_id(path)) for name, path in SPLIT_PATHS.items()}
    filter_predictions = load_jsonl_map(FILTER_PATH)
    if set(filter_predictions) != union_set:
        raise AssertionError("T8-3 prediction IDs do not match the T8 union")

    candidates_by_id: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_examples: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    source_order = 0
    with GEN_PATH.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            row_id = str(raw["id"])
            if row_id not in union_set:
                raise AssertionError(f"Generation outside frozen union: {row_id}")
            output = str(raw["raw_generation"])
            extraction = extract_answer(output)
            compact = {
                "sample_index": int(raw["sample_index"]),
                "source_order": source_order,
                "answer": extraction.answer,
                "path": extraction.path,
                "output_tokens": int(raw["output_tokens"]),
                "hit_max": bool(raw["hit_max_new_tokens"]),
            }
            candidates_by_id[row_id].append(compact)
            if row_id in EXAMPLE_IDS:
                raw_examples[row_id].append({**compact, "raw_generation": output})
            source_order += 1

    if source_order != 119584:
        raise AssertionError(f"Expected 119,584 generations, found {source_order}")
    if set(candidates_by_id) != union_set:
        raise AssertionError("Generation IDs do not match the frozen union")

    diagnostics: list[dict[str, Any]] = []
    correct_path_counts: Counter[str] = Counter()
    winner_path_counts: Counter[str] = Counter()
    correct_tokens: list[int] = []
    winner_tokens: list[int] = []

    for row_id in union_ids:
        question = str(labels[row_id]["question"])
        gold = normalize_integer(str(labels[row_id]["answer"]))
        if gold is None:
            raise AssertionError(f"Non-integer label for {row_id}")
        candidates = sorted(
            candidates_by_id[row_id],
            key=lambda row: (row["sample_index"], row["source_order"]),
        )
        if len(candidates) != 32 or [row["sample_index"] for row in candidates] != list(range(32)):
            raise AssertionError(f"Expected sample indices 0..31 for {row_id}")

        answers = [row["answer"] for row in candidates]
        valid_answers = [answer for answer in answers if answer is not None]
        counts = Counter(valid_answers)
        if counts:
            selected, selected_votes = counts.most_common(1)[0]
            tied_top = sum(value == selected_votes for value in counts.values()) > 1
        else:
            selected, selected_votes, tied_top = None, 0, False
        correct_votes = counts.get(gold, 0)
        passed = correct_votes > 0
        majority_correct = selected == gold
        selection_failure = passed and not majority_correct
        oracle_failure = not passed
        # Competition rank: one plus the number of distinct answer candidates
        # with strictly more votes.  A correct answer tied for the top count is
        # rank 1; equal-count answers below the top occupy separate positions.
        correct_rank = (
            1 + sum(value > correct_votes for value in counts.values())
            if passed
            else None
        )
        winner_margin = selected_votes - correct_votes if selection_failure else None

        filtered = filter_predictions[row_id]
        filtered_answer = filtered.get("filtered_answer")
        filtered_correct = bool(filtered.get("filtered_correct"))

        if selection_failure:
            for candidate in candidates:
                if candidate["answer"] == gold:
                    correct_path_counts[candidate["path"]] += 1
                    correct_tokens.append(int(candidate["output_tokens"]))
                if candidate["answer"] == selected:
                    winner_path_counts[candidate["path"]] += 1
                    winner_tokens.append(int(candidate["output_tokens"]))

        correct_explicit = sum(
            1
            for row in candidates
            if row["answer"] == gold and row["path"] in {"final_answer_marker", "boxed"}
        )
        correct_weak = correct_votes - correct_explicit
        winner_explicit = sum(
            1
            for row in candidates
            if row["answer"] == selected and row["path"] in {"final_answer_marker", "boxed"}
        )
        winner_weak = selected_votes - winner_explicit

        record: dict[str, Any] = {
            "id": row_id,
            "question": question,
            "ground_truth": gold,
            "selected_answer": selected,
            "selected_votes": selected_votes,
            "correct_votes": correct_votes,
            "correct_rank": correct_rank,
            "winner_margin": winner_margin,
            "valid_candidates": len(valid_answers),
            "distinct_answers": len(counts),
            "tied_top": tied_top,
            "pass_at32": passed,
            "majority_correct": majority_correct,
            "selection_failure": selection_failure,
            "oracle_failure": oracle_failure,
            "outcome": (
                "majority_correct"
                if majority_correct
                else "selection_failure"
                if selection_failure
                else "oracle_failure"
            ),
            "problem_type": classify_problem_type(question),
            "question_length": len(question),
            "question_length_bucket": question_length_bucket(question),
            "filtered_answer": filtered_answer,
            "filtered_correct": filtered_correct,
            "filter_changed_answer": filtered_answer != selected,
            "correct_explicit_votes": correct_explicit,
            "correct_weak_votes": correct_weak,
            "winner_explicit_votes": winner_explicit,
            "winner_weak_votes": winner_weak,
            "vote_counts_json": json.dumps(counts, ensure_ascii=False, separators=(",", ":")),
        }
        for split_name, split_ids in splits.items():
            record[split_name] = row_id in split_ids
        diagnostics.append(record)

    majority_count = sum(int(row["majority_correct"]) for row in diagnostics)
    pass_count = sum(int(row["pass_at32"]) for row in diagnostics)
    selection_count = sum(int(row["selection_failure"]) for row in diagnostics)
    oracle_count = sum(int(row["oracle_failure"]) for row in diagnostics)
    if (majority_count, pass_count, selection_count, oracle_count) != (2590, 3154, 564, 583):
        raise AssertionError(
            "Metric reproduction mismatch: "
            f"{majority_count=}, {pass_count=}, {selection_count=}, {oracle_count=}"
        )

    sweep = json.loads(SWEEP_PATH.read_text(encoding="utf-8"))
    fixed32 = sweep["fixed_sweep"]["fixed_k32"]["metrics"]
    if fixed32["questions"] != 3737:
        raise AssertionError("Frozen sweep question count mismatch")
    if abs(fixed32["majority@k"] - majority_count / 3737) > 1e-15:
        raise AssertionError("Majority reproduction differs from frozen sweep")
    if abs(fixed32["pass@k"] - pass_count / 3737) > 1e-15:
        raise AssertionError("Pass@32 reproduction differs from frozen sweep")

    selection_rows = [row for row in diagnostics if row["selection_failure"]]
    support_order = ["1–2표", "3–4표", "5–8표", "9–16표"]
    support_counter = Counter(count_band(int(row["correct_votes"])) for row in selection_rows)
    support_rows = []
    for order, band in enumerate(support_order, start=1):
        group = [row for row in selection_rows if count_band(int(row["correct_votes"])) == band]
        recovered = sum(int(row["filtered_correct"]) for row in group)
        support_rows.append(
            {
                "order": order,
                "support_band": band,
                "selection_failures": support_counter[band],
                "share_pct": round(100 * support_counter[band] / selection_count, 4),
                "filter_recovered": recovered,
                "filter_recovery_pct": round(100 * recovered / len(group), 4),
            }
        )

    margin_order = ["동률", "+1표", "+2표", "+3–4표", "+5–8표", "+9표 이상"]
    margin_counter = Counter(margin_band(int(row["winner_margin"])) for row in selection_rows)
    margin_rows = [
        {
            "order": order,
            "margin_band": band,
            "selection_failures": margin_counter[band],
            "share_pct": round(100 * margin_counter[band] / selection_count, 4),
        }
        for order, band in enumerate(margin_order, start=1)
    ]

    rank_counter = Counter(int(row["correct_rank"]) for row in selection_rows)
    rank_rows = [
        {
            "correct_rank": rank,
            "selection_failures": rank_counter[rank],
            "share_pct": round(100 * rank_counter[rank] / selection_count, 4),
        }
        for rank in sorted(rank_counter)
    ]

    ceiling_rows = [
        {
            "policy": "현재 plurality",
            "rank_threshold": 0,
            "oracle_correct": majority_count,
            "oracle_accuracy_pct": round(100 * majority_count / 3737, 4),
            "gain_vs_majority_pp": 0.0,
        }
    ]
    for threshold in [1, 2, 3, 4, 5, 8, 10]:
        correct = majority_count + sum(
            1
            for row in selection_rows
            if int(row["correct_rank"]) <= threshold
        )
        ceiling_rows.append(
            {
                "policy": f"정답이 rank≤{threshold}이면 완벽 선택",
                "rank_threshold": threshold,
                "oracle_correct": correct,
                "oracle_accuracy_pct": round(100 * correct / 3737, 4),
                "gain_vs_majority_pp": round(100 * (correct - majority_count) / 3737, 4),
            }
        )

    outcome_rows = [
        {
            "order": 1,
            "outcome": "plurality 정답",
            "questions": majority_count,
            "share_pct": round(100 * majority_count / 3737, 4),
        },
        {
            "order": 2,
            "outcome": "정답 생성·선택 실패",
            "questions": selection_count,
            "share_pct": round(100 * selection_count / 3737, 4),
        },
        {
            "order": 3,
            "outcome": "정답 미생성",
            "questions": oracle_count,
            "share_pct": round(100 * oracle_count / 3737, 4),
        },
    ]

    type_rows = segment_rows(
        diagnostics,
        "problem_type",
        TYPE_LABELS,
        ["algebra", "arithmetic_word_problem", "combinatorics_probability", "geometry", "number_theory"],
    )
    length_rows = segment_rows(
        diagnostics,
        "question_length_bucket",
        LENGTH_LABELS,
        ["le128", "129_256", "257_512", "gt512"],
    )

    split_rows: list[dict[str, Any]] = []
    for order, split_name in enumerate(SPLIT_PATHS, start=1):
        group = [row for row in diagnostics if row[split_name]]
        n = len(group)
        majority = sum(int(row["majority_correct"]) for row in group)
        passed = sum(int(row["pass_at32"]) for row in group)
        selection = sum(int(row["selection_failure"]) for row in group)
        oracle = sum(int(row["oracle_failure"]) for row in group)
        split_rows.append(
            {
                "order": order,
                "split": split_name,
                "split_label": SPLIT_LABELS[split_name],
                "questions": n,
                "pass_count": passed,
                "majority_correct_count": majority,
                "selection_failure_count": selection,
                "oracle_failure_count": oracle,
                "pass_pct": round(100 * passed / n, 4),
                "majority_accuracy_pct": round(100 * majority / n, 4),
                "selection_gap_pp": round(100 * selection / n, 4),
                "oracle_failure_pct": round(100 * oracle / n, 4),
            }
        )

    explicit_paths = {"final_answer_marker", "boxed"}
    weak_paths = {"last_integer", "standalone_last_line"}
    path_rows: list[dict[str, Any]] = []
    for group_label, counter in [
        ("정답 후보", correct_path_counts),
        ("선택된 오답 후보", winner_path_counts),
    ]:
        total = sum(counter.values())
        for order, path in enumerate(
            ["final_answer_marker", "boxed", "standalone_last_line", "last_integer"],
            start=1,
        ):
            path_rows.append(
                {
                    "candidate_group": group_label,
                    "path_order": order,
                    "extraction_path": path,
                    "path_class": "명시적" if path in explicit_paths else "약한 경로",
                    "candidates": counter[path],
                    "share_pct": round(100 * counter[path] / total, 4),
                }
            )

    filtered_correct_count = sum(int(row["filtered_correct"]) for row in diagnostics)
    filter_rescues = sum(
        int(not row["majority_correct"] and row["filtered_correct"]) for row in diagnostics
    )
    filter_breaks = sum(
        int(row["majority_correct"] and not row["filtered_correct"]) for row in diagnostics
    )
    if (filtered_correct_count, filter_rescues, filter_breaks) != (2645, 69, 14):
        raise AssertionError("T8-3 paired comparison mismatch")

    all_correct_explicit = sum(
        int(row["correct_votes"] == row["correct_explicit_votes"]) for row in selection_rows
    )
    all_winner_weak = sum(
        int(row["winner_explicit_votes"] == 0) for row in selection_rows
    )
    majority_winner_weak = sum(
        int(row["winner_weak_votes"] >= row["selected_votes"] / 2) for row in selection_rows
    )
    parser_like = sum(
        int(
            row["winner_weak_votes"] >= row["selected_votes"] / 2
            and row["correct_explicit_votes"] >= row["correct_votes"] / 2
        )
        for row in selection_rows
    )
    diagnostic_flag_rows = [
        {
            "flag": "정답 후보가 모두 명시적 경로",
            "questions": all_correct_explicit,
            "share_pct": round(100 * all_correct_explicit / selection_count, 4),
            "interpretation": "정답 측 출력 형식은 이미 안정적",
        },
        {
            "flag": "우세 오답 지지가 전부 약한 추출 경로",
            "questions": all_winner_weak,
            "share_pct": round(100 * all_winner_weak / selection_count, 4),
            "interpretation": "파서/필터만으로 회수 가능성이 높은 하위집합",
        },
        {
            "flag": "우세 오답 지지의 절반 이상이 약한 경로",
            "questions": majority_winner_weak,
            "share_pct": round(100 * majority_winner_weak / selection_count, 4),
            "interpretation": "형식 품질 신호가 의미 있는 하위집합",
        },
        {
            "flag": "오답은 약한 경로 절반 이상·정답은 명시적 경로 절반 이상",
            "questions": parser_like,
            "share_pct": round(100 * parser_like / selection_count, 4),
            "interpretation": "형식 기반 선택이 유리한 보수적 proxy",
        },
    ]

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    hash_specs = [
        ("T8 generations", GEN_PATH, manifest["sources"]["full_generations"]["sha256"]),
        ("T8 sweep", SWEEP_PATH, manifest["outputs"]["sweep"]["sha256"]),
        ("T8 union IDs", IDS_PATH, manifest["sources"]["union_ids"]["sha256"]),
        ("canonical labels", LABEL_PATH, manifest["sources"]["canonical"]["sha256"]),
    ]
    for split_name, path in SPLIT_PATHS.items():
        hash_specs.append(
            (
                split_name,
                path,
                manifest["sources"]["splits"][split_name]["sha256"],
            )
        )
    hash_rows = []
    for label, path, expected in hash_specs:
        raw_hash, normalized_hash = sha256_pair(path)
        status = "exact" if raw_hash == expected else "newline-normalized" if normalized_hash == expected else "mismatch"
        if status == "mismatch":
            raise AssertionError(f"Frozen-source hash mismatch for {label}")
        hash_rows.append(
            {
                "source": label,
                "path": str(path.relative_to(ROOT)),
                "expected_sha256": expected,
                "raw_sha256": raw_hash,
                "normalized_sha256": normalized_hash,
                "status": status,
            }
        )

    example_rows: list[dict[str, Any]] = []
    example_json: list[dict[str, Any]] = []
    diagnostics_by_id = {row["id"]: row for row in diagnostics}
    for row_id in [
        "train-012155",
        "train-003341",
        "train-008317",
        "train-003015",
        "train-007230",
        "train-013974",
        "train-008043",
    ]:
        diagnostic = diagnostics_by_id[row_id]
        notes = EXAMPLE_NOTES[row_id]
        votes = json.loads(diagnostic["vote_counts_json"])
        ordered_votes = sorted(votes.items(), key=lambda item: (-int(item[1]), list(votes).index(item[0])))
        vote_summary = ", ".join(f"{answer}={count}" for answer, count in ordered_votes[:5])
        raw_candidates = sorted(raw_examples[row_id], key=lambda row: row["sample_index"])
        illustrative = []
        for target in [diagnostic["ground_truth"], diagnostic["selected_answer"]]:
            match = next((row for row in raw_candidates if row["answer"] == target), None)
            if match is None or any(item["sample_index"] == match["sample_index"] for item in illustrative):
                continue
            illustrative.append(
                {
                    "sample_index": match["sample_index"],
                    "extracted_answer": match["answer"],
                    "extraction_path": match["path"],
                    "raw_tail": match["raw_generation"][-700:],
                }
            )
        table_row = {
            "id": row_id,
            "mechanism": notes["mechanism"],
            "ground_truth": diagnostic["ground_truth"],
            "vote_summary": vote_summary,
            "correct_votes": diagnostic["correct_votes"],
            "winner_votes": diagnostic["selected_votes"],
            "evidence": notes["evidence"],
            "implication": notes["implication"],
            "recommended_fix": notes["recommended_fix"],
            "t8_3_result": (
                "회수"
                if diagnostic["filtered_correct"] and not diagnostic["majority_correct"]
                else "유지 정답"
                if diagnostic["filtered_correct"]
                else "미회수"
            ),
        }
        example_rows.append(table_row)
        example_json.append(
            {
                **table_row,
                "question": diagnostic["question"],
                "selected_answer": diagnostic["selected_answer"],
                "correct_rank": diagnostic["correct_rank"],
                "winner_margin": diagnostic["winner_margin"],
                "verification": notes["verification"],
                "illustrative_outputs": illustrative,
            }
        )

    summary = {
        "schema_version": 1,
        "generated_at_utc": generated_at,
        "scope": {"questions": 3737, "samples_per_question": 32, "generations": 119584},
        "reproduced_metrics": {
            "majority_correct": majority_count,
            "majority_accuracy": majority_count / 3737,
            "pass_count": pass_count,
            "pass_at32": pass_count / 3737,
            "selection_failures": selection_count,
            "selection_gap_pp": 100 * selection_count / 3737,
            "oracle_failures": oracle_count,
            "oracle_failure_pct": 100 * oracle_count / 3737,
            "selection_share_of_current_errors_pct": 100 * selection_count / (3737 - majority_count),
        },
        "selection_failure_severity": {
            "correct_support_1_to_4": sum(int(row["correct_votes"] <= 4) for row in selection_rows),
            "correct_support_1_to_4_pct": 100 * sum(int(row["correct_votes"] <= 4) for row in selection_rows) / selection_count,
            "tie_or_one_vote_behind": sum(int(row["winner_margin"] <= 1) for row in selection_rows),
            "tie_or_one_vote_behind_pct": 100 * sum(int(row["winner_margin"] <= 1) for row in selection_rows) / selection_count,
            "nine_or_more_votes_behind": sum(int(row["winner_margin"] >= 9) for row in selection_rows),
            "nine_or_more_votes_behind_pct": 100 * sum(int(row["winner_margin"] >= 9) for row in selection_rows) / selection_count,
        },
        "vote_filter_comparison": {
            "filtered_correct": filtered_correct_count,
            "filtered_accuracy": filtered_correct_count / 3737,
            "net_gain_questions": filtered_correct_count - majority_count,
            "net_gain_pp": 100 * (filtered_correct_count - majority_count) / 3737,
            "selection_failures_recovered": filter_rescues,
            "base_correct_broken": filter_breaks,
            "selection_failure_recovery_pct": 100 * filter_rescues / selection_count,
        },
        "extraction_path_diagnostic": {
            "correct_candidate_paths": dict(correct_path_counts),
            "winning_wrong_candidate_paths": dict(winner_path_counts),
            "all_correct_candidates_explicit_questions": all_correct_explicit,
            "all_winning_wrong_candidates_weak_questions": all_winner_weak,
            "majority_winning_wrong_support_weak_questions": majority_winner_weak,
            "parser_like_proxy_questions": parser_like,
            "parser_like_proxy_pct": 100 * parser_like / selection_count,
            "correct_output_tokens_median": statistics.median(correct_tokens),
            "correct_output_tokens_mean": statistics.mean(correct_tokens),
            "winning_wrong_output_tokens_median": statistics.median(winner_tokens),
            "winning_wrong_output_tokens_mean": statistics.mean(winner_tokens),
        },
        "validation": {
            "frozen_hashes_passed": True,
            "newline_normalization_required_for": [row["source"] for row in hash_rows if row["status"] == "newline-normalized"],
            "metric_assertions_passed": True,
        },
    }

    diagnostic_fields = [
        "id", "question", "ground_truth", "selected_answer", "selected_votes", "correct_votes",
        "correct_rank", "winner_margin", "valid_candidates", "distinct_answers", "tied_top",
        "pass_at32", "majority_correct", "selection_failure", "oracle_failure", "outcome",
        "problem_type", "question_length", "question_length_bucket", "filtered_answer",
        "filtered_correct", "filter_changed_answer", "correct_explicit_votes", "correct_weak_votes",
        "winner_explicit_votes", "winner_weak_votes", "random_holdout", "template_holdout",
        "hard_diagnostic", "format_diagnostic", "vote_counts_json",
    ]
    write_csv(REPORT_DIR / "question_diagnostics.csv", diagnostics, diagnostic_fields)
    write_csv(REPORT_DIR / "outcome_decomposition.csv", outcome_rows, list(outcome_rows[0]))
    write_csv(REPORT_DIR / "support_bands.csv", support_rows, list(support_rows[0]))
    write_csv(REPORT_DIR / "margin_bands.csv", margin_rows, list(margin_rows[0]))
    write_csv(REPORT_DIR / "correct_rank.csv", rank_rows, list(rank_rows[0]))
    write_csv(REPORT_DIR / "oracle_rank_ceiling.csv", ceiling_rows, list(ceiling_rows[0]))
    write_csv(REPORT_DIR / "problem_type_segments.csv", type_rows, list(type_rows[0]))
    write_csv(REPORT_DIR / "question_length_segments.csv", length_rows, list(length_rows[0]))
    write_csv(REPORT_DIR / "split_segments.csv", split_rows, list(split_rows[0]))
    write_csv(REPORT_DIR / "path_composition.csv", path_rows, list(path_rows[0]))
    write_csv(REPORT_DIR / "diagnostic_flags.csv", diagnostic_flag_rows, list(diagnostic_flag_rows[0]))
    write_csv(REPORT_DIR / "example_cases.csv", example_rows, list(example_rows[0]))
    write_csv(REPORT_DIR / "source_hash_validation.csv", hash_rows, list(hash_rows[0]))
    (REPORT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (REPORT_DIR / "example_cases.json").write_text(
        json.dumps({"schema_version": 1, "cases": example_json}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
