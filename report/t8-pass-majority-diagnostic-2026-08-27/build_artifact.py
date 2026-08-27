#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the portable report artifact from reproduced diagnostic outputs."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


REPORT_DIR = Path(__file__).resolve().parent
ROOT = REPORT_DIR.parents[1]


def coerce(value: str) -> Any:
    if value == "":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    if re.fullmatch(r"-?(?:0|[1-9][0-9]*)", value):
        return int(value)
    if re.fullmatch(r"-?(?:0|[1-9][0-9]*)\.[0-9]+", value):
        return float(value)
    return value


def read_csv(name: str) -> list[dict[str, Any]]:
    with (REPORT_DIR / name).open("r", encoding="utf-8", newline="") as handle:
        return [{key: coerce(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def source(id_: str, label: str, path: str) -> dict[str, str]:
    return {"id": id_, "label": label, "path": path}


def query_source(
    id_: str,
    path: str,
    language: str,
    description: str,
    tables_used: list[str],
    generated_at: str,
    filters: list[str] | None = None,
    sql: str | None = None,
    metric_definitions: list[str] | None = None,
) -> dict[str, Any]:
    query: dict[str, Any] = {
        "engine": "filesystem" if language != "python" else "python",
        "language": language,
        "description": description,
        "tables_used": tables_used,
        "executed_at": generated_at,
    }
    if filters:
        query["filters"] = filters
    if sql:
        query["sql"] = sql
    if metric_definitions:
        query["metric_definitions"] = metric_definitions
    return {"id": id_, "path": path, "query": query}


def main() -> None:
    summary = json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))
    generated_at = summary["generated_at_utc"]
    metrics = summary["reproduced_metrics"]
    severity = summary["selection_failure_severity"]
    filter_metrics = summary["vote_filter_comparison"]
    path_metrics = summary["extraction_path_diagnostic"]

    outcomes = read_csv("outcome_decomposition.csv")
    support_bands = read_csv("support_bands.csv")
    margin_bands = read_csv("margin_bands.csv")
    oracle_ceiling = read_csv("oracle_rank_ceiling.csv")
    type_segments = read_csv("problem_type_segments.csv")
    length_segments = read_csv("question_length_segments.csv")
    split_segments = read_csv("split_segments.csv")
    diagnostic_flags = read_csv("diagnostic_flags.csv")
    examples = read_csv("example_cases.csv")

    correct_paths = path_metrics["correct_candidate_paths"]
    wrong_paths = path_metrics["winning_wrong_candidate_paths"]
    correct_total = sum(correct_paths.values())
    wrong_total = sum(wrong_paths.values())
    correct_explicit = correct_paths.get("final_answer_marker", 0) + correct_paths.get("boxed", 0)
    wrong_explicit = wrong_paths.get("final_answer_marker", 0) + wrong_paths.get("boxed", 0)
    path_summary = [
        {
            "candidate_group": "정답 후보",
            "candidates": correct_total,
            "explicit_candidates": correct_explicit,
            "explicit_pct": round(100 * correct_explicit / correct_total, 2),
            "weak_candidates": correct_total - correct_explicit,
            "weak_pct": round(100 * (correct_total - correct_explicit) / correct_total, 2),
        },
        {
            "candidate_group": "선택된 오답 후보",
            "candidates": wrong_total,
            "explicit_candidates": wrong_explicit,
            "explicit_pct": round(100 * wrong_explicit / wrong_total, 2),
            "weak_candidates": wrong_total - wrong_explicit,
            "weak_pct": round(100 * (wrong_total - wrong_explicit) / wrong_total, 2),
        },
    ]

    strategy_comparison = [
        {
            "strategy": "T8 plurality@32",
            "correct": metrics["majority_correct"],
            "accuracy_pct": round(100 * metrics["majority_accuracy"], 4),
            "rescued_vs_base": 0,
            "broken_vs_base": 0,
            "net_gain": 0,
            "net_gain_pp": 0.0,
        },
        {
            "strategy": "T8-3 출력 품질 필터",
            "correct": filter_metrics["filtered_correct"],
            "accuracy_pct": round(100 * filter_metrics["filtered_accuracy"], 4),
            "rescued_vs_base": filter_metrics["selection_failures_recovered"],
            "broken_vs_base": filter_metrics["base_correct_broken"],
            "net_gain": filter_metrics["net_gain_questions"],
            "net_gain_pp": round(filter_metrics["net_gain_pp"], 4),
        },
    ]

    key_metrics = [
        {
            "majority_accuracy_pct": round(100 * metrics["majority_accuracy"], 2),
            "pass_at32_pct": round(100 * metrics["pass_at32"], 2),
            "selection_gap_pp": round(metrics["selection_gap_pp"], 2),
            "selection_share_of_errors_pct": round(metrics["selection_share_of_current_errors_pct"], 2),
            "filter_net_gain_pp": round(filter_metrics["net_gain_pp"], 2),
            "filter_gap_recovery_pct": round(filter_metrics["selection_failure_recovery_pct"], 2),
        }
    ]

    chart_sources = [
        source("t8_analysis", "T8 pass@32–plurality 재현 분석", "report/t8-pass-majority-diagnostic-2026-08-27/analyze.py"),
        source("t8_generations", "T8 원시 32-sample 생성", "artifacts/t8_self_consistency/generations.jsonl"),
        source("t8_sweep", "T8 frozen sweep metrics", "artifacts/t8_self_consistency/sweep.json"),
        source("labels_splits", "Canonical labels and diagnostic splits", "data/canonical/train.csv"),
        source("t8_3_filter", "T8-3 paired vote-filter predictions", "artifacts/t8_3_vote_filter/holdout/predictions.jsonl"),
        source("vote_code", "Production extraction and plurality implementation", "src/evaluate.py"),
        source("verified_examples", "수동 검증된 genuine-error 사례", "artifacts/t8_6_base_vote_policy/train_error_analysis/verified_genuine_error_examples.json"),
        source("method_notes", "지표 정의·검증·해석 guardrail", "report/t8-pass-majority-diagnostic-2026-08-27/source-notes.md"),
    ]

    manifest: dict[str, Any] = {
        "version": 1,
        "surface": "report",
        "title": "T8의 15%p 선택 격차는 얼마나 회수 가능한가",
        "description": "T8의 119,584개 원시 생성을 재파싱해 pass@32와 plurality 사이의 오류를 분해한 기술 진단",
        "generatedAt": generated_at,
        "filters": [],
        "cards": [
            {
                "id": "accuracy_card",
                "description": "같은 32개 후보에서의 실제 선택 성능과 gold oracle",
                "dataset": "key_metrics",
                "sourceId": "t8_analysis",
                "metrics": [
                    {"label": "plurality 정확도", "field": "majority_accuracy_pct", "format": "number", "unit": "%"},
                    {"label": "pass@32", "field": "pass_at32_pct", "format": "number", "unit": "%"},
                ],
            },
            {
                "id": "gap_card",
                "description": "정답 후보가 있었지만 현재 규칙이 틀린 구간",
                "dataset": "key_metrics",
                "sourceId": "t8_analysis",
                "metrics": [
                    {"label": "selection oracle gap", "field": "selection_gap_pp", "format": "number", "unit": "%p"},
                    {"label": "현재 오답 중 비중", "field": "selection_share_of_errors_pct", "format": "number", "unit": "%"},
                ],
            },
            {
                "id": "filter_card",
                "description": "동일 후보에 단순 출력 품질 필터를 적용한 실제 효과",
                "dataset": "key_metrics",
                "sourceId": "t8_3_filter",
                "metrics": [
                    {"label": "순정확도 개선", "field": "filter_net_gain_pp", "format": "number", "signed": True, "unit": "%p"},
                    {"label": "선택 실패 회수율", "field": "filter_gap_recovery_pct", "format": "number", "unit": "%"},
                ],
            },
        ],
        "charts": [
            {
                "id": "outcome_chart",
                "title": "현재 오답 1,147건은 선택 564건과 생성 583건으로 거의 반반이다",
                "subtitle": "전체 3,737문항 기준",
                "headerMarkdown": "**15.09%p는 선택 oracle**, **15.60%는 32번 안에 정답이 한 번도 없었던 생성 실패**입니다.",
                "intent": "comparison",
                "question": "T8의 전체 문항은 어떤 결과 상태로 나뉘는가?",
                "rationale": "세 결과 상태의 절대 규모와 전체 비중을 직접 비교합니다.",
                "comparisonContext": {"denominator": "3,737문항", "grain": "결과 상태", "unit": "문항"},
                "type": "horizontalBar",
                "dataset": "outcomes",
                "sourceId": "t8_analysis",
                "valueFormat": "number",
                "unit": "문항",
                "encodings": {
                    "x": {"field": "outcome", "type": "nominal", "label": "결과"},
                    "y": {"field": "questions", "type": "quantitative", "label": "문항 수", "unit": "문항"},
                    "tooltip": [{"field": "share_pct", "type": "quantitative", "label": "전체 비중", "format": "number", "unit": "%"}],
                },
                "labels": {"values": "all"},
                "palette": {"kind": "sequential"},
                "settings": {"orientation": "horizontal", "showValues": True, "categoryLabelPolicy": "wrap"},
                "layout": "full",
            },
            {
                "id": "support_chart",
                "title": "선택 실패의 67%는 정답 후보가 1–4개뿐이다",
                "subtitle": "selection failure 564문항에서 정답 표 수",
                "headerMarkdown": f"정답이 희소한 1–4표 구간이 **{severity['correct_support_1_to_4']}건**입니다. 출력 품질 필터의 회수율도 정답 지지가 클수록 4.7%→30.9%로 올라갑니다.",
                "intent": "comparison",
                "question": "선택 실패에서 정답은 몇 표를 받았는가?",
                "rationale": "정답 지지 강도별 문항 수와 필터 회수율을 비교합니다.",
                "comparisonContext": {"denominator": "selection failure 564문항", "grain": "정답 표 band", "unit": "문항"},
                "type": "horizontalBar",
                "dataset": "support_bands",
                "sourceId": "t8_analysis",
                "valueFormat": "number",
                "unit": "문항",
                "encodings": {
                    "x": {"field": "support_band", "type": "nominal", "label": "정답 표 수"},
                    "y": {"field": "selection_failures", "type": "quantitative", "label": "선택 실패", "unit": "문항"},
                    "tooltip": [
                        {"field": "share_pct", "type": "quantitative", "label": "선택 실패 내 비중", "format": "number", "unit": "%"},
                        {"field": "filter_recovery_pct", "type": "quantitative", "label": "T8-3 회수율", "format": "number", "unit": "%"},
                    ],
                },
                "labels": {"values": "all"},
                "palette": {"kind": "sequential"},
                "settings": {"orientation": "horizontal", "showValues": True, "categoryLabelPolicy": "wrap"},
                "layout": "full",
            },
            {
                "id": "margin_chart",
                "title": "동률·1표 차이는 11.5%뿐이고, 41%는 9표 이상 뒤진다",
                "subtitle": "우세 오답 표 수 − 정답 표 수, selection failure 564문항",
                "headerMarkdown": f"작은 tie-break 변경이 직접 겨냥할 수 있는 동률·1표 차이는 **{severity['tie_or_one_vote_behind']}건**인 반면, **{severity['nine_or_more_votes_behind']}건**은 9표 이상 열세입니다.",
                "intent": "comparison",
                "question": "우세 오답과 정답 후보의 표 차이는 얼마나 큰가?",
                "rationale": "근소한 선택 오류와 상관된 대규모 오답 모드를 분리합니다.",
                "comparisonContext": {"denominator": "selection failure 564문항", "grain": "표 차이 band", "unit": "문항"},
                "type": "horizontalBar",
                "dataset": "margin_bands",
                "sourceId": "t8_analysis",
                "valueFormat": "number",
                "unit": "문항",
                "encodings": {
                    "x": {"field": "margin_band", "type": "nominal", "label": "우세 오답의 표 차이"},
                    "y": {"field": "selection_failures", "type": "quantitative", "label": "선택 실패", "unit": "문항"},
                    "tooltip": [{"field": "share_pct", "type": "quantitative", "label": "선택 실패 내 비중", "format": "number", "unit": "%"}],
                },
                "labels": {"values": "all"},
                "palette": {"kind": "sequential"},
                "settings": {"orientation": "horizontal", "showValues": True, "categoryLabelPolicy": "wrap"},
                "layout": "full",
            },
            {
                "id": "type_gap_chart",
                "title": "기하·정수론의 선택 격차는 21%p를 넘지만, 건수는 산술 문장제가 가장 많다",
                "subtitle": "문제 유형별 pass@32 − plurality accuracy",
                "headerMarkdown": "기하는 **21.68%p**, 정수론은 **21.57%p**입니다. 다만 산술 문장제가 전체 선택 실패의 **58.5%(330건)**를 차지합니다.",
                "intent": "comparison",
                "question": "어떤 문제 유형에서 선택 격차가 크며, 전체 오류 기여는 얼마나 되는가?",
                "rationale": "유형별 오류율과 절대 오류 기여를 함께 비교해 개선 우선순위를 정합니다.",
                "comparisonContext": {"denominator": "유형별 문항 수", "grain": "문제 유형", "unit": "%p"},
                "type": "horizontalBar",
                "dataset": "type_segments",
                "sourceId": "t8_analysis",
                "valueFormat": "number",
                "unit": "%p",
                "encodings": {
                    "x": {"field": "segment_label", "type": "nominal", "label": "문제 유형"},
                    "y": {"field": "selection_gap_pp", "type": "quantitative", "label": "선택 격차", "unit": "%p"},
                    "tooltip": [
                        {"field": "selection_failure_count", "type": "quantitative", "label": "선택 실패", "format": "number", "unit": "문항"},
                        {"field": "selection_failure_share_pct", "type": "quantitative", "label": "전체 선택 실패 기여", "format": "number", "unit": "%"},
                        {"field": "oracle_failure_pct", "type": "quantitative", "label": "정답 미생성률", "format": "number", "unit": "%"},
                    ],
                },
                "labels": {"values": "all"},
                "palette": {"kind": "sequential"},
                "settings": {"orientation": "horizontal", "sort": "descending", "showValues": True, "categoryLabelPolicy": "wrap"},
                "layout": "full",
            },
        ],
        "tables": [
            {
                "id": "oracle_ceiling_table",
                "title": "완벽한 의미 verifier가 있을 때의 이론적 상한",
                "subtitle": "정답 후보의 vote-rank까지 완벽하게 식별한다고 가정",
                "dataset": "oracle_ceiling",
                "sourceId": "t8_analysis",
                "density": "spacious",
                "columns": [
                    {"field": "policy", "label": "가정", "type": "text"},
                    {"field": "oracle_correct", "label": "정답", "type": "number"},
                    {"field": "oracle_accuracy_pct", "label": "정확도 (%)", "type": "number"},
                    {"field": "gain_vs_majority_pp", "label": "현재 대비 (%p)", "type": "number"},
                ],
                "layout": "full",
            },
            {
                "id": "path_summary_table",
                "title": "오답 쪽에 약한 추출 경로가 더 많지만 대부분은 명시적 답이다",
                "subtitle": "564개 선택 실패 안의 후보-level 경로 구성",
                "dataset": "path_summary",
                "sourceId": "t8_analysis",
                "density": "spacious",
                "columns": [
                    {"field": "candidate_group", "label": "후보 집합", "type": "text"},
                    {"field": "candidates", "label": "후보 수", "type": "number"},
                    {"field": "explicit_candidates", "label": "명시적 final/boxed", "type": "number"},
                    {"field": "explicit_pct", "label": "명시적 비중 (%)", "type": "number"},
                    {"field": "weak_candidates", "label": "약한 추출 경로", "type": "number"},
                    {"field": "weak_pct", "label": "약한 경로 비중 (%)", "type": "number"},
                ],
                "layout": "full",
            },
            {
                "id": "filter_comparison_table",
                "title": "단순 출력 품질 필터는 oracle gap의 12.2%만 회수했다",
                "subtitle": "동일한 32개 후보에 대한 paired 결과",
                "dataset": "strategy_comparison",
                "sourceId": "t8_3_filter",
                "density": "spacious",
                "columns": [
                    {"field": "strategy", "label": "정책", "type": "text"},
                    {"field": "correct", "label": "정답", "type": "number"},
                    {"field": "accuracy_pct", "label": "정확도 (%)", "type": "number"},
                    {"field": "rescued_vs_base", "label": "회수", "type": "number"},
                    {"field": "broken_vs_base", "label": "파손", "type": "number"},
                    {"field": "net_gain", "label": "순증", "type": "number"},
                    {"field": "net_gain_pp", "label": "순증 (%p)", "type": "number"},
                ],
                "layout": "full",
            },
            {
                "id": "example_table",
                "title": "실제 32개 출력에서 확인한 대표 사례",
                "subtitle": "형식 오류 1건, 상관된 추론 오류 5건, 정답 미생성 1건",
                "dataset": "examples",
                "sourceId": "verified_examples",
                "density": "spacious",
                "columns": [
                    {"field": "id", "label": "ID", "type": "text"},
                    {"field": "mechanism", "label": "메커니즘", "type": "text"},
                    {"field": "vote_summary", "label": "상위 표 분포", "type": "text"},
                    {"field": "evidence", "label": "정답과 오답 분기", "type": "text"},
                    {"field": "implication", "label": "왜 투표만으로 어려운가", "type": "text"},
                    {"field": "recommended_fix", "label": "맞는 처방", "type": "text"},
                    {"field": "t8_3_result", "label": "T8-3", "type": "text"},
                ],
                "layout": "full",
            },
            {
                "id": "length_table",
                "title": "문제가 길어질수록 선택과 생성이 동시에 악화된다",
                "subtitle": "질문 문자열 길이 bucket",
                "dataset": "length_segments",
                "sourceId": "t8_analysis",
                "density": "spacious",
                "columns": [
                    {"field": "segment_label", "label": "길이", "type": "text"},
                    {"field": "questions", "label": "문항", "type": "number"},
                    {"field": "pass_pct", "label": "pass@32 (%)", "type": "number"},
                    {"field": "majority_accuracy_pct", "label": "plurality (%)", "type": "number"},
                    {"field": "selection_gap_pp", "label": "선택 격차 (%p)", "type": "number"},
                    {"field": "oracle_failure_pct", "label": "정답 미생성 (%)", "type": "number"},
                ],
                "layout": "full",
            },
        ],
        "sources": chart_sources,
        "blocks": [
            {
                "id": "title",
                "type": "markdown",
                "body": "# T8의 15%p 선택 격차는 얼마나 회수 가능한가\n\n2026-08-27 · frozen union 3,737문항 · 32 samples/question",
            },
            {
                "id": "technical_summary",
                "type": "markdown",
                "sourceId": "t8_analysis",
                "body": "## Technical Summary\n\n**‘정답을 이미 만들었는데 잘못 골랐다’는 진술은 정확하지만, 그 15.09%p가 곧바로 투표 규칙의 실현 가능한 개선 폭이라는 뜻은 아닙니다.** 현재 오답 1,147건은 선택 실패 564건과 정답 미생성 583건으로 거의 정확히 반반입니다. 선택 실패 중 동률 또는 1표 차이는 65건뿐이고, 378건은 정답이 1–4번만 등장했으며, 231건은 우세 오답보다 9표 이상 뒤졌습니다.\n\n따라서 개선은 두 갈래여야 합니다. **파서·출력 품질 정책은 확인된 1–2%p 구간을 먼저 회수**하고, 나머지 상관된 추론 오류는 **의미 verifier와 targeted SFT/teacher 데이터**로 다뤄야 합니다. pass@32는 gold를 아는 oracle이므로 실제 selector 목표치로 그대로 사용할 수 없습니다.",
            },
            {"id": "metrics", "type": "metric-strip", "cardIds": ["accuracy_card", "gap_card", "filter_card"]},
            {
                "id": "error_budget_text",
                "type": "markdown",
                "sourceId": "t8_analysis",
                "body": "## 1. 오류 예산은 선택과 생성으로 반씩 갈립니다\n\nplurality가 맞힌 2,590건을 제외한 1,147건 중 **49.17%는 선택 실패**, **50.83%는 32회 모두 정답이 없었던 생성 실패**입니다. 즉 selector만 개선하면 절반의 오류에는 손도 대지 못합니다.",
            },
            {"id": "outcome", "type": "chart", "chartId": "outcome_chart"},
            {
                "id": "headroom_text",
                "type": "markdown",
                "sourceId": "t8_analysis",
                "body": "## 2. 15.09%p의 대부분은 단순 tie-break 문제가 아닙니다\n\n정답이 우세 오답과 동률인 문항은 29건뿐입니다. 완벽한 tie-break로도 69.31%→70.08%, **+0.78%p**가 상한입니다. 완벽한 top-2 의미 verifier를 가정하면 76.85%까지 갈 수 있지만, 이는 표 수가 아니라 실제 풀이의 옳고 그름을 판정하는 별도 능력을 가정한 값입니다.",
            },
            {"id": "support", "type": "chart", "chartId": "support_chart"},
            {"id": "margin", "type": "chart", "chartId": "margin_chart"},
            {"id": "oracle_table", "type": "table", "tableId": "oracle_ceiling_table"},
            {
                "id": "format_text",
                "type": "markdown",
                "sourceId": "t8_3_filter",
                "body": "## 3. 출력 형식은 실제 병목이지만 전체 병목은 아닙니다\n\n선택 실패에서 정답 후보의 89.91%는 명시적 `FINAL_ANSWER` 또는 `\\boxed{}` 경로였고, 선택된 오답 지지의 73.71%도 명시적이었습니다. 오답 쪽 약한 추출 경로가 26.29%로 더 많아 품질 신호는 분명하지만, **대부분의 오답도 형식상 정상**입니다. 실제 T8-3 필터는 69건을 살리고 14건을 망쳐 순 +55건, **+1.47%p**를 만들었습니다.",
            },
            {"id": "path_table", "type": "table", "tableId": "path_summary_table"},
            {"id": "filter_table", "type": "table", "tableId": "filter_comparison_table"},
            {
                "id": "parser_example",
                "type": "markdown",
                "sourceId": "t8_generations",
                "body": "### 사례 A — `train-012155`: 모델은 32번 모두 400을 계산했는데 파서가 16번을 50으로 바꿨습니다\n\n25야드는 75피트이고 1.5피트 간격이면 50그루, 그루당 $8이므로 $400입니다. 16개 출력은 `FINAL_ANSWER: 400`이라 400으로 추출됐습니다. 나머지 16개도 풀이 결론은 $400이지만 `FINAL_ANSWER: $400.00`으로 썼습니다. 정수 전용 explicit parser가 이를 거부한 뒤 본문의 마지막 정수 50으로 fallback하여 **50=16, 400=16 동률**이 되었고, 먼저 등장한 50이 선택됐습니다. T8-3는 약한 경로 16개를 제외해 이 문항을 회수했습니다.\n\n이 사례는 **SFT보다 추출기·출력 계약을 먼저 고쳐야 하는 명백한 구간**입니다.",
            },
            {
                "id": "semantic_examples_text",
                "type": "markdown",
                "sourceId": "verified_examples",
                "body": "## 4. 다수는 같은 오개념이 반복된 ‘상관된 오답’입니다\n\n아래 사례는 저장 라벨과 수학적 풀이를 수동 대조했습니다. `train-003341`은 1표 차이지만 양쪽 모두 정상적인 final marker를 사용하고, `train-008317`, `train-007230`, `train-013974`는 우세 오답이 23–28표에 달합니다. 이런 경우 표의 개수는 확신처럼 보이지만, 실제로는 **같은 모델의 같은 편향이 반복된 횟수**입니다.",
            },
            {"id": "examples", "type": "table", "tableId": "example_table"},
            {
                "id": "segments_text",
                "type": "markdown",
                "sourceId": "t8_analysis",
                "body": "## 5. semantic selector와 SFT의 우선순위는 다릅니다\n\n선택 격차율은 기하·정수론이 가장 높지만, 전체 선택 실패 건수의 58.5%는 데이터량이 큰 산술 문장제입니다. 또 질문이 512자를 넘으면 선택 격차가 25.57%p, 정답 미생성률이 37.02%까지 동시에 상승합니다. **selector는 산술 문장제의 대량 오류와 format을 먼저**, **SFT는 긴 문제·기하·정수론·조합론의 체계적 추론 실패를 먼저** 겨냥하는 편이 합리적입니다.",
            },
            {"id": "type_gap", "type": "chart", "chartId": "type_gap_chart"},
            {"id": "length", "type": "table", "tableId": "length_table"},
            {
                "id": "what_to_do",
                "type": "markdown",
                "sourceId": "method_notes",
                "body": "## Recommended Next Experiments\n\n1. **파서/포맷 패치부터 분리 평가합니다.** 통화기호와 `.00`을 안전하게 정수로 정규화하고, conflicting explicit answer와 weak fallback을 보수적으로 처리합니다. 같은 frozen 32-pool에 재적용해 순증과 파손을 paired로 측정합니다.\n2. **선생은 ‘정답 생성기’보다 ‘두 풀이 중 무엇이 틀렸는지 설명하는 verifier’로 설계합니다.** 비holdout 문제에서 정답 풀이와 우세 오답 풀이를 쌍으로 만들고, 관계 반전·불필요 숫자·중복 조합 나눗셈·엄격 경계·Euclidean 산술 같은 오류 태그를 붙입니다.\n3. **selector 학습과 generator SFT를 분리합니다.** verifier/reranker는 이미 존재하는 정답을 고르는 564건 유형을, targeted SFT/RFT는 상관된 오답과 583건의 no-correct 유형을 개선해야 합니다.\n4. **이 3,737문항은 학습에 넣지 않습니다.** 여기서는 오류 패턴만 정의하고, 별도 train pool에서 유사 문항을 수확·검증합니다. 최종 평가는 frozen union과 fresh holdout에서 각각 한 번씩 수행합니다.\n5. 채택 판단은 `accuracy`, `rescued`, `broken`, hard/format guardrail을 함께 봅니다. 목표를 pass@32에 두지 말고, label-blind end-to-end 정확도 개선에 둡니다.",
            },
            {
                "id": "bottom_line",
                "type": "markdown",
                "sourceId": "method_notes",
                "body": "## Bottom Line\n\n**투표·선택 정책 개선은 맞는 첫 단계지만, 그것만으로 15%p를 얻는 것은 불가능에 가깝습니다.** 현재 데이터가 보여주는 보수적 실증치는 단순 품질 필터의 +1.47%p입니다. 나머지 큰 구간에는 문제를 실제로 다시 판단하는 semantic verifier가 필요하고, 정답 자체가 없거나 같은 오개념이 압도하는 사례에는 targeted SFT/teacher 데이터가 필요합니다. 즉 ‘선생을 얻는다’는 방향은 맞되, 한 모델에게 모든 역할을 맡기기보다 **파서 → verifier → generator SFT**의 세 층으로 분리하는 것이 핵심입니다.",
            },
            {
                "id": "caveats",
                "type": "markdown",
                "sourceId": "method_notes",
                "body": "## Caveats and Assumptions\n\n- `pass@32`와 selection failure는 stored gold label을 사용하는 사후 oracle입니다. 테스트 시 그대로 계산할 수 없습니다.\n- visible 사례는 수동 검증한 illustration이며, 전체 3,737 라벨의 수동 감사 결과가 아닙니다.\n- 네 진단 split은 서로 겹치므로 split별 오류 수를 합산하면 안 됩니다.\n- T8-3 비교는 동일 후보 pool의 paired 분석이라 선택 정책 효과를 깨끗하게 보여주지만, 다른 생성 모델이나 prompt로 일반화된다는 보장은 없습니다.\n- 반복적인 holdout 튜닝을 피하기 위해 다음 정책은 사전등록하고 fresh holdout으로 확인해야 합니다.",
            },
            {
                "id": "further_questions",
                "type": "markdown",
                "sourceId": "method_notes",
                "body": "## Further Questions\n\n- 통화·소수점 정규화와 explicit-answer 충돌 처리만으로 T8-3의 +1.47%p를 얼마나 넘어설 수 있는가?\n- top-2 pairwise verifier가 69개의 기존 회수 건 외에 얼마나 더 살리며, 기존 정답을 몇 건 파손하는가?\n- 오류 유형별 targeted SFT가 `pass@32`뿐 아니라 plurality와 greedy를 함께 높이는가?\n- 긴 문제에서 hit-max, 문맥 누락, 관계 추적 중 무엇이 selection과 generation을 각각 지배하는가?",
            },
        ],
    }

    datasets = {
        "key_metrics": key_metrics,
        "outcomes": outcomes,
        "support_bands": support_bands,
        "margin_bands": margin_bands,
        "oracle_ceiling": oracle_ceiling,
        "type_segments": type_segments,
        "length_segments": length_segments,
        "split_segments": split_segments,
        "diagnostic_flags": diagnostic_flags,
        "path_summary": path_summary,
        "strategy_comparison": strategy_comparison,
        "examples": examples,
    }

    artifact_sources = [
        query_source(
            "t8_analysis",
            "report/t8-pass-majority-diagnostic-2026-08-27/analyze.py",
            "python",
            "Production extractor로 119,584개 생성을 재파싱하고 question-level outcome, vote support, margin, rank, extraction path, segment를 집계했습니다.",
            [
                "artifacts/t8_self_consistency/generations.jsonl",
                "artifacts/t8_self_consistency/holdout_union_ids.txt",
                "data/canonical/train.csv",
                "data/splits/random_holdout.csv",
                "data/splits/template_holdout.csv",
                "data/splits/hard_diagnostic.csv",
                "data/splits/format_diagnostic.csv",
                "artifacts/t8_3_vote_filter/holdout/predictions.jsonl",
            ],
            generated_at,
            ["frozen union IDs", "k=32", "exact integer match", "labels attached after prediction freeze"],
            """SELECT * FROM read_csv_auto('report/t8-pass-majority-diagnostic-2026-08-27/outcome_decomposition.csv');
SELECT * FROM read_csv_auto('report/t8-pass-majority-diagnostic-2026-08-27/support_bands.csv');
SELECT * FROM read_csv_auto('report/t8-pass-majority-diagnostic-2026-08-27/margin_bands.csv');
SELECT * FROM read_csv_auto('report/t8-pass-majority-diagnostic-2026-08-27/oracle_rank_ceiling.csv');
SELECT * FROM read_csv_auto('report/t8-pass-majority-diagnostic-2026-08-27/problem_type_segments.csv');
SELECT * FROM read_csv_auto('report/t8-pass-majority-diagnostic-2026-08-27/question_length_segments.csv');""",
            [
                "plurality accuracy = plurality-selected answer가 stored label과 일치한 문항 수 / 3,737",
                "pass@32 = 32개 extracted answer 중 stored label이 하나 이상 존재한 문항 수 / 3,737",
                "selection gap (%p) = 100 × (pass count − plurality-correct count) / 3,737",
                "selection share of errors = selection failures / (3,737 − plurality-correct count)",
            ],
        ),
        query_source(
            "t8_generations",
            "artifacts/t8_self_consistency/generations.jsonl",
            "jsonl",
            "T8 base model의 3,737×32 원시 생성, sample index, token 수, 종료 상태를 사용했습니다.",
            ["artifacts/t8_self_consistency/generations.jsonl"],
            generated_at,
        ),
        query_source(
            "t8_sweep",
            "artifacts/t8_self_consistency/sweep.json",
            "json",
            "재현한 plurality@32와 pass@32를 frozen fixed-k32 metric에 assert했습니다.",
            ["artifacts/t8_self_consistency/sweep.json"],
            generated_at,
        ),
        query_source(
            "labels_splits",
            "data/canonical/train.csv",
            "csv",
            "저장 정답과 질문 텍스트, 네 진단 split membership을 결합했습니다.",
            ["data/canonical/train.csv", *[f"data/splits/{name}.csv" for name in ["random_holdout", "template_holdout", "hard_diagnostic", "format_diagnostic"]]],
            generated_at,
        ),
        query_source(
            "t8_3_filter",
            "artifacts/t8_3_vote_filter/holdout/predictions.jsonl",
            "jsonl",
            "동일 32-candidate pool에서 frozen T8-3 품질 필터의 rescue/break/net effect를 paired 비교했습니다.",
            ["artifacts/t8_3_vote_filter/holdout/predictions.jsonl", "artifacts/t8_3_vote_filter/final_config.json"],
            generated_at,
            sql="""WITH paired AS (
  SELECT id, unfiltered_correct, filtered_correct
  FROM read_json_auto('artifacts/t8_3_vote_filter/holdout/predictions.jsonl')
)
SELECT
  SUM(CASE WHEN filtered_correct THEN 1 ELSE 0 END) AS filtered_correct,
  SUM(CASE WHEN NOT unfiltered_correct AND filtered_correct THEN 1 ELSE 0 END) AS rescued,
  SUM(CASE WHEN unfiltered_correct AND NOT filtered_correct THEN 1 ELSE 0 END) AS broken,
  rescued - broken AS net_gain
FROM paired;""",
            metric_definitions=[
                "rescued = base plurality 오답이 T8-3 filtered prediction에서 정답으로 바뀐 문항 수",
                "broken = base plurality 정답이 T8-3 filtered prediction에서 오답으로 바뀐 문항 수",
                "net gain = rescued − broken; net gain (%p) = 100 × net gain / 3,737",
            ],
        ),
        query_source(
            "vote_code",
            "src/evaluate.py",
            "python",
            "plurality tie-break와 문제 유형·길이 분류 규칙을 확인하고, 답 추출에는 src/extract.py를 직접 호출했습니다.",
            ["src/evaluate.py", "src/extract.py"],
            generated_at,
        ),
        query_source(
            "verified_examples",
            "artifacts/t8_6_base_vote_policy/train_error_analysis/verified_genuine_error_examples.json",
            "json",
            "기존 수동 검증 사례를 원시 T8 표 분포와 교차 확인하고, 추가 두 사례를 직접 산술 검증했습니다.",
            ["artifacts/t8_6_base_vote_policy/train_error_analysis/verified_genuine_error_examples.json", "report/t8-pass-majority-diagnostic-2026-08-27/example_cases.json"],
            generated_at,
            sql="SELECT * FROM read_csv_auto('report/t8-pass-majority-diagnostic-2026-08-27/example_cases.csv');",
        ),
        query_source(
            "method_notes",
            "report/t8-pass-majority-diagnostic-2026-08-27/source-notes.md",
            "markdown",
            "지표 정의, frozen hash 검증, chart contract, 해석 guardrail과 leakage 방지 원칙을 기록했습니다.",
            ["report/t8-pass-majority-diagnostic-2026-08-27/source-notes.md", "report/t8-pass-majority-diagnostic-2026-08-27/source_hash_validation.csv"],
            generated_at,
        ),
    ]

    artifact = {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": datasets,
        },
        "sources": artifact_sources,
    }
    (REPORT_DIR / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {REPORT_DIR / 'artifact.json'}")


if __name__ == "__main__":
    main()
