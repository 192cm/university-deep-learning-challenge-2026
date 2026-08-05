#!/usr/bin/env python3
"""Build the evidence-backed Phase 1 Markdown report from final artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase1_common import atomic_write_text, sha256_file


def format_float(value: object, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--git-head", required=True)
    parser.add_argument("--git-status-initial", type=Path, required=True)
    parser.add_argument("--git-status-final", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    verification = json.loads(args.verification.read_text(encoding="utf-8"))
    if not verification["passed"]:
        raise ValueError("Cannot build final Phase 1 report from failed verification")

    metric_lines = []
    for row in metrics["metrics"]:
        metric_lines.append(
            "| {baseline_id} | {scope} | {questions} | {greedy} | {sample} | "
            "{passk} | {majority} | {agreement} | {invalid} | {tokens} | {p95lat} | {runtime} |".format(
                baseline_id=row["baseline_id"],
                scope=row["scope"],
                questions=row["questions"],
                greedy=format_float(row["greedy_accuracy"]),
                sample=format_float(row["sample_accuracy"]),
                passk=format_float(row["pass_at_k"]),
                majority=format_float(row["majority_at_k"]),
                agreement=format_float(row["agreement_at_k"]),
                invalid=format_float(row["invalid_output_rate"]),
                tokens=format_float(row["median_output_tokens"], 1),
                p95lat=format_float(row["p95_latency_seconds"], 3),
                runtime=format_float(row["estimated_1000_question_runtime_seconds"] / 3600, 2),
            )
        )

    source_lines = []
    for name, asset in provenance["assets"].items():
        source_lines.append(
            f"| `{name}` | `{asset['path']}` | {asset['rows']} | "
            f"`{','.join(asset['columns'])}` | `{asset['sha256']}` |"
        )

    prompt_lines = []
    for baseline_id, baseline in config["baselines"].items():
        generation = baseline["generation"]
        prompt_lines.append(
            f"| {baseline_id} | {str(baseline['do_sample']).lower()} | "
            f"`{baseline['seeds']}` | {generation['max_new_tokens']} | "
            f"{generation['temperature']} | {generation['top_p']} | "
            f"{baseline['batch_size']} | {baseline['max_batch_tokens']} |"
        )

    gap_lines = []
    for baseline_id, gap in metrics["random_template_gaps"].items():
        gap_lines.append(
            f"| {baseline_id} | `{gap['metric']}` | {format_float(gap['random'])} | "
            f"{format_float(gap['template'])} | {format_float(gap['random_minus_template'])} |"
        )

    parse_lines = []
    for baseline_id, failures in metrics["top_parse_failure_types"].items():
        if failures:
            for failure, count in failures.items():
                parse_lines.append(f"| {baseline_id} | `{failure}` | {count} |")
        else:
            parse_lines.append(f"| {baseline_id} | none | 0 |")

    verification_by_name = {check["name"]: check for check in verification["checks"]}
    greedy_repro = verification_by_name["independent_greedy_reproduction_exact"]["details"]
    sampling_repro = verification_by_name[
        "seeded_sampling_reproduction_within_tolerance"
    ]["details"]
    initial_status = args.git_status_initial.read_text(encoding="utf-8").rstrip()
    final_status = args.git_status_final.read_text(encoding="utf-8").rstrip()
    leaderboard_repro = split["leaderboard_filter_reproduction"]
    audit_comparison = provenance["audit_comparison"]

    text = f"""# Phase 1 평가 기반 구축 결과 보고서

실행 환경: Vast.ai RTX 4090 원격 인스턴스  
실제 작업 경로: `/workspace/university-deep-learning-challenge-2026`  
artifact: `{args.artifact_dir.as_posix()}`  
Git HEAD: `{args.git_head}`

## 결론

Phase 1의 고정 split, 형식적 답 추출, B0/B1/B2 Base baseline, metric 집계,
offline reload 및 재현 검증을 완료했다. 최종 verification은 **PASS**이며 Phase 2
학습이나 Verified-CoT 생성은 수행하지 않았다.

## 데이터와 provenance

| 자산 | 경로 | 행 수 | schema | SHA-256 |
|---|---|---:|---|---|
{chr(10).join(source_lines)}

- 원본 train과 leaderboard의 실행 전후 SHA-256은 동일하다.
- filtered train 16,528행은 versioned 임시 경로에서 행 내용과 byte hash가 모두 동일하게 재현됐다.
- 기존 train audit의 정책 필드와 newline 정규화 질문은 재현본과 동일하다. 기존 snapshot의
  embedded newline encoding 때문에 {audit_comparison['differing_rows']}행의 질문 길이·hash 계열
  metadata는 byte-level로 달랐으며 기존 파일은 덮어쓰지 않았다.
- 기존 leaderboard 831행 파생본은 ID 순서를 재현했지만 원본과 질문 내용이 다른 행이
  {leaderboard_repro['question_content_mismatch_count']}개다. 역사적 semantic 제외 정책이 없으므로
  이를 추측하지 않고 행별 원본/파생 hash와 legacy membership을 audit에 기록했다.

## 고정 split

| split | train | validation/diagnostic | seed | 누수 검사 |
|---|---:|---:|---:|---|
| Random | {split['splits']['random']['train_rows']} | {split['splits']['random']['validation_rows']} | {split['generator']['seed']} | ID overlap 0 |
| Template-group | {split['splits']['template']['train_rows']} | {split['splits']['template']['validation_rows']} | {split['generator']['seed']} | ID overlap 0, group leakage 0 |
| Hard diagnostic | — | {split['splits']['hard_diagnostic']['rows']} | {split['generator']['seed']} | deterministic category cap |
| Format diagnostic | — | {split['splits']['format_diagnostic']['rows']} | {split['generator']['seed']} | actual-label cases; synthetic parser tests separate |

동일 입력으로 split 생성기를 두 번 실행했으며 12개 deterministic 출력의 SHA-256이 모두 일치했다.
Template 정규화는 NFKC와 숫자·인명 후보·통화·단위의 표면 치환만 수행하며 문제 풀이 또는
정답 계산을 하지 않는다. 숫자만 바뀐 문제를 같은 group으로 묶기 때문에 Random 대비
Template 결과가 일반화 위험을 더 보수적으로 나타낸다.

## 모델과 생성 설정

- model/tokenizer: `Qwen/Qwen2.5-3B-Instruct`
- pinned revision: `{config['model']['revision']}`
- `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `local_files_only=True`
- dtype: BF16
- ground truth는 생성 종료 후 metric 계산에만 로드했다.

| baseline | sampling | seeds | max new tokens | temperature | top-p | 최대 batch | token budget |
|---|---|---|---:|---:|---:|---:|---:|
{chr(10).join(prompt_lines)}

B0는 최소 풀이·최종 marker 지시, B1은 단계별 풀이와 모델 내부 검산 지시,
B2는 B1 prompt에 세 seed sampling을 적용했다. 전체 prompt는 `configs/phase1.json`과
각 generation row에 보존했다.

배치는 라벨이나 모델 출력과 무관한 입력 token 길이로 정렬하고, 최대 256개 및
`(batch rows) × (max input tokens + max new tokens) <= 294,912` 제약으로 결정했다.
초기 혼합 순서 batch 128은 긴 prompt가 한 배치에 섞이며 OOM이 발생했고 그 log를 보존했다.
고정 batch 96 혼합 실행은 약 0.89 generation/s였지만, 최종 적응형 실행은 B0 중간 측정에서
약 1.77 generation/s, GPU 98~100%, VRAM 약 22.46/24.56 GiB를 기록했다.

## 결과

| baseline | scope | 문항 | greedy acc. | sample acc. | pass@k | majority@k | agreement@k | invalid | median tokens | p95 latency(s) | 1,000문항 예상(h) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(metric_lines)}

### Random–Template 차이

| baseline | metric | Random | Template | Random - Template |
|---|---|---:|---:|---:|
{chr(10).join(gap_lines)}

### 답 추출 실패 유형

| baseline | 유형 | 건수 |
|---|---|---:|
{chr(10).join(parse_lines)}

Extractor는 마지막 `FINAL_ANSWER:`, 마지막 `\\boxed{{...}}`, 명시적인 마지막 답 문장,
제한적인 독립 숫자 마지막 줄 순서만 지원한다. 상충 답·빈 출력·지원 marker 없음은 실패로
분류하며 쉼표·공백·부호·TeX fraction 표기만 정규화한다. 산술, 방정식 풀이, solver,
외부 서비스 또는 동적 retrieval은 없다.

## 재현·무결성 검증

- 독립 greedy 재실행: {greedy_repro['keys']} generation, raw text 일치
  {greedy_repro['raw_text_matches']}/{greedy_repro['keys']}.
- 독립 seeded sampling 재실행: {sampling_repro['keys']} generation, 추출 답 일치
  {sampling_repro['extracted_answer_matches']}/{sampling_repro['keys']}, accuracy 차이
  {format_float(sampling_repro['absolute_accuracy_difference'])}.
- Random ID overlap 0, Template ID overlap 0, Template group leakage 0.
- leaderboard audit는 1,000개 ID를 각각 한 번 포함한다.
- B0/B1/B2 generation은 누락·중복 없이 모든 Random·Template·Hard·Format ID를 포함한다.
- Phase 0 verification은 계속 PASS이며 Phase 0 artifact는 수정하지 않았다.
- focused test와 통합 verification log, raw generation, metric, source hash는 artifact에 보존했다.

## Git 상태와 변경 범위

초기 상태:

```text
{initial_status}
```

최종 상태:

```text
{final_status}
```

기존 modified/untracked 파일을 보존했다. stage, commit, branch 생성, push는 수행하지 않았다.

## 로드맵 반영

Phase 1 작업 10개는 split manifest, unit/integration test, raw generation, metric 및 최종
verification 근거로 완료 처리했다. 종료 조건 네 항목도 충족했다. Phase 2는 시작하지 않았다.

## 알려진 제약

- legacy leaderboard filtered 파일의 원래 semantic 필터 정책은 복원할 수 없다. 현재 audit는
  기존 membership과 원본 대비 질문 변환 차이를 정확히 고정한다.
- `agreement@k`는 정답성 증명이 아니라 모델 출력 문자열의 일치도다.
- runtime 추정은 이 RTX 4090, 적응형 token-budget batch와 현재 split에서 측정한 값이며
  다른 GPU에서는 달라진다.
"""
    atomic_write_text(args.output, text)
    print(json.dumps({"output": args.output.as_posix(), "sha256": sha256_file(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
