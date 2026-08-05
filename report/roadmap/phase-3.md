# 단계 3 최종 보고서 — QLoRA SFT

문서 상태: **작성 전**

실행 기간: 미정

기준 로드맵: [`단계 3 — QLoRA SFT`](../../docs/strategy/roadmap.md#8-단계-3--qlora-sft)

## 1. 요약

단계 완료 후 최선 checkpoint, 비교 결론과 단계 종료 여부를 작성한다.

## 2. 수행 작업

F0~F3의 데이터, 설정, seed, 하드웨어와 변경 변수를 기록한다.

## 3. 실험 결과

| 실험 | 핵심 변경 | Random greedy | Template greedy | pass@k | majority@k | invalid rate | p95 latency | 결정 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| F0 | Answer-only | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미정 |
| F1 | Local verified-CoT | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미정 |
| F2 | External curriculum + local | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미정 |
| F3 | 품질 등급 가중치 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미정 |

## 4. 산출물과 재현 명령

학습 설정, 로그, checkpoint, prediction, 원시 generation과 오프라인 재로딩 로그를 연결한다.

## 5. 오류·회귀 분석

Base 대비 새로 맞힌·틀린 문제, Template 성능, 출력 형식과 길이 문제를 기록한다.

## 6. 종료 조건 검증

로드맵의 단계 3 종료 조건별 결과와 근거를 기록한다.

## 7. 실패·한계·결정

기각한 checkpoint, 재실행 결과와 최종 SFT 후보 선택 이유를 기록한다.

## 8. 단계 4 인계 사항

고정 checkpoint, preference pair 생성 입력과 fallback을 기록한다.

## 9. 변경 이력

| 일자(KST) | 변경 내용 | 근거 |
|---|---|---|
| 2026-08-03 | 보고서 골격 생성 | [`roadmap.md`](../../docs/strategy/roadmap.md) |
