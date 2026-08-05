# 단계 5 최종 보고서 — 모델 출력만 사용하는 Test-Time Inference

문서 상태: **작성 전**

실행 기간: 미정

기준 로드맵: [`단계 5 — 모델 출력만 사용하는 test-time inference`](../../docs/strategy/roadmap.md#10-단계-5--모델-출력만-사용하는-test-time-inference)

## 1. 요약

단계 완료 후 최종 sampling 정책, 정확도·처리량 결론과 단계 종료 여부를 작성한다.

## 2. 수행 작업

답 추출, sampling 탐색, voting, early stopping과 규칙 준수 검증 범위를 기록한다.

## 3. 추론 정책 결과

| 정책 | Random accuracy | Template accuracy | 평균 samples | invalid rate | p95 latency | 전체 예상 시간 | 결정 |
|---|---:|---:|---:|---:|---:|---:|---|
| Greedy | 미측정 | 미측정 | 1 | 미측정 | 미측정 | 미측정 | 미정 |
| Fixed 8 | 미측정 | 미측정 | 8 | 미측정 | 미측정 | 미측정 | 미정 |
| Fixed 16 | 미측정 | 미측정 | 16 | 미측정 | 미측정 | 미측정 | 미정 |
| Adaptive | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미정 |

## 4. Calibration과 Pareto 분석

Agreement 구간별 정확도, stopping threshold와 accuracy-latency Pareto 근거를 기록한다.

## 5. 산출물과 재현 명령

답 추출기 테스트, adaptive 구현, 원시 generation, calibration과 시간 측정 로그를 연결한다.

## 6. 규칙 준수·종료 조건 검증

금지된 계산·도구·동적 retrieval이 없음을 포함해 단계 5 종료 조건별 근거를 기록한다.

## 7. 실패·한계·결정

기각한 정책, 동률·parse 실패 처리와 남은 처리량 위험을 기록한다.

## 8. 단계 6 인계 사항

주·저비용 추론 설정, 예상 시간과 fallback을 기록한다.

## 9. 변경 이력

| 일자(KST) | 변경 내용 | 근거 |
|---|---|---|
| 2026-08-03 | 보고서 골격 생성 | [`roadmap.md`](../../docs/strategy/roadmap.md) |
