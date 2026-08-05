# 단계 4 최종 보고서 — DPO와 선택적 RL

문서 상태: **작성 전**

실행 기간: 미정

기준 로드맵: [`단계 4 — DPO와 선택적 RL`](../../docs/strategy/roadmap.md#9-단계-4--dpo와-선택적-rl)

## 1. 요약

단계 완료 후 DPO·GRPO 채택 여부와 단계 종료 여부를 작성한다.

## 2. 수행 작업

Preference pair 구성, 학습 설정, seed, 하드웨어와 ablation 범위를 기록한다. GRPO를 생략했다면 진입 조건 미충족 근거를 남긴다.

## 3. 실험 결과

| 실험 | Random greedy | Template greedy | Hard accuracy | pass@k | median tokens | total runtime | 결정 |
|---|---:|---:|---:|---:|---:|---:|---|
| SFT 기준 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미정 |
| DPO correctness | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미정 |
| DPO length-control | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미정 |
| GRPO(선택) | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미정 |

## 4. 산출물과 재현 명령

Preference manifest, 설정, 로그, checkpoint와 평가 결과를 연결한다.

## 5. 종료 조건 검증

로드맵의 단계 4 종료 조건별 결과와 근거를 기록한다.

## 6. 실패·한계·결정

정확도·다양성 회귀, 길이 감소 효과와 F1 복귀 여부를 기록한다.

## 7. 단계 5 인계 사항

주 checkpoint, fallback과 추론 calibration 입력을 기록한다.

## 8. 변경 이력

| 일자(KST) | 변경 내용 | 근거 |
|---|---|---|
| 2026-08-03 | 보고서 골격 생성 | [`roadmap.md`](../../docs/strategy/roadmap.md) |
