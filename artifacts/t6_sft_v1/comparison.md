# T6 SFT-v1 대조군 비교

모든 행은 T4에서 확정한 동일한 greedy 설정(입력 2048, 출력 2048, seed 42)과 동일한 표기 전용 답 추출기로 평가했다.

| 실험 | random | template | hard | format | invalid (random) | 평균 출력 토큰 (random) |
|---|---:|---:|---:|---:|---:|---:|
| base (T4 재사용) | 67.99% | 66.95% | 32.55% | 46.88% | 0.61% | 491.1 |
| answer-only SFT | 23.21% | 22.48% | 14.36% | 18.75% | 0.06% | 8.2 |
| 외부 CoT SFT | 60.84% | 61.33% | 21.64% | 37.50% | 0.79% | 439.9 |
| RFT SFT | 68.85% | 67.50% | 31.45% | 41.80% | 0.55% | 481.0 |
| RFT + 외부 CoT | 67.93% | 67.62% | 29.45% | 40.62% | 0.49% | 488.1 |

## 판정

- 본안(RFT + 외부 CoT)의 random holdout 변화: base 대비 -0.06pp, answer-only 대비 +44.72pp.
- base 대비 invalid 변화: -0.12pp.
- 채택 판정: **본안 미채택; T4 base 유지**.

## 결과 해석과 채택 가드

- 본안은 base 대비 3/4개 split에서 정확도가 하락했다(random_holdout, hard_diagnostic, format_diagnostic). 따라서 어댑터를 후속 단계에 전달하지 않고 T4 base를 유지한다.
- random 평균 출력 길이 변화는 -3.0 tokens, invalid 변화는 -0.12pp이다. invalid가 줄었는데도 정확도가 하락했다면 주된 실패는 형식 위반이 아니라 풀이 능력 퇴행으로 해석한다.
- RFT는 RFT pool 12636문제 중 c>=1인 10835문제만 덮고 c=0인 1801문제를 제외한다. 채택 trace는 40645행(덮은 문제당 평균 3.75행)이어서 이미 base가 풀기 쉬운 문제와 다중 성공 trace가 더 큰 가중치를 받는다.
- 이 실행만으로 인과를 분리할 수는 없지만, 위 solved-subset 편향과 고정된 2 epoch/LR 1e-4에서의 과적응·catastrophic forgetting이 가장 직접적인 원인 가설이다. 별도 ablation 없이 어느 하나를 확정 원인으로 단정하지 않는다.

## 대조군 해석 각주

answer-only는 RFT pool 전체에서 이미지 의존 문항만 제외한 범위를 사용하므로, 파손 문항과 오답 라벨도 그대로 타겟으로 학습한다. 반면 RFT 계열은 c=0 문항에서 채택 풀이가 없어 해당 문항이 자동으로 빠진다. 따라서 데이터 품질 비대칭은 ‘본안 > answer-only’ 결론에 유리한 방향이며, 대조군 범위는 사전 설계대로 바꾸지 않았다.

요청된 c>=1 holdout 부분 지표는 계산하지 않았다. c는 T5가 생성한 RFT pool 문항에만 정의되고 네 holdout은 RFT pool과 엄격히 분리되어 있어, c audit과 holdout의 교집합이 0개이기 때문이다. 라벨을 보거나 추가 생성으로 c를 새로 만드는 것은 대조군 설계를 바꾸므로 수행하지 않았다. 대신 학습 범위 진단으로 RFT pool 12636문제 중 c>=1은 10835문제, c=0은 1801문제이며, answer-only는 이미지 의존 18문제를 제외한 12618문제를 학습했다. 이 answer-only 학습 범위 안에서는 c>=1이 10826문제, c=0이 1792문제다.

## 학습 효율

각 학습 실행의 `training` 블록과 calibration.json에 스텝 시간, peak VRAM, GPU 사용률, gradient-checkpointing 비교 및 packing 마스크 검증이 보존되어 있다.
