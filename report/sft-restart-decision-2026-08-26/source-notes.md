# SFT 재시작 판단 — source notes

작성 시각: 2026-08-26T14:00:06Z

## 질문과 판단 범위

- 질문: 지금 새 SFT를 시작하는 것이 성능 향상에 도움이 되는가?
- 의사결정 범위: 현재 데이터와 학습 파이프라인을 그대로 재사용하는 즉시 재학습 여부.
- 비교 기준: T4c greedy와 현재 최종 기준인 T8 base majority@32.
- 주 지표: 합집합 3,737문항 정확도와 paired McNemar 검정.
- 보조 guardrail: hard/format split 각각 최대 -2%p 회귀.

## 출처 우선순위

1. `docs/strategy/execution-prompts.md`의 누적 실험 표와 데이터 품질 감사 표를 현재 의사결정 기록으로 사용했다.
2. `artifacts/t6_1_sft_v1r/manifest.json`으로 RFT-v2 두 arm의 정확도, 신뢰구간, p-value를 교차 확인했다.
3. `artifacts/t8_1_rft_self_consistency/comparison.json`으로 RFT+majority@32의 T8 대비 효과와 채택 판정을 교차 확인했다.
4. `artifacts/t7_rft_r2/metrics.json`으로 R2 수확량, 0/48 의심 집합 규모, SFT 미실행 상태를 확인했다.

## 핵심 수치 점검

- T4c greedy 합집합 정확도: 62.70%.
- SFT 변형별 T4c 대비 변화: answer-only -40.81%p, external CoT -7.06%p, RFT +0.40%p, RFT+external -0.48%p, RFT-v2 +0.27%p, RFT-v2+external -8.30%p.
- RFT-v2 manifest의 정확한 값: 2,353/3,737 = 62.964945%, Δ +0.267594%p, p=0.678729, 95% CI [-0.871825, +1.407014]%p.
- RFT+majority@32: 69.842119%, T8 69.306931% 대비 Δ +0.535189%p, p=0.227554, 95% CI [-0.290578, +1.360955]%p.
- base+vote-quality filter: 70.78%, T8 대비 +1.47%p, p=6.80e-10, 95% CI [+1.00, +1.95]%p.
- RFT+vote-quality filter: 70.30%, T8 대비 +0.99%p; base+filter보다 -0.482%p.
- weighted vote P2: 70.48%, T8 대비 +1.18%p.

## 데이터 품질 진단

- R1 SFT 행의 94.7%가 c≥4 문제에서 왔고 c=1~3은 5.3%였다. 목표 병목과 학습 분포가 불일치한다.
- pack당 평균 4.6개 샘플이고 attention isolation이 없어 약 78%의 샘플이 무관한 선행 문제를 관측했다. 이는 학습 파이프라인의 Critical 위험으로 분류했다.
- 0/48 의심 집합은 1,311문항이다. 20개 수동 표본 중 파손 4, 오답 9, 고난도 7로 분류됐으나 n=20이라 비율 추정 불확실성이 크다.
- RFT-v2는 hard를 유지했지만 format -3.91%p로 guardrail을 넘었다.
- external CoT는 format hit-max가 5.08%에서 22.66%로 증가해 종료·추출 실패를 확대했다.

## 해석과 사실의 구분

- 사실: 현재까지 실행된 SFT 변형은 모두 채택되지 않았다.
- 사실: 관측된 최고 SFT greedy 개선은 +0.40%p이고 유의하지 않았다.
- 사실: 같은 end-to-end 기준에서 RFT+majority@32의 증분은 +0.54%p였고, 학습 없는 base+filter의 증분은 +1.47%p였다.
- 해석: 현재 조건에서 추가 SFT의 기대값은 추론 정책 개선보다 낮다.
- 가설: packing 격리와 라벨 감사, hard-tail 재구성을 완료한 SFT-v3는 기존 SFT와 다른 결과를 낼 수 있다. 아직 직접 검증되지 않았다.

## 차트 맵

1. `sft_delta_chart`
   - 목적: SFT 변형별 T4c 대비 정확도 변화의 방향과 크기 비교.
   - 데이터: `sft_results` 6행.
   - 형식: 가로 막대, 0%p 기준선, 값 라벨 표시.
   - 독자가 내려야 할 결론: 반복 SFT의 관측 효과가 불안정하며 양의 효과도 작고 유의하지 않다.
2. `end_to_end_delta_chart`
   - 목적: 재학습과 학습 없는 추론 정책을 T8 기준으로 비교.
   - 데이터: `end_to_end_results` 5행.
   - 형식: 가로 막대, 사전등록 +1.5%p 채택선 표시.
   - 독자가 내려야 할 결론: 현재까지는 vote filtering이 RFT보다 더 큰 개선을 보였으나 채택선을 근소하게 넘지 못했다.

## 검증 상태

- 주요 수치는 누적 기록과 하위 JSON artifact를 교차 확인했다.
- 새 통계 모델을 적합하지 않았고, 보고서의 값은 기존 확정 artifact에서 직접 옮겼다.
- 추천은 관측 결과에 대한 자원 배분 판단이며 인과적 일반화를 주장하지 않는다.
- 전체 평가셋의 반복 사용에 따른 적응 위험은 남아 있다. 후속 SFT-v3는 fresh holdout 또는 사전등록된 nested validation에서 재현해야 한다.
- portable HTML의 canonical validation과 package 단계가 통과했다.
- 2026-08-26 인앱 브라우저 1,280px 렌더 QA에서 가로 overflow 0, 차트 2개, 표 2개, 콘솔 warning/error 0건을 확인했다.
- enhanced reader가 시작된 뒤 정적 fallback은 `display: none`으로 전환되는 것을 확인했다.
