-- Reproducible report extracts.
-- Each statement is self-contained so it can be run independently with SQLite.

WITH decision_metrics(
  best_sft_greedy_delta_pp,
  adopted_sft_count,
  rft_majority_delta_pp,
  inference_filter_delta_pp
) AS (
  VALUES (0.40, 0, 0.54, 1.47)
)
SELECT * FROM decision_metrics;

WITH sft_results(
  variant,
  delta_pp,
  union_accuracy_pct,
  p_value_label,
  decision
) AS (
  VALUES
    ('RFT SFT', 0.40, 63.10, '0.517', '미채택'),
    ('RFT-v2', 0.27, 62.96, '0.679', '보류'),
    ('RFT + 외부', -0.48, 62.22, '0.449', '미채택'),
    ('외부 CoT', -7.06, 55.64, '2e-24', '기각'),
    ('RFT-v2 + 외부', -8.30, 54.40, '5.30e-32', '기각'),
    ('answer-only', -40.81, 21.89, '≈0', '대조군')
)
SELECT * FROM sft_results ORDER BY delta_pp DESC;

WITH end_to_end_results(
  strategy,
  delta_vs_t8_pp,
  union_accuracy_pct,
  p_value_label,
  training_required
) AS (
  VALUES
    ('base + vote filter', 1.47, 70.78, '6.80e-10', '없음'),
    ('weighted vote P2', 1.18, 70.48, '6.21e-8', '없음'),
    ('RFT + vote filter', 0.99, 70.30, '0.0245', '있음'),
    ('RFT + majority@32', 0.54, 69.84, '0.228', '있음'),
    ('T8 기준선', 0.00, 69.31, '—', '없음')
)
SELECT * FROM end_to_end_results ORDER BY delta_vs_t8_pp DESC;

WITH data_quality_findings(finding, evidence, risk, severity) AS (
  VALUES
    (
      '쉬운 문제 편중',
      'R1 학습 40,645행 중 c≥4가 94.7%; c=1~3은 5.3%',
      '이미 맞히는 패턴을 반복하고 hard-tail 개선 신호가 약해짐',
      'High'
    ),
    (
      'packing 문맥 누수',
      'pack당 4.6샘플, 약 78%가 무관한 앞 문제를 관측',
      '문제 경계를 넘어선 허위 상관과 불안정한 추론 형식 학습',
      'Critical'
    ),
    (
      '의심 라벨',
      '0/48 집합 1,311문항; 표본 20개 중 파손·오답 13개',
      '정답 일치로 수확한 CoT가 잘못된 라벨을 강화할 수 있음',
      'High'
    ),
    (
      'format 회귀',
      'RFT-v2의 format 정확도 -3.91%p',
      '전체 평균이 비슷해도 제출 파싱과 긴 출력 안정성이 악화',
      'High'
    ),
    (
      '외부 CoT 종료 실패',
      'format hit-max 5.08%→22.66%, 합집합 -7.06%p',
      '긴 해설이 max token 종료와 답 추출 실패를 증가',
      'High'
    )
)
SELECT * FROM data_quality_findings;

WITH go_no_go_criteria(stage, required_condition, stop_condition) AS (
  VALUES
    (
      '데이터 준비',
      '정답·추출·종료 검증 통과, hard/long/non-arithmetic 층을 명시적으로 확보',
      '의심 라벨을 제거·교정할 수 없거나 hard-tail 규모가 불충분'
    ),
    (
      '학습 파이프라인',
      '샘플 간 attention 격리와 response-only loss를 테스트로 확인',
      '문제 경계 누수 또는 prompt 토큰 loss 혼입이 재현됨'
    ),
    (
      '소규모 pilot',
      'format·hard 각 -2%p 이내, 출력 길이·hit-max 악화 없음',
      '명백한 회귀가 보이면 full run 전에 중단'
    ),
    (
      '전체 paired 평가',
      'T8 대비 합집합 +1.5%p 이상, p<0.05, guardrail 통과',
      '세 조건 중 하나라도 실패하면 미채택'
    ),
    (
      '최종 확인',
      'fresh holdout 또는 사전등록된 nested validation에서 방향 재현',
      '기존 holdout에만 국한된 개선이면 보류'
    )
)
SELECT * FROM go_no_go_criteria;
