-- Portable-report snapshot adapter for the frozen T12 diagnostics.
-- The primary computation is report/t12-orm-diagnostic-2026-08-28/analyze.py.
-- These SQLite-compatible views materialize the exact bounded rows embedded in
-- artifact.json so every rendered chart and table has auditable SQL provenance.

CREATE TEMP VIEW accuracy_comparison AS
SELECT 'ORM argmax' AS method, 0.828 AS accuracy, -4.6 AS delta_vs_raw_pp, 'diagnostic' AS role
UNION ALL SELECT 'Raw majority@32', 0.874, 0.0, 'baseline'
UNION ALL SELECT 'ORM weighted@32', 0.876, 0.2, 'candidate'
UNION ALL SELECT 'Oracle pass@32', 0.969, 9.5, 'ceiling';

CREATE TEMP VIEW baseline_error_decomposition AS
SELECT '정답 후보가 있었지만 오답' AS category, 87 AS questions, 0.690476 AS share_of_raw_errors, 'missed' AS selector_status
UNION ALL SELECT '정답 후보 없음', 31, 0.246032, 'impossible'
UNION ALL SELECT 'ORM이 rescue', 8, 0.063492, 'rescued';

CREATE TEMP VIEW changed_outcomes AS
SELECT '오답 → 다른 오답' AS outcome, 10 AS questions, '정확도 변화 없음' AS meaning
UNION ALL SELECT 'Rescue', 8, '+8 정답'
UNION ALL SELECT 'Break', 6, '-6 정답';

CREATE TEMP VIEW failure_mechanisms AS
SELECT '오답 그룹 ORM 점수도 정답 그룹 이상' AS mechanism, 45 AS questions, 0.517241 AS share
UNION ALL SELECT '정답 그룹 점수는 높지만 표 수에 패배', 42, 0.482759;

CREATE TEMP VIEW change_group_averages AS
SELECT '오답 → 다른 오답' AS outcome, 10 AS questions, 3.0 AS raw_support, 2.6 AS selected_support,
       0.34458 AS raw_geometric_mean, 0.584933 AS selected_geometric_mean
UNION ALL SELECT 'Rescue', 8, 8.5, 7.5, 0.314688, 0.583651
UNION ALL SELECT 'Break', 6, 11.0, 7.833333, 0.305384, 0.489267;

CREATE TEMP VIEW ranking_diagnostics AS
SELECT 1 AS "order", 'Global candidate ROC-AUC' AS metric, '0.812' AS value, '전체 후보를 질문 구분 없이 비교' AS interpretation
UNION ALL SELECT 2, '질문별 macro pairwise AUC', '0.740', '정답·오답 후보가 모두 있는 509문항'
UNION ALL SELECT 3, 'ORM argmax 정확도', '82.8%', 'Raw majority보다 4.6pp 낮음'
UNION ALL SELECT 4, 'Fresh 후보 실제 정답률', '81.1%', '유효 후보 30,266개 기준'
UNION ALL SELECT 5, 'Fresh 후보 평균 ORM 점수', '64.2%', '실제 정답률보다 16.9pp 낮음'
UNION ALL SELECT 6, 'Fresh ECE', '16.9%', 'Fresh 후보에서 calibration 오차가 큼';

CREATE TEMP VIEW segment_effects AS
SELECT 'Hard' AS segment, 222 AS questions, 0.536036 AS baseline_accuracy, 0.554054 AS orm_accuracy,
       1.801802 AS delta_pp, 4 AS net_correct
UNION ALL SELECT 'Format', 187, 0.764706, 0.775401, 1.069519, 2
UNION ALL SELECT 'Non-format', 813, 0.899139, 0.899139, 0.0, 0
UNION ALL SELECT 'Non-hard', 778, 0.970437, 0.967866, -0.257069, -2;

CREATE TEMP VIEW training_context AS
SELECT 1 AS "order", '고유 학습 문항' AS dimension, '6,034' AS local, '약 7,000' AS cmu
UNION ALL SELECT 2, 'Problem-solution pairs', '30,912', '37,880'
UNION ALL SELECT 3, 'Reward model', 'Qwen2.5-3B-Instruct', 'DeepSeekMath-7B-RL'
UNION ALL SELECT 4, '학습 방식', 'Pointwise LoRA, 2 epochs', 'Full fine-tuning, 2 epochs';
