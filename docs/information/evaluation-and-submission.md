# 평가와 제출

## 리더보드

| 구분 | 데이터 | 비중 | 용도 |
|---|---|---:|---|
| Public Leaderboard | 테스트 일부 | 30% | 대회 중 모델 개선을 위한 참고 점수 |
| Private Leaderboard | 테스트 나머지 | 70% | 대회 종료 후 공식 순위 결정 |

Public 점수는 최종 순위에 영향을 주지 않으며, **Private Leaderboard 점수로 최종 순위를 결정**합니다.

## 평가 지표

- 지표: **Accuracy (Exact Match)**
- 모든 정답은 정수입니다.
- 제출한 정수와 정답이 정확히 같을 때만 정답으로 처리됩니다.
- 모델 출력에 풀이, 수식, 설명이 포함되어도 되지만, 참가자가 후처리하여 최종 제출에는 정수만 남겨야 합니다.

## 제출 파일

- 파일명: `submisson.csv`i
- 컬럼: `ID`, `answer`
- `answer`에는 정수만 기입해야 합니다.
- 모든 문제의 답을 포함해야 하며 빈 값은 오답 처리됩니다.

```csv
ID,answer
prob_0001,42
prob_0002,7
prob_0003,125
```

> 데이터 원본 필드명은 소문자 `id`로 설명되지만 제출 예시는 대문자 `ID`를 사용합니다. 제출 직전 제공되는 sample submission의 정확한 컬럼명과 순서를 우선 확인하세요.

## 최종 결과 선정

Overview에는 최종 12팀(명)이 발표 대상이며, **모델 성능 50% + 발표 평가(모델 우수성) 50%**로 검증한 뒤 최종 9팀(명)을 수상자로 정한다고 안내되어 있습니다. 발표 날짜는 추후 공지 예정입니다.

따라서 Kaggle Private Leaderboard는 모델 성능 순위를 정하지만, 최종 수상은 발표 평가까지 합산되는 구조입니다.

## 출처

- [Kaggle Overview](https://www.kaggle.com/competitions/deep-learning-challenge-2026/overview)
- [Kaggle Rules](https://www.kaggle.com/competitions/deep-learning-challenge-2026/rules)
