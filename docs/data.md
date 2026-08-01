# 데이터

## 파일 구성

Data 페이지 설명에는 총 3개 CSV가 제공된다고 안내되어 있습니다.

| 설명에 적힌 파일명 | 용도 | 정답 공개 여부 |
|---|---|---|
| `deep_chal_math_dataset_train.csv` | 모델 학습에 직접 사용하는 훈련 세트 | `answer` 포함 |
| `deep_chal_math_dataset_leaderboard.csv` | 실시간 리더보드 평가 | `answer` 비공개 |
| `deep_chal_math_dataset_test.csv` | 최종 순위 평가 | `answer` 비공개 |

최종 테스트 파일은 **2026-08-31 00:00**에 공개되며, **2026-08-31 23:59**까지 답안을 제출해야 합니다.

## 필드

| 필드 | 설명 |
|---|---|
| `id` | 문항 고유 식별자. 제출 답과 매칭되므로 변경하면 안 됨 |
| `question` | 자연어 설명과 LaTeX 수식이 포함될 수 있는 수학 문제 |
| `answer` | 정수형 최종 정답. 평가 데이터에서는 비어 있음 |

예시:

```json
{
  "id": "train-000000",
  "question": "What is the molecular weight of some moles of Aluminum chloride if the molecular weight of 3 moles is 396?",
  "answer": "132"
}
```

## 데이터 사용 조건

- 주최 측 제공 훈련 데이터를 기본으로 사용합니다.
- 모든 참가자가 무료로 동등하게 접근할 수 있는 공개 외부 데이터는 사용할 수 있습니다.
- 유료 구독, 특수 라이선스 또는 비공개 협약이 필요한 데이터는 사용할 수 없습니다.
- 사용한 외부 데이터셋과 접근 방법은 최종 제출 시 명시해야 합니다.
- 테스트 문제를 학습 데이터로 쓰거나 검색 엔진·외부 서비스에 입력해 답을 찾는 행위는 금지됩니다.
- 학습 데이터 구축을 위한 상용 API 사용(예: 풀이 생성, 데이터 증강)은 허용되지만, 테스트 문제의 답을 상용 API로 직접 생성하는 것은 금지됩니다.

## 라이선스와 페이지 표기 차이

- Data 페이지의 라이선스 표시는 Apache 2.0입니다.
- 2026-08-02 당시 화면의 Files 영역에는 2개 파일, 총 4.59 MB로 표시됐습니다. 설명에는 3개 파일이라고 되어 있으나 최종 테스트 파일은 8월 31일 공개 예정이므로 현재 파일 수와 차이가 나는 것으로 보입니다.
- 설명의 파일명은 `deep_chal_math_dataset_leaderboard.csv`이지만 실제 Files 영역에는 `deep_chal_math_leaderboard.csv`로 표시됐습니다.
- Overview/Rules는 평가 데이터를 `test.parquet`로 지칭하지만 Data 페이지는 CSV 3종으로 안내합니다. 구현 시에는 **실제 다운로드 파일과 최신 공지**를 기준으로 삼아야 합니다.

## 출처

- [Kaggle Data](https://www.kaggle.com/competitions/deep-learning-challenge-2026/data)
- [Kaggle Rules](https://www.kaggle.com/competitions/deep-learning-challenge-2026/rules)
