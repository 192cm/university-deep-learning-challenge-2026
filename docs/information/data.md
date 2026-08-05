# 데이터

## 파일 구성

Data 페이지 설명에는 총 3개 CSV가 제공된다고 안내되어 있습니다.

| 설명에 적힌 파일명 | 용도 | 정답 공개 여부 |
|---|---|---|
| `deep_chal_math_dataset_train.csv` | 모델 학습에 직접 사용하는 훈련 세트 | `answer` 포함 |
| `deep_chal_math_dataset_leaderboard.csv` | 실시간 리더보드 평가 | `answer` 비공개 |
| `deep_chal_math_dataset_test.csv` | 최종 순위 평가 | `answer` 비공개 |

최종 테스트 파일은 **2026-08-31 00:00**에 공개되며, **2026-08-31 23:59**까지 답안을 제출해야 합니다.

## 로컬 데이터 자산

현재 저장소에서 확인한 원본과 파생 데이터는 다음과 같습니다.

| 경로 | 구분 | 행 수 | 열 | SHA-256 |
|---|---|---:|---|---|
| `data/deep_chal_math_train.csv` | 변경하지 않는 훈련 원본 | 17,000 | `id`, `question`, `answer` | `94f3302a6240b91b6fb3d093696b898750b8c4ca1d8ae1eb54210358664af9df` |
| `data/deep_chal_math_train_filtered.csv` | 기존 자체 정책 필터링 파생본(보존용) | 16,528 | `id`, `question`, `answer` | `2844386e4d5c7355f773ac58f4b735be4e9ce1caa70b4b1a3576244e1346ff98` |
| `data/train_filtered_ids.csv` | 대회 측 훈련 제외 목록 | 627 | `id`, `answer`, `question` | `67e4674afa685b985a6dc52e9050d9fb17116a99dbd9606cba82c976c904b4f3` |
| `data/train_filter_supplemental_ids_v1.csv` | 추가 검토 제외 원장 | 14 | `id`, 검토 분류·근거·원문 provenance | `cbed2dcc310cb758788ba9803ce9e2daf0e749f61b5f99773e2e8515f7e3d66f` |
| `data/deep_chal_math_train_filtered_final_v1.csv` | 최종 훈련 필터링 파생본 | 16,359 | `id`, `question`, `answer` | `9d5de9320861830bf6d0a9113685dd00dcc90b02baa877c4fecc4eb22cd61b4a` |
| `data/deep_chal_math_train_filter_final_v1_audit.csv` | 최종 필터 행별 감사표 | 17,000 | 판정·출처·사유·질문 해시 포함 | `4044c4a9936c766680f62665e7064b52047bdd8b3a155dfe265134d38f5af315` |
| `data/deep_chal_math_train_filtered_final_v1_manifest.json` | 최종 필터 생성 manifest | - | 입력·출력·스크립트·검증 해시 | `520d7f2e9291c107cbd00f6ab6d17afaca7f02b661a4d81e54b2c9324d584e90` |
| `data/deep_chal_math_leaderboard.csv` | 변경하지 않는 리더보드 원본 | 1,000 | `id`, `question`, `answer` | `f00b83805479140fb4d59fedb01c092e16c6cd35ac588f387b281ffea55eb2d7` |
| `data/deep_chal_math_leaderboard_filtered.csv` | 리더보드 필터링 파생본 | 831 | `id`, `question` | `032333a1361c8083093674ad19817e024c38dc7c9f4bdf05c0c9b0c71940dcf1` |

### Canonical modeling dataset

2026-08-04 이후 새로 시작하는 모든 train 기반 평가·teacher 생성·SFT·DPO·GRPO·학습 sampling은 `data/deep_chal_math_train_filtered_final_v1.csv`를 canonical modeling dataset으로 사용합니다. 이 파일은 대회 측 제외 ID 627개와 추가 검토 ID 14개를 합친 서로 겹치지 않는 641개를 원본 17,000행에서 제외한 결과입니다. 원본 `data/deep_chal_math_train.csv`와 기존 16,528행 `data/deep_chal_math_train_filtered.csv`는 provenance와 과거 실험 재현 용도로 변경하지 않고 보존합니다.

생성 설정은 `configs/train_filter_final_v1.json`, 실행 코드는 `scripts/build_final_filtered_train.py`, 17,000행 전체의 판정과 제외 사유는 `data/deep_chal_math_train_filter_final_v1_audit.csv`, 입력·출력 해시와 검증 결과는 `data/deep_chal_math_train_filtered_final_v1_manifest.json`에 기록합니다. 동일 타임스탬프로 두 번 재생성했을 때 dataset·audit·manifest의 SHA-256이 모두 일치했습니다.

이 원칙은 이후 실행과 새 버전 산출물에 적용합니다. 이미 완료된 Phase 0·1 및 Phase 2 v1·v2 산출물 중 기존 16,528행 파일과 그 해시를 명시한 결과는 과거 실험으로 보존하며, 새 final v1을 사용한 결과로 재해석하지 않습니다. 새 split이나 모델 실험은 final v1에서 별도 버전으로 생성해야 합니다.

### 리더보드 필터링 파생본 점검 결과

`deep_chal_math_leaderboard_filtered.csv`는 원본 1,000행 중 831행(83.1%)을 보존하고 169행을 제외합니다. 보존된 ID는 모두 고유하고 원본 순서를 유지하며, 빈 ID나 빈 질문은 없습니다. 원본의 `answer` 열은 1,000행 모두 비어 있고 파생본에서는 해당 열이 제거됐습니다. 보존된 질문 831개는 내용상 원본과 같으며, 다중행 질문 62개의 CSV 줄바꿈만 CRLF에서 LF로 정규화됐습니다.

이 파일에는 현재 생성 스크립트, 생성 시각을 기록한 manifest, 1,000행 전체의 행별 판정 감사표가 없습니다. 따라서 169개 제외 ID와 제외 사유의 재현 가능한 근거는 아직 문서화되지 않은 상태입니다. 해당 산출물이 준비되기 전에는 이 파생본을 잠정 분석 자산으로만 취급하고, 재생성하거나 기존 파일을 덮어쓰지 않습니다.

리더보드 보호·오염 검사는 필터링 파생본이 아니라 **1,000행 원본 전체**를 기준으로 수행해야 합니다. 리더보드 추론과 제출 리허설도 원본의 전체 ID를 사용해야 하며, 831행 파생본으로 원본을 대체하면 169개 ID가 누락되므로 완전한 제출 파일을 만들 수 없습니다. 라벨이 없는 필터링 리더보드 문항을 학습 데이터로 사용해서도 안 됩니다.

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
