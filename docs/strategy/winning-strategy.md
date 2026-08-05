# 수상을 위한 모델 개발 전략

## 1. 결론

이 대회에서 가장 유력한 방향은 프롬프트 엔지니어링만으로 성능을 끌어올리는 것이 아니라 다음 파이프라인을 완성하는 것이다.

```text
Qwen2.5-3B-Instruct
  → 정답이 검증된 풀이 데이터 구축
  → QLoRA 기반 SFT
  → 선택적인 DPO 또는 GRPO
  → 문제별 adaptive self-consistency
  → 답 추출·투표·제출 형식 검증
```

자원과 시간이 제한된다면 다음 순서로 진행하는 것이 좋다.

1. 신뢰할 수 있는 로컬 평가 환경 구축
2. 정답 검증 풀이 데이터 생성
3. QLoRA SFT
4. Adaptive self-consistency와 후처리
5. 짧고 정확한 풀이를 선호하는 DPO
6. 선택적인 GRPO

핵심 아이디어는 **정답 검증 기반 데이터 증류**와 **문제 난이도에 따른 test-time compute 배분**이다. 프롬프트는 독립적인 핵심 기법이라기보다 모델의 추론 및 출력 형식을 안정시키는 기반으로 사용한다.

## 2. 대회 특성에서 도출되는 전략

대회는 범용 모델인 `Qwen/Qwen2.5-3B-Instruct`만을 출발점으로 허용하지만 SFT, LoRA·QLoRA, GRPO·DPO 등의 학습 기법과 Majority Voting, Self-Consistency, Best-of-N 등의 test-time 기법을 허용한다. 반면 2026-08-01 운영진 답변에 따라 추론 중 코드 실행과 도구 호출은 금지된다. 따라서 AIMO 상위권 전략 중 verified-CoT, 데이터 정제, SFT·DPO, self-consistency와 답 일치도 기반 early stopping은 적용할 수 있지만, TIR·Program-of-Thought·계산 verifier는 적용할 수 없다. 상세한 허용·금지 사항은 [모델·데이터·추론 규칙](../information/rules.md)을 따른다.

평가는 정답 exact match를 기반으로 하므로 그럴듯한 풀이보다 최종 답을 정확하게 생성하는 것이 중요하다. 또한 최종 수상은 모델 성능 50%와 발표 평가 50%를 합산하므로, 높은 점수뿐 아니라 재현성, ablation, 방법론의 설득력도 함께 준비해야 한다. 자세한 내용은 [평가와 제출](../information/evaluation-and-submission.md) 및 [일정과 수상](../information/schedule-and-awards.md)을 참고한다.

현재 제공 데이터에서 확인한 주요 특징은 다음과 같다.

| 항목 | 확인 결과 |
|---|---:|
| 학습 문항 | 17,000개 |
| 리더보드 문항 | 1,000개 |
| 필터링 리더보드 문항 | 831개 |
| 필터링에서 제외된 리더보드 문항 | 169개 |
| 학습 질문 길이 중앙값 | 약 203자 |
| 리더보드 질문 길이 중앙값 | 약 206자 |
| 학습 데이터의 음수 정답 | 502개 |
| 학습 데이터의 0 정답 | 222개 |
| 학습 데이터의 이미지·URL 포함 문항 | 152개 |
| 리더보드의 이미지·URL 포함 문항 | 6개 |
| 숫자만 다른 동일 템플릿 | 리더보드에서 최소 18개 |

### Canonical train 원칙

2026-08-04 이후 새로 시작하는 모든 train 기반 평가·teacher 생성·SFT·DPO·GRPO·학습 sampling은 `data/deep_chal_math_train_filtered_final_v1.csv`를 canonical modeling dataset으로 사용한다. 이 파일은 대회 측 제외 ID 627개와 추가 검토 ID 14개를 합친 641개를 원본 17,000행에서 제외한 16,359행 파생본이다. 원본 `data/deep_chal_math_train.csv`와 기존 16,528행 `data/deep_chal_math_train_filtered.csv`는 provenance와 과거 실험 재현 용도로 보존한다. 기존 파일과 해시를 기록한 완료 산출물을 새 final v1 결과로 재해석하지 않으며, 향후 split과 모델 실험은 final v1에서 새 버전으로 생성한다.

필터링 리더보드 파일은 83.1%를 보존한 분석용 파생본이다. 현재 생성 스크립트와 행별 제외 사유 감사표가 없으므로 필터 정책은 아직 재현 가능한 상태가 아니다. 외부 데이터 오염 검사, 리더보드 추론과 제출 범위는 필터링 여부와 관계없이 원본 1,000문항 전체를 기준으로 한다. 필터링 파생본에는 라벨이 없으며 학습 데이터로 사용하지 않는다. 자세한 파일 해시와 무결성 점검 결과는 [데이터 문서](../information/data.md#로컬-데이터-자산)를 따른다.

학습 세트와 리더보드 세트의 질문 길이와 형식은 비교적 유사하지만, 데이터에는 다음과 같은 노이즈가 존재한다.

- 이미지가 없으면 풀기 어려운 문항
- 증명을 요구하지만 정수형 라벨이 연결된 문항
- 질문이 불완전하거나 명시적인 요구사항이 없는 문항
- 외부 데이터셋에서 가져온 것으로 보이는 다양한 수학 문제 형식
- 극단적으로 크거나 음수인 정답

따라서 전체 데이터를 동일한 품질로 취급해 answer-only SFT를 수행하기보다, 풀이의 신뢰도를 계산하고 문항별 학습 가중치를 달리하는 접근이 적절하다.

## 3. 평가 환경을 먼저 구축해야 하는 이유

### 3.1 두 종류의 validation split

랜덤 split만 사용하면 숫자만 달라진 유사 템플릿이 train과 validation에 동시에 포함되어 성능을 과대평가할 수 있다. 최소한 다음 두 split을 함께 운영한다.

1. **Random holdout**
   - 실제 리더보드와 유사한 혼합 분포 성능을 측정한다.
   - 길이, 정답 부호, 정답 크기, 문제 형식을 가능한 한 계층화한다.

2. **Template-group holdout**
   - 숫자, 이름, 단위 등을 정규화해 유사한 템플릿을 같은 그룹에 넣는다.
   - 같은 그룹이 train과 validation에 동시에 포함되지 않게 한다.
   - 진짜 일반화 성능과 템플릿 암기 의존성을 확인한다.

추가로 이미지 누락, 긴 문항, 증명형 문항, 복잡한 기하·정수론 문제를 모은 hard holdout을 두면 오류 분석에 유용하다.

### 3.2 기록할 지표

모든 실험에서 단일 accuracy만 보지 말고 다음 지표를 함께 기록한다.

- Greedy accuracy
- pass@k
- majority@k
- 최종 답 추출 실패율
- 샘플 간 답 일치율과 실제 정답률의 관계
- 문제당 생성 토큰 수와 추론 시간
- 문제 유형별 accuracy
- Random holdout과 template-group holdout의 차이

리더보드 1,000문항에서는 1~2%p 차이가 불안정할 수 있다. Public 점수의 작은 변동보다 여러 split과 seed에서 반복되는 개선을 우선한다.

## 4. 프롬프트 전략

프롬프트의 목적은 모델에 새로운 수학 지식을 가르치는 것이 아니라 다음 동작을 안정시키는 것이다.

- 문제의 조건과 요구사항 식별
- 단계적인 계산
- 독립적인 검산
- 일관된 최종 답 형식

기준 프롬프트 예시는 다음과 같다.

```text
Solve the problem carefully.

1. Identify the quantities and constraints.
2. Derive the answer step by step.
3. Independently verify the calculation and units.
4. Write the final answer on the last line as:
FINAL_ANSWER: <answer>

Do not write anything after the final-answer line.
```

비교할 프롬프트는 다음 세 종류면 충분하다.

1. 직접적인 단계별 풀이
2. 풀이 후 독립적인 재계산 또는 검산
3. 문제를 방정식, 경우의 수 열거 또는 알고리즘으로 변환한 풀이

few-shot을 사용한다면 테스트 문항에 따라 검색하지 말고, 검증셋에서 미리 선정한 짧은 예시를 모든 문항에 고정적으로 적용한다. 테스트 타임 동적 retrieval은 운영진의 추가 허용 답변이 없는 한 사용하지 않는다.

## 5. 정답 검증 풀이 데이터 구축

### 5.1 생성 원칙

학습 데이터는 정답만 제공하므로 그대로 `question → answer`를 학습하면 모델이 추론 과정보다 단답 패턴을 학습할 수 있다. 다음 방법으로 풀이 데이터를 구축한다.

현재 Phase 2 v2에서는 다음 계약을 사용한다.

1. 각 filtered 학습 문제에 대해 정답 라벨을 숨긴 독립 풀이 후보를 정확히 2개 생성한다.
2. teacher의 `final_answer`는 `^-?(?:0|[1-9][0-9]*)$`와 일치하는 canonical 정수 문자열만 허용한다.
3. 후보가 직접 출력한 정수가 filtered 라벨과 exact match하고 비수정 verifier를 통과할 때만 채택한다.
4. 두 후보가 통과하면 A, 하나 이상이 통과하면 B로 분류한다.
5. 모두 실패해도 정답 조건부 생성으로 채우지 않으며 C 등급은 core dataset에 사용하지 않는다.
6. 계산식 검사는 안전한 독립 단순 등식에 한정하고, 복합식은 오류가 아닌 `not_checked_complex_expression`으로 기록한다.
7. 최종 target의 마지막 줄은 정확히 `FINAL_ANSWER: <integer>`이며 뒤에 단위·설명·문장부호를 붙이지 않는다.

정답을 teacher에게 제공하면 잘못된 라벨에도 억지로 도달하는 풀이가 만들어질 수 있다. 따라서 Phase 2 v2는 처음부터 끝까지 `정답 비공개 생성 → canonical 정수 일치 검사 → A/B만 채택` 순서를 유지한다.

학습 데이터 구축을 위한 상용 API 사용은 현재 규칙상 허용되지만, 다음 정보를 모두 기록해야 한다.

- 사용한 모델과 API
- 생성 날짜
- 전체 프롬프트
- 샘플링 설정
- 풀이 채택·제외 기준
- 자동 검증 코드
- 최종 데이터의 라이선스와 접근 방법

리더보드 및 최종 테스트 문제를 API, 검색 엔진 또는 외부 서비스에 입력해서는 안 된다.

이 단계의 코드 실행과 계산 검증은 **학습 데이터 구축에만** 사용한다. 검증된 풀이를 모델에 학습할 때는 실행 도구 호출을 제거하고, 테스트 시 모델 출력만으로 완결될 수 있는 자연어 풀이 형식으로 변환한다.

### 5.2 데이터 신뢰도 등급

문항마다 다음과 같은 신뢰도 등급을 두고 loss weight 또는 sampling weight에 반영할 수 있다.

| 등급 | 조건 | 활용 방식 |
|---|---|---|
| A | answer-hidden 후보 2개가 정답과 exact match하고 검증 통과 | core dataset |
| B | answer-hidden 후보 중 1개 이상이 정답과 exact match하고 검증 통과 | core dataset |
| C | Phase 2 v2에서는 생성하지 않음 | core dataset 제외 |
| D | 이미지 누락, 불완전 질문, 풀이·정답 불일치 | 제외 또는 분석 전용 |

### 5.3 외부 데이터

공개 외부 데이터는 대회 데이터와 유사한 문제만 선별한다. 예를 들어 [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2)는 GSM8K와 MATH 훈련 문제 및 증강 문제에 대한 풀이를 제공한다. 전체 데이터가 매우 크므로 다음과 같이 사용한다.

- 대회 질문과 길이 및 주제가 가까운 5만~20만 개를 우선 샘플링
- 지나치게 긴 풀이 제거
- 최종 답 형식을 대회 형식에 맞게 정규화
- 대회 validation 및 평가 문제와의 오염 검사
- 공개 데이터셋의 train split 위주로 사용

외부 데이터를 많이 넣는 것보다 대회 분포에 맞는 데이터를 선택하는 것이 중요하다. 외부 데이터로 일반적인 수학 추론 능력을 먼저 적응시키고, 마지막 단계에서 대회 데이터로 domain adaptation을 수행하는 curriculum도 비교한다.

## 6. QLoRA SFT

우선 QLoRA로 빠르게 실험을 반복한다. Full fine-tuning은 충분한 GPU와 명확한 성능 이득이 확인된 이후에만 검토한다.

초기 탐색 범위는 다음과 같이 설정할 수 있다.

| 하이퍼파라미터 | 초기 후보 |
|---|---|
| Quantization | 4-bit NF4 |
| LoRA rank | 32, 64, 128 |
| LoRA alpha | rank의 1~2배 |
| Target modules | attention 및 MLP projection |
| Learning rate | `5e-5`, `1e-4`, `2e-4` |
| Epoch | 1~3 |
| Sequence length | 1,024 또는 2,048 |
| Loss | assistant 응답 토큰에만 적용 |

중요한 비교 실험은 다음과 같다.

- Answer-only SFT
- 검증된 풀이 SFT
- 외부 수학 데이터 사전 SFT 후 대회 데이터 SFT
- 모든 풀이와 고신뢰 풀이만 사용한 경우
- 짧은 풀이와 긴 풀이

QLoRA의 방법론적 배경은 [QLoRA 논문](https://papers.neurips.cc/paper_files/paper/2023/file/1feb87871436031bdc0f2beaa62a049b-Paper-Conference.pdf)을 참고한다.

## 7. GRPO와 DPO

### 7.1 GRPO

정답 exact match는 자동 reward를 만들기 좋은 조건이다. SFT 모델이 안정된 후 다음과 같은 reward를 사용할 수 있다.

```text
최종 답 정확히 일치       +1.00
최종 답 형식 준수         +0.05
답 추출 실패              -0.05
복수의 상충하는 최종 답   -0.05
불필요하게 긴 출력         소폭 패널티
```

형식 reward가 정답 reward보다 커지지 않게 해야 한다. 또한 정답만 우연히 맞힌 잘못된 풀이를 과도하게 강화하지 않도록, 풀이 일관성 또는 로컬 계산 검증을 보조 reward로 검토할 수 있다.

처음부터 GRPO를 수행하기보다 다음 순서를 권장한다.

```text
Base model → verified-CoT SFT → GRPO
```

수학 추론에 대한 GRPO의 배경은 [DeepSeekMath 논문](https://arxiv.org/abs/2402.03300)을 참고한다.

### 7.2 DPO

풀이 데이터 생성 과정에서는 자연스럽게 같은 문제에 대한 정답 풀이와 오답 풀이가 만들어진다. 이를 preference pair로 구성하면 DPO를 적용할 수 있다.

- Chosen: 최종 답과 풀이 검증을 모두 통과한 응답
- Rejected: 계산 오류, 조건 누락 또는 최종 답 불일치 응답

GRPO 구현과 안정화에 시간이 많이 걸린다면 DPO를 비용이 낮은 대안으로 비교한다.

## 8. Retrieval의 사용 범위

학습 세트와 리더보드에는 숫자나 이름만 바뀐 유사 문제가 일부 존재하지만, 테스트 문항에 따라 BM25·embedding 검색 결과를 프롬프트에 동적으로 넣는 방식은 운영진이 명시적으로 허용하지 않았다. "추론 시 모델의 추론 출력만으로 답을 도출"해야 한다는 원칙을 보수적으로 적용해 테스트 타임 retrieval은 사용하지 않는다.

Retrieval과 템플릿 정규화는 다음과 같은 **학습·평가 준비 단계**에서만 사용한다.

- 중복 및 근접 중복 학습 데이터 제거
- template-group validation split 구성
- 외부 학습 데이터 중 대회 분포와 가까운 문제 선별
- 테스트 공개 전에 고정할 공통 few-shot 예시 후보 분석

문제별 동적 retrieval을 사용하려면 구현 전에 운영진에게 별도로 질문하고 답변을 보관해야 한다.

## 9. Adaptive test-time inference

### 9.1 기본 절차

모든 문제에 동일한 수의 풀이를 생성하는 대신 답 일치도에 따라 계산량을 배분한다.

1. 서로 다른 seed 또는 풀이 프롬프트로 3개 응답 생성
2. 세 최종 답이 모두 같으면 종료
3. 불일치하면 총 8개까지 확장
4. 계속 불일치하면 16개까지 확장
5. 정규화된 최종 답에 majority voting 적용
6. 동률이거나 모든 답이 다르면 추가 모델 샘플 또는 모델 내부 검산 프롬프트를 사용

Adaptive sampling controller는 모델이 출력한 답의 일치도와 남은 시간만 사용한다. 외부 계산 결과를 만들거나 모델 입력으로 되먹임해서는 안 된다.

초기 후보 설정은 다음과 같다.

| 설정 | 초기값 |
|---|---:|
| Temperature | 0.6~0.8 |
| Top-p | 0.9 |
| 초기 sample 수 | 3 |
| 1차 확장 | 8 |
| 최대 확장 | 16 |
| 최대 생성 길이 | 512~1,024 tokens |

Self-consistency의 이론과 실험적 근거는 [Self-Consistency 논문](https://arxiv.org/abs/2203.11171)을 참고한다.

### 9.2 허용 범위 안의 검산

검산도 모델 출력만으로 수행한다.

- 하나의 응답 안에서 풀이 후 독립적으로 다시 계산하도록 프롬프트한다.
- 같은 Qwen 체크포인트가 후보 풀이를 읽고 모델 출력으로 최종 후보를 고르게 할 수 있으나, Best-of-N의 구체적인 구현 범위는 운영진 답변과 일치하는지 확인한다.
- 출력 형식, 답 추출 성공 여부와 후보 답 일치도만 코드로 확인한다.

다음은 사용하지 않는다.

- 로컬 Python·SymPy·수치 해석기·SAT solver를 통한 계산 검산
- 미리 작성한 방정식, 소인수분해, 최대공약수 등의 계산 함수
- 계산 결과로 후보 답을 수정하거나 재순위화하는 deterministic verifier
- 계산 결과를 다시 모델 입력에 넣는 반복 추론

## 10. 답 추출과 제출 안정성

최종 답 추출기는 최소한 다음 표현을 처리해야 한다.

- 양의 정수와 음의 정수
- 0
- 매우 큰 정수
- 쉼표가 포함된 수
- `\boxed{...}`
- `FINAL_ANSWER: ...`
- 소수와 분수

문서상 모든 답은 정수지만 리더보드 원본의 `val-000007`은 주어진 식을 풀면 `2.5`가 된다. 이 행은 현재 필터링 파생본에서는 제외됐지만 전체 리더보드 제출 범위에는 남아 있으므로, 다음 사항을 운영진에 확인해야 한다.

1. 실제 제출 답이 정수로 제한되는가
2. 소수와 분수의 canonical format은 무엇인가
3. 문제와 라벨 형식이 맞지 않는 문항은 어떻게 처리되는가
4. 제출 컬럼명이 `ID`인지 `id`인지
5. 리더보드 원본의 ` answer` 선행 공백은 의도된 것인지

확인 전에는 내부 답 표현을 정수로 강제 변환하지 말고 소수와 분수를 보존한다. 최종 제출 생성 단계에서만 공식 sample submission과 metric에 맞게 변환한다.

답 추출기는 모델이 이미 출력한 표현을 읽고 형식만 정규화해야 한다. 새로운 산술 계산, 방정식 풀이 또는 후보 답 교정은 수행하지 않는다.

## 11. 권장 실험 순서

| 단계 | 실험 | 진입 조건 |
|---|---|---|
| B0 | Base greedy prompt | 최초 기준점 |
| B1 | 모델 내부 검산 프롬프트 | B0 대비 개선 확인 |
| B2 | Base self-consistency 3/8/16 | pass@k 잠재력 확인 |
| F0 | Answer-only QLoRA | 단순 SFT 기준점 |
| F1 | Verified-CoT QLoRA | 핵심 모델 |
| F2 | 외부 데이터 curriculum + F1 | domain shift 확인 |
| D1 | F1 + 길이 제어 DPO | 정확도·출력 길이 비교 |
| T1 | F1 + adaptive self-consistency | 핵심 추론 방식 |
| R1 | F1 + GRPO | 안정적인 F1·D1 이후 |
| Final | 최선 checkpoint/추론 조합 | latency 포함 최종 검증 |

각 실험은 한 번에 하나의 요소만 변경하고, validation prediction과 원시 generation을 모두 저장한다. 그래야 어떤 오류가 교정되거나 새로 생겼는지 확인할 수 있다.

## 12. 최종 테스트 운영

최종 테스트는 공개 후 제출까지 시간이 짧으므로 미리 다음을 준비한다.

- 입력 파일명과 컬럼 변화에 대응하는 loader
- sample submission 기반 출력 컬럼 자동 검증
- 중간 결과 checkpoint 및 재시작 기능
- 문항 단위 generation cache
- GPU별 예상 처리량 측정
- 3, 8, 16 samples에 대한 전체 예상 시간
- extraction 실패 및 빈 답 자동 탐지
- 제출 파일 row 수, ID 중복, 누락 검사
- 완전한 offline 실행 확인

Adaptive sampling은 성능뿐 아니라 제한된 제출 시간 내에 추론을 끝내기 위한 장치이기도 하다.

## 13. 발표 및 수상 전략

최종 수상은 발표 평가의 비중이 크므로 방법론을 다음 문장으로 설명할 수 있어야 한다.

> 정답 검증 기반 풀이 증류로 작은 범용 모델의 수학 추론 능력을 학습하고, 답 불확실도에 따라 test-time compute를 동적으로 할당했다.

발표에는 다음 ablation을 포함한다.

```text
Base
 Prompt and output normalization
 Verified-CoT SFT
 DPO length control
 Self-consistency
 Optional GRPO
 Adaptive inference
```

추천 시각화는 다음과 같다.

- 단계별 accuracy 개선 그래프
- Accuracy-latency Pareto curve
- 답 일치율에 따른 실제 정답률
- 문제 유형별 성능 변화
- Random split과 template-group split 비교
- 데이터 신뢰도 등급별 학습 효과

재현 검증을 위해 코드, 가중치, 환경, seed, 데이터 출처, 생성 프롬프트, 학습 하이퍼파라미터와 추론 설정을 처음부터 버전 관리한다.

## 14. 최종 권고안

가장 현실적인 1차 목표는 다음 구성이다.

```text
Qwen2.5-3B-Instruct
  + 고신뢰 teacher 풀이 17,000개
  + QLoRA SFT
  + 선택적인 길이 제어 DPO
  + adaptive self-consistency 3 → 8 → 16
  + robust answer extraction와 majority voting
```

이 구성이 안정된 후 validation의 pass@k가 높고 majority@k가 충분히 따라오지 못하는 문항이 많다면 모델 출력만 사용하는 Best-of-N 또는 내부 검산 프롬프트를 비교한다. pass@k 자체가 낮다면 test-time 기법보다 풀이 데이터의 품질과 SFT를 먼저 개선한다. TIR, Program-of-Thought, 동적 retrieval과 계산 verifier는 최종 파이프라인에서 제외한다.
