# 아주 소중한 딥러닝 챌린지 2026 개발 로드맵

기준일: 2026-08-03(KST)
개발 기간: 2026-07-31 ~ 2026-08-30
최종 테스트 및 제출: 2026-08-31 00:00 ~ 23:59
평가·재현 검증: 2026-09-20까지 예정
수상자 발표: 2026-09-28 예정

이 문서는 `Qwen/Qwen2.5-3B-Instruct`만을 사용해 수학 문제의 최종 답을 생성하는 모델을 개발하고, 제한된 최종 테스트 시간 안에 재현 가능한 제출물을 만드는 실행 계획이다. 최신 공식 공지나 운영진 답변과 충돌하면 [모델·데이터·추론 규칙](../information/rules.md)을 우선한다.

## 1. 최종 목표

최종 시스템은 다음 조건을 모두 만족해야 한다.

```text
Qwen2.5-3B-Instruct
  → 오염을 제거한 검증용 split
  → 도구 없이 완결되는 verified-CoT 학습 데이터
  → QLoRA 기반 SFT
  → 선택적인 길이 제어 DPO
  → 모델 출력만 사용하는 adaptive self-consistency
  → 형식적 답 추출과 majority voting
  → 오프라인 재현 가능한 submission.csv
```

성공의 기준은 리더보드 점수 하나가 아니다. 다음 네 축을 함께 최적화한다.

1. **정확도**: Private Test에서의 exact-match accuracy
2. **일반화**: 숫자 치환형 템플릿과 새로운 문제 구조에 대한 성능
3. **처리량**: 최종 24시간 안에 전체 문제 추론과 제출 검증 완료
4. **재현성**: 코드, 데이터, 설정, 가중치와 실행 기록으로 동일 결과 재생성

## 2. 절대 준수 조건

### 2.1 모델과 가중치

- 출발점은 `Qwen/Qwen2.5-3B-Instruct` 하나뿐이다.
- 다른 모델의 가중치를 추론 모델에 불러오거나 병합하지 않는다.
- 외부 모델은 규칙이 허용하는 학습 데이터 생성 단계에서만 사용할 수 있다.
- 사용한 모든 외부 데이터와 teacher API의 출처, 라이선스, 접근 방법을 기록한다.

### 2.2 테스트 타임

허용되는 핵심 기법은 모델 응답의 다중 생성, Majority Voting, Self-Consistency와 모델 출력만을 사용하는 Best-of-N이다.

다음은 최종 추론 파이프라인에 포함하지 않는다.

- 모델이 생성한 코드를 실행하는 TIR·Program-of-Thought
- Python, SymPy, SAT solver, 수치 해석기 또는 사전 작성 계산 함수
- 계산 결과를 모델 입력에 되먹임하는 방식
- 계산 verifier로 답을 수정하거나 후보 순위를 변경하는 방식
- 운영진의 추가 허용 답변이 없는 문제별 동적 BM25·embedding retrieval

답 추출 코드는 모델이 이미 출력한 답을 읽고 표기만 정규화할 수 있다. 새로운 수학 계산이나 방정식 풀이는 수행하지 않는다.

### 2.3 테스트 데이터 보호

- 리더보드와 최종 테스트 질문을 외부 API, 검색 엔진 또는 외부 모델에 보내지 않는다.
- 리더보드 질문은 라벨이 없어도 학습 데이터로 사용하지 않는다.
- 외부 학습 데이터는 리더보드 **원본 1,000문항 전체**와 exact, 숫자 정규화 템플릿, 의미적 근접 중복 검사를 수행한다.
- 831문항 필터링 파생본은 보호 범위나 제출 범위를 줄이는 근거로 사용하지 않는다.
- 최종 테스트 공개 후에는 사전에 고정한 모델, 프롬프트와 추론 코드만 사용한다.

## 3. 현재 기준선

### 3.1 확인된 자산

- 원본 훈련 데이터: 17,000문항
- 현재 canonical 필터링 데이터: 16,359문항(`final_v1`)
- 대회 측 627개와 추가 검토 14개를 합친 641개 제외 원장·17,000행 감사표·생성 manifest
- 기존 자체 정책 필터링 데이터 16,528문항과 감사표(과거 실험 재현용 보존)
- 리더보드 데이터: 1,000문항
- 리더보드 필터링 파생본: 831문항(169문항 제외, 보존율 83.1%)
- 대회 정보·규칙·평가·일정 문서
- 초기 전략 문서와 데이터 필터링 QA 노트북

### 3.2 아직 필요한 핵심 자산

- 고정된 random holdout과 template-group holdout
- 리더보드 필터링 파생본의 재현 스크립트·행별 판정 감사표·생성 manifest
- 재현 가능한 greedy baseline
- verified-CoT 데이터셋과 생성 manifest
- QLoRA 학습·평가 파이프라인
- 실험 registry와 비교 표
- adaptive self-consistency 추론기
- 최종 테스트 리허설 기록과 운영 runbook

### 3.3 기준 경로

| 용도 | 권장 경로 |
|---|---|
| 공식 정보 | `docs/information/` |
| 전략·로드맵 | `docs/strategy/` |
| 재현 가능한 변환 코드 | `scripts/` |
| 원본·파생 데이터 | `data/` |
| 분석 보고서 | `report/` |
| 향후 모델 설정 | `configs/` |
| 향후 실험 기록 | `artifacts/experiments/` |
| 향후 검증 코드 | `tests/` |

새 디렉터리는 실제 산출물이 생길 때만 만든다. 빈 구조를 먼저 생성하지 않는다.

## 4. 공통 평가 계약

모든 실험은 동일한 데이터 split, 답 추출기와 지표 정의를 사용해야 한다.

### 4.1 필수 지표

| 지표 | 정의 | 목적 |
|---|---|---|
| `greedy_accuracy` | 결정론적 단일 생성 accuracy | 기본 모델 능력 |
| `sample_accuracy` | 개별 확률 생성의 평균 accuracy | 샘플 품질 |
| `pass@k` | k개 중 하나 이상 정답인 문항 비율 | 잠재적인 후보 생성력 |
| `majority@k` | k개 후보 최빈 답의 accuracy | 실제 투표 성능 |
| `agreement@k` | 최빈 답 득표율 | 불확실도 추정 |
| `invalid_output_rate` | 답 추출 실패 비율 | 출력 안정성 |
| `median_output_tokens` | 응답 토큰 길이 중앙값 | 추론 비용 |
| `p95_latency` | 문항별 지연시간 95백분위 | 최종 운영 위험 |
| `total_runtime_estimate` | 전체 테스트 예상 시간 | 24시간 적합성 |

### 4.2 필수 평가 split

1. **Random holdout**
   - 필터링 훈련 데이터의 약 10%를 고정 seed로 분리한다.
   - 질문 길이, 답 부호·크기와 문제 유형의 분포를 가능한 한 유지한다.

2. **Template-group holdout**
   - 숫자, 인명, 통화와 단위를 정규화해 근접 템플릿을 그룹화한다.
   - 같은 그룹이 train과 validation에 동시에 들어가지 않게 한다.
   - 단순 템플릿 암기가 아닌 일반화 성능을 측정한다.

3. **Hard diagnostic set**
   - 기하, 정수론, 경우의 수, 긴 문장제, 조건 누락 위험, 큰 정수 문제를 포함한다.
   - 전체 점수 계산보다 오류 유형 분석에 사용한다.

4. **Format diagnostic set**
   - 음수, 0, 큰 정수, `\boxed{}`, `FINAL_ANSWER:`, 수식이 긴 출력 등을 포함한다.
   - 답 추출기의 형식 처리만 검사한다.

### 4.3 실험 기록 필수 필드

모든 실험은 최소한 다음 정보를 남긴다.

```yaml
experiment_id:
created_at_kst:
objective:
code_commit_or_diff:
base_model_revision:
dataset_paths:
dataset_sha256:
split_version:
training_config:
inference_config:
seed:
hardware:
runtime:
metrics:
artifact_paths:
decision: keep | reject | rerun
decision_reason:
```

한 실험에서는 핵심 변수를 하나만 변경한다. 원시 generation을 보존해 점수 변화의 원인을 재검토할 수 있어야 한다.

## 5. 단계 0 — 규칙·환경 고정

### GPU 필요 여부

**부분 필요.** 규칙 정리, manifest 작성과 디렉터리 설계는 CPU만으로 가능하다. CUDA 호환성·VRAM 확인, 고정 revision 모델 로딩과 동일 seed의 짧은 추론 재현에는 GPU가 필요하다.

### 목표

실험 결과가 환경 차이나 규칙 오해로 무효화되지 않도록 실행 조건을 먼저 고정한다.

### 작업

- [x] Python, PyTorch, Transformers, PEFT, TRL과 CUDA 버전을 기록한다.
- [x] 사용 가능한 GPU 종류·개수·VRAM과 저장 공간을 기록한다.
- [x] `Qwen/Qwen2.5-3B-Instruct`의 정확한 revision과 tokenizer revision을 고정한다.
- [x] 모든 실험에 사용할 기본 seed 목록을 고정한다.
- [x] 공식 규칙과 2026-08-01 Discord 추론 제한을 팀 전체가 확인한다.
- [x] 외부 데이터·API 사용 manifest 형식을 확정한다.
- [x] 실험 ID와 산출물 디렉터리 규칙을 확정한다.

### 산출물

- 환경 manifest
- 모델 revision 기록
- 공통 실험 설정 템플릿
- 규칙 확인 체크리스트

### 종료 조건

- 동일 환경에서 동일 seed의 짧은 추론 결과를 두 번 재현할 수 있다.
- 모델과 tokenizer revision이 명시돼 있다.
- 금지된 테스트 타임 도구가 설계에서 제거돼 있다.

## 6. 단계 1 — 평가 기반 구축

### GPU 필요 여부

**필수.** 데이터 split, 정규화기와 답 추출기 개발은 CPU로 가능하지만, Base 모델의 greedy·다중 샘플 baseline 생성과 실제 지연시간 측정에는 GPU가 필요하다.

### 목표

리더보드 대신 로컬 지표로 모델 선택이 가능한 상태를 만든다.

### 작업

- [x] 필터링 훈련 데이터의 ID와 SHA-256을 고정한다.
- [x] 리더보드 원본·필터링 파생본의 SHA-256과 1,000행 행별 판정을 manifest로 고정한다.
- [x] Random holdout ID 목록을 생성하고 버전 관리한다.
- [x] 숫자·인명·단위 정규화기를 구현한다.
- [x] Template-group holdout ID 목록을 생성한다.
- [x] Hard·Format diagnostic set을 구성한다.
- [x] 답 추출기를 구현하되 산술 계산 기능은 넣지 않는다.
- [x] Base 모델 greedy baseline을 모든 split에서 실행한다.
- [x] 최소 3개 seed로 sampling baseline을 실행한다.
- [x] greedy, pass@k, majority@k, 토큰 길이와 지연시간을 한 표로 저장한다.

### 권장 baseline

| ID | 설정 |
|---|---|
| B0 | Base + 최소 지시문 + greedy |
| B1 | Base + 단계별 풀이·모델 내부 검산 + greedy |
| B2 | B1 + 3 samples |
| B3 | B1 + 8 samples |

### 산출물

- 고정 split 파일과 생성 스크립트
- 답 추출기 테스트
- Base 모델 baseline 보고서
- 문항별 원시 출력과 오류 분류

### 종료 조건

- split 사이 ID 중복이 없다.
- template group이 train과 validation에 걸쳐 있지 않다.
- 동일 설정의 지표가 재실행에서 허용 오차 안에 들어온다.
- 답 추출 실패 원인의 상위 유형이 식별돼 있다.

## 7. 단계 2 — Verified-CoT 데이터 구축

### GPU 필요 여부

**부분 필요.** 정제, 중복 제거, 라벨 비교, 감사표와 manifest 생성은 CPU로 수행할 수 있다. 로컬 모델로 풀이 후보를 대량 생성하거나 생성 품질을 모델로 평가할 때는 GPU가 필요하다. 허용된 외부 teacher API를 사용한다면 로컬 GPU 의존도는 줄지만, 출처·비용·응답과 생성 설정을 모두 기록해야 한다.

### 목표

정답 패턴이 아니라 도구 없이 완결되는 수학적 추론을 학습시키는 고신뢰 데이터셋을 만든다.

> 2026-08-04 실행 상태: 기존 Phase 2 v1은 변경하지 않고 보존했다. 새 `phase2_verified_cot_luna_3k_v2`는 filtered train 직접 입력 감사, v1 raw 무료 재분석, 정수 형식 계약과 verifier 회귀 검증을 완료했다. 10행 `low` smoke는 통과했지만, 동일한 고정 40행 비교에서 `low`와 `medium`이 모두 정확도 게이트를 통과하지 못했다. 기준을 낮추지 않고 고정 100행 audit와 main Batch를 시작하지 않았으며 상태는 `blocked_comparison_gate`이다.

> 2026-08-04 데이터 개정: `data/deep_chal_math_train_filtered_final_v1.csv`를 16,359행으로 생성하고 입력·출력 SHA-256, 17,000행 감사표와 결정적 재생성을 검증했다. 아래 Phase 2 v2 실행 증거는 기존 16,528행 파일을 사용한 과거 결과이므로 final v1 결과로 간주하지 않는다. 이후 split·teacher 생성·SFT는 final v1에서 새 버전으로 시작해야 한다.

### Phase 2 v2 실행 증거

- [x] `data/deep_chal_math_train_filtered.csv` 16,528행·schema·ID uniqueness·SHA-256을 직접 검증한다.
- [x] Phase 2 v2의 holdout·audit·eligible ID가 filtered train ID의 부분집합임을 감사한다.
- [x] 기존 v1 raw 응답을 수정하지 않고 completion·JSON·schema·정수 형식 지표로 다시 분석한다.
- [x] 쉼표 숫자와 복합식 관련 verifier 회귀 사례를 추가하고 전후 오류를 기록한다.
- [x] 10행 schema/completion smoke를 통과한다.
- [ ] 40행 `low`·`medium` 비교에서 품질 게이트를 통과한 최저 비용 설정을 선택한다.
- [ ] 선택 설정으로 고정 100행 audit를 통과한 뒤에만 main Batch를 시작한다.
- [ ] A/B 등급 verified-CoT를 비용·품질 한도 안에서 최대 3,000행 확정한다.

비교 실행 증거는 `report/phase2_v2/phase2_v2_quality_report.md`와 immutable raw response 180건에 보존한다. smoke는 completion·completed JSON·schema·canonical integer가 모두 100%였다. 비교에서는 `low`가 completion 100%, first exact 60%, pass@2 65%, `medium`이 completion 95%, first exact 62.5%, pass@2 65%로 고정 기준을 통과하지 못했다. 따라서 “통과 설정 선택”, 100행 audit, main Batch, 3,000행 확정 항목은 완료로 표시하지 않는다.

> 2026-08-05 실행 상태: 새 `phase2_verified_cot_luna_3k_final_v1_v3`를 `data/deep_chal_math_train_filtered_final_v1.csv`에서 다시 구축했다. Phase 1 네 split의 protected ID 합집합, leaderboard 원본 1,000행 exact/template/near-duplicate audit, 기존 v2 실패 audit의 high-confidence 품질 제외 9행을 새 versioned audit로 고정했다. 기존 raw response는 재사용하지 않았다. low smoke는 completion/JSON/schema/canonical integer 100%, non-integer 0%, fatal verifier 0%로 통과했다. 고정 40행 비교는 low가 completion 100%, first exact 52.5%, pass@2 52.5%, medium이 completion 97.5%, first exact 52.5%, pass@2 55.0%로 모두 gate 실패했다. medium truncation은 이전 4건에서 2건으로 감소했지만 completion gate도 통과하지 못했다. 상태는 `blocked_comparison_gate`이며 100행 quality audit와 main Batch는 시작하지 않았다. 최종 verification은 `report/phase2_v3_final_v1/phase2_v3_final_v1_verification.json` 41/41 PASS다.

### 작업 A — 입력 정제와 오염 방지

- [x] 기존 필터링 정책과 감사표를 재실행해 원본 불변성을 확인한다.
- [x] 훈련 내부 exact·정규화 템플릿 중복을 표시한다.
- [x] 리더보드 원본 1,000문항과 exact·템플릿 근접 중복인 외부 데이터는 제거한다.
- [x] 이미지·URL·복수 출력·증명 전용 문항을 학습 목적별로 분리한다.
- [x] 문제 유형, 길이, 답 크기와 신뢰도 metadata를 추가한다.

### 작업 B — 풀이 생성

Phase 2 v2는 각 문제에 대해 다음 순서를 사용한다.

1. `data/deep_chal_math_train_filtered.csv`의 라벨을 공개하거나 암시하지 않고 독립 후보 2개를 생성한다.
2. 후보가 직접 출력한 canonical 정수와 filtered 라벨을 exact match로 비교한다.
3. 두 후보가 검증을 통과하면 A, 하나 이상이 통과하면 B로 채택한다.
4. 모두 실패해도 정답 조건부 후보를 생성하지 않고 D·`unsuitable`·미생성으로 제외한다.
5. 학습 데이터 구축 단계의 verifier는 안전한 독립 단순 등식만 검사하며 풀이 또는 정답을 수정하지 않는다.
6. 최종 학습 target에서는 코드 실행, 도구 호출과 실행 결과 되먹임을 제거한다.
7. target 마지막 줄은 설명이나 단위 없이 `FINAL_ANSWER: <integer>`로 끝낸다.

### 작업 C — 품질 등급

| 등급 | 조건 | 사용 |
|---|---|---|
| A | answer-hidden 독립 후보 2개가 canonical 정수 라벨과 일치하고 검증 통과 | core dataset |
| B | answer-hidden 독립 후보 중 1개 이상이 canonical 정수 라벨과 일치하고 검증 통과 | core dataset |
| C | Phase 2 v2에서는 생성하지 않음 | core dataset 제외 |
| D | 불완전·모순·라벨 의심·도구 없이는 불완전 | 제외 또는 분석 전용 |

### 작업 D — 외부 데이터 실험

검토 후보는 NuminaMath-CoT, OpenMathReasoning, OpenMathInstruct 계열처럼 공개적이고 출처를 기록할 수 있는 데이터다.

- [x] 라이선스와 무료 접근 가능 여부를 확인한다.
- [x] 단일 수치 정답·영어·텍스트 문항만 선별한다.
- [x] 대회 훈련 데이터와 리더보드 원본 1,000문항에 대해 오염 검사를 수행한다.
- [ ] 지나치게 어려운 데이터가 3B 모델 성능을 떨어뜨리지 않도록 난도별로 샘플링한다.
- [ ] 로컬 데이터 중심 혼합과 외부 데이터 curriculum을 별도 실험한다.

### 산출물

- verified-CoT 데이터셋
- 행 단위 생성·검증 감사표
- 외부 데이터 manifest와 라이선스 목록
- 채택률·신뢰도 등급·문제 유형 분포 보고서
- 데이터셋 SHA-256

### 종료 조건

- 모든 학습 행에 출처와 생성 방법이 있다.
- 라벨과 불일치한 독립 풀이가 학습 target에 없다.
- 리더보드 원본 1,000문항과 확인된 중복이 없다.
- 무작위 표본을 사람이 검토해 치명적 오류 유형이 통제돼 있다.
- 최종 target은 테스트 타임 도구 없이 완결된다.

## 8. 단계 3 — QLoRA SFT

### GPU 필요 여부

**필수.** QLoRA 학습, checkpoint별 generation과 정확도·처리량 평가는 CUDA GPU에서 수행한다. CPU는 데이터 준비와 로그 집계에만 사용하며, CPU-only 학습은 현재 일정의 실행안으로 간주하지 않는다.

### 목표

Answer-only와 verified-CoT 학습의 효과를 분리해 측정하고, 일반화 가능한 핵심 checkpoint를 선택한다.

### 실험 순서

| ID | 학습 데이터 | 목적 |
|---|---|---|
| F0 | question → answer | 단순 SFT 기준점 |
| F1 | 고신뢰 로컬 verified-CoT | 핵심 후보 |
| F2 | 외부 수학 데이터 → 로컬 verified-CoT | curriculum 효과 |
| F3 | A/B 등급 중심 + C 저가중치 | 품질 가중치 효과 |

### 초기 탐색 범위

| 설정 | 1차 후보 |
|---|---|
| Quantization | 4-bit NF4 QLoRA |
| LoRA rank | 32, 64 |
| LoRA alpha | rank 또는 2 × rank |
| Target modules | attention·MLP projection |
| Learning rate | `5e-5`, `1e-4` |
| Epoch | 1~3 |
| Sequence length | 1,024 우선, 필요 시 2,048 |
| Loss mask | assistant 응답 토큰만 |

처음부터 전체 조합을 실행하지 않는다. 작은 대표 subset에서 학습 안정성, 메모리와 처리량을 확인한 뒤 유망 조합만 전체 데이터로 확장한다.

### 평가

- Random·Template holdout의 greedy accuracy
- Hard set의 문제 유형별 accuracy
- pass@3, pass@8, majority@3, majority@8
- 답 추출 실패율
- 평균·중앙 토큰 길이
- Base 대비 새로 맞힌 문제와 새로 틀린 문제

### 산출물

- 학습 설정과 로그
- checkpoint별 validation prediction
- 원시 generation 표본
- F0/F1/F2/F3 비교 보고서

### 종료 조건

- F1이 F0보다 template holdout에서 명확히 우수하거나, 최소한 random 개선을 유지하면서 template 성능을 악화시키지 않는다.
- 출력 형식 오류가 통제 가능하다.
- 같은 설정을 재학습했을 때 결론이 뒤집히지 않는다.
- checkpoint와 tokenizer를 오프라인에서 다시 로드할 수 있다.

### 중단·회귀 조건

- Random 성능만 오르고 template 성능이 하락하면 데이터 중복과 과적합을 재점검한다.
- pass@k 자체가 낮으면 test-time sampling을 늘리지 말고 데이터와 SFT로 돌아간다.
- 긴 풀이가 잘려 미완성되는 비율이 높으면 sequence length보다 데이터 풀이 길이 정제를 먼저 검토한다.

## 9. 단계 4 — DPO와 선택적 RL

### GPU 필요 여부

**필수.** Preference pair 구성과 정적 검사는 CPU로 가능하지만, DPO 학습과 전후 checkpoint 평가는 GPU가 필요하다. GRPO는 생성과 학습을 반복하므로 DPO보다 더 많은 GPU 시간과 여유 VRAM이 확보된 경우에만 진행한다.

### 목표

정확도를 유지하면서 지나치게 긴 추론, 조건 누락과 반복 출력을 줄인다.

### DPO 데이터 구성

초기 규모는 1,000~3,000 preference pair로 제한한다.

- **Chosen**: 정답, 조건 충족, 검산 포함, 비교적 짧은 풀이
- **Rejected 1**: 최종 답 오류 또는 조건 누락
- **Rejected 2**: 정답이지만 불필요하게 반복되고 긴 풀이
- **Rejected 3**: 답 형식 불일치 또는 상충하는 최종 답

Chosen을 지나치게 짧은 answer-only 응답으로 만들지 않는다. 풀이 능력을 보존할 최소한의 논리 구조가 있어야 한다.

### 실험

- [ ] SFT checkpoint를 고정하고 DPO 데이터만 변경한다.
- [ ] correctness pair와 length-control pair를 분리해 ablation한다.
- [ ] `beta`는 작은 범위에서 탐색하고 극단적인 선호 학습을 피한다.
- [ ] accuracy와 출력 길이를 동시에 비교한다.
- [ ] DPO 후 pass@k 다양성이 과도하게 감소하는지 확인한다.

### GRPO 진입 조건

GRPO는 다음 조건을 모두 만족할 때만 시작한다.

- F1과 DPO 파이프라인이 안정적이다.
- 충분한 GPU와 남은 일정이 있다.
- 반복 가능한 exact-answer reward가 준비돼 있다.
- GRPO가 실패해도 최종 후보로 돌아갈 checkpoint가 고정돼 있다.

GRPO reward 계산에서 학습 문제의 라벨과 학습 단계의 계산 검증을 사용할 수 있다. 이 기능을 최종 테스트 추론 코드로 가져오지 않는다.

### 종료 조건

- DPO 모델이 SFT 대비 accuracy를 실질적으로 악화시키지 않는다.
- 중앙 출력 길이 또는 전체 추론 시간이 의미 있게 감소한다.
- template holdout과 hard set에서 성능 붕괴가 없다.
- 개선이 없으면 DPO를 최종 구성에서 제외하고 F1으로 복귀한다.

## 10. 단계 5 — 모델 출력만 사용하는 test-time inference

### GPU 필요 여부

**필수.** 답 추출기, 투표 controller와 단위 테스트는 CPU로 개발할 수 있지만, 3·8·16개 샘플 생성, sampling calibration과 정확도-지연시간 측정에는 GPU가 필요하다.

### 목표

외부 계산 도구 없이 여러 모델 응답의 일치도를 이용해 정확도와 처리량을 함께 높인다.

### 10.1 답 추출

답 추출 우선순위의 초기안은 다음과 같다.

1. `FINAL_ANSWER: <answer>`
2. 마지막 `\boxed{<answer>}`
3. 명시적인 최종 답 문장
4. 사전에 검증된 제한적 fallback

답 추출기는 표기만 읽는다. 후보 답에 산술 연산을 적용하거나 문제를 다시 풀지 않는다.

### 10.2 Adaptive self-consistency

초기 정책은 다음과 같다.

1. 서로 다른 seed로 3개 응답을 생성한다.
2. 세 답이 모두 같으면 종료한다.
3. 불일치하면 총 8개까지 확장한다.
4. 8개 중 최빈 답이 5표 이상이고 2위와 2표 이상 차이면 종료 후보로 본다.
5. 계속 불확실하면 16개까지 확장한다.
6. 추출 실패 응답은 정답 후보로 투표하지 않는다.
7. 최종 동률이면 사전에 정한 모델-only 추가 샘플 또는 내부 검산 프롬프트를 사용한다.

위 임계값은 고정 진리가 아니다. Random·Template validation에서 정확도-시간 Pareto가 가장 좋은 값으로 교체한다.

### 10.3 샘플링 탐색

| 설정 | 후보 |
|---|---|
| Temperature | 0.6, 0.7, 0.8 |
| Top-p | 0.9, 0.95 |
| 최대 샘플 | 8, 16 |
| 최대 출력 길이 | 512, 1,024 |
| 프롬프트 | 직접 풀이, 풀이+내부 검산 |

동적 retrieval 대신 테스트 전에 고정한 공통 few-shot만 비교한다.

### 10.4 허용된 early stopping

Controller가 사용할 수 있는 정보는 다음으로 제한한다.

- 이미 종료된 모델 출력의 최종 답
- 답 추출 성공 여부
- 답별 득표수와 일치도
- 사용 시간과 남은 전체 시간

문제 계산 결과, Python 실행 결과, solver 출력 또는 외부 검색 결과는 사용하지 않는다.

### 산출물

- 답 추출기와 단위 테스트
- adaptive sampling 구현
- agreement와 실제 정답률의 calibration 표
- accuracy-latency Pareto curve
- 전체 테스트 예상 시간표

### 종료 조건

- adaptive 정책이 고정 16-sample 대비 비슷한 정확도를 더 짧은 시간에 달성한다.
- 전체 예상 시간이 가용 시간의 70% 이내다.
- 남은 30%는 파일 검증, 재시작, 오류 복구와 제출 버퍼로 확보한다.
- 추론 코드에 금지된 계산·도구·동적 retrieval이 없다.

## 11. 단계 6 — 최종 후보 선정과 리허설

### GPU 필요 여부

**필수.** 제출 schema·hash 검사는 CPU로 가능하지만, 전체 오프라인 추론 리허설, 처리량 검증, 중단·재시작과 GPU 장애 fallback 검증에는 최종 테스트와 동급의 GPU 환경이 필요하다.

### 목표

최종 테스트 공개 후 모델 연구를 계속하지 않아도 되는 동결된 제출 시스템을 만든다.

### 후보 구성

- 주 모델 checkpoint 1개
- 보수적인 fallback checkpoint 1개
- 주 adaptive inference 설정 1개
- 처리량 부족 시 사용할 저비용 inference 설정 1개

후보 수를 늘리기보다 각 후보의 재현성과 실행 안정성을 높인다.

### 리허설

- [ ] 리더보드 원본 1,000문항과 같은 schema의 복사본으로 전체 추론을 수행한다.
- [ ] 인터넷이 완전히 차단된 상태에서 실행한다.
- [ ] 중단 후 문항 단위 cache에서 재시작한다.
- [ ] GPU 하나가 실패하거나 느려졌을 때 fallback 절차를 실행한다.
- [ ] 예상 전체 시간과 실제 전체 시간의 차이를 측정한다.
- [ ] 제출 파일의 row 수, ID, 중복, 누락과 빈 답을 검사한다.
- [ ] 서로 다른 작업 디렉터리에서 두 번 재현한다.
- [ ] 모델·코드·설정·데이터 manifest의 SHA-256을 기록한다.

### 동결 일정

- 8/27: 기능 추가 마감
- 8/28: 1차 전체 리허설
- 8/29: 후보와 설정 동결
- 8/30: 버그 수정만 허용, 최종 runbook 검토

### 종료 조건

- 전체 오프라인 리허설을 두 번 완료했다.
- 24시간 안에 실패 복구 시간을 포함해 제출 가능하다.
- 주 모델과 fallback 모델의 선택 기준이 문서화돼 있다.
- 최종 코드가 테스트 질문을 외부로 전송하지 않는다.

## 12. 단계 7 — 8월 31일 최종 테스트 운영

### GPU 필요 여부

**필수.** 입력·제출 형식 검사는 CPU로 가능하지만, 동결된 모델의 최종 추론은 GPU로 실행한다. 주 GPU와 사전에 검증한 fallback 실행 경로 없이는 이 단계를 시작하지 않는다.

### 12.1 시작 전

- [ ] 시스템 시간과 마감 시간대를 확인한다.
- [ ] 모델, tokenizer, Python 환경과 GPU 상태를 확인한다.
- [ ] 디스크 여유 공간과 cache 경로를 확인한다.
- [ ] runbook과 fallback 명령을 준비한다.

### 12.2 테스트 공개 직후

- [ ] 테스트 파일을 변경하지 않는 원본으로 보관하고 SHA-256을 기록한다.
- [ ] 파일명, encoding, column과 ID uniqueness만 확인한다.
- [ ] 테스트 질문을 외부 API·검색·모델 서비스에 보내지 않는다.
- [ ] sample submission의 정확한 column명과 row 순서를 확인한다.
- [ ] 동결된 loader가 schema를 처리하는지 소수 행으로 확인한다.

### 12.3 추론 실행

- [ ] 주 설정으로 문항 단위 추론과 cache 저장을 시작한다.
- [ ] 처리량, 남은 시간과 실패 문항 수를 주기적으로 기록한다.
- [ ] 예상 시간을 초과하면 사전에 정한 저비용 설정으로 전환한다.
- [ ] 재시작 시 완료 문항을 다시 생성하지 않는다.
- [ ] 코드 실행·계산 도구·동적 retrieval을 문제 풀이에 사용하지 않는다.

### 12.4 제출 전 검증

- [ ] sample submission과 row 수가 같다.
- [ ] 모든 ID가 정확히 한 번 존재한다.
- [ ] 누락·중복·빈 답·파싱 실패가 없다.
- [ ] 답 column은 공식 형식과 일치한다.
- [ ] 제출 파일 SHA-256과 생성 시각을 기록한다.
- [ ] 주 제출과 fallback 제출을 구분해 보관한다.
- [ ] 마감보다 충분히 앞서 제출한다.

## 13. 단계 8 — 재현 검증과 발표

### GPU 필요 여부

**부분 필요.** 문서 작성, 표·그래프 생성, manifest와 발표 자료 구성은 CPU로 가능하다. 최종 checkpoint 재학습, 추론 재현과 필수 ablation 재실행에는 원 실험과 호환되는 GPU가 필요하다.

### 재현 패키지

- 학습·추론 코드
- 모델 adapter 또는 허용된 최종 가중치
- 데이터 출처·라이선스·접근 방법
- 생성 프롬프트와 teacher 설정
- 환경·하드웨어·라이브러리 버전
- 학습 hyperparameter, seed와 checkpoint 선택 기준
- 최종 추론 설정과 예상 처리량
- 원본부터 제출까지의 실행 순서

### 필수 ablation

```text
Base greedy
  + 모델 내부 검산 프롬프트
  + 데이터 필터링
  + Verified-CoT QLoRA
  + 외부 데이터 curriculum
  + DPO length control
  + Self-consistency
  + Adaptive early stopping
```

### 발표용 분석

- 단계별 accuracy 개선
- Random과 Template holdout 비교
- 문제 유형별 성능
- pass@k와 majority@k 차이
- 답 일치도별 실제 정답률
- Accuracy-latency Pareto
- DPO 전후 출력 길이와 정확도
- 데이터 신뢰도 등급별 효과
- 실패 사례와 규칙 준수 설계

## 14. 의사결정 규칙

### 14.1 다음 행동 결정

| 관찰 | 해석 | 다음 행동 |
|---|---|---|
| `pass@k`도 낮음 | 좋은 후보 자체를 못 만듦 | 데이터·SFT 개선 |
| `pass@k`는 높고 `majority@k`가 낮음 | 후보 선택 문제 | sampling calibration, model-only Best-of-N 검토 |
| Random만 개선, Template 하락 | 템플릿 과적합 | 중복 제거·split 재검토 |
| 정확도 동일, 출력 길이 감소 | 처리량 개선 | DPO 후보 유지 |
| 양자화 후 속도 향상·정확도 급락 | 품질 손실 과다 | 더 높은 정밀도로 복귀 |
| 16 samples가 8보다 이득 없음 | 계산 낭비 | 최대 샘플 축소 |
| agreement가 높지만 자주 오답 | 같은 오류로 수렴 | 데이터 오류 유형 보강 |
| 규칙 해석이 불명확 | 실격 위험 | 구현 중단 후 운영진 확인 |

### 14.2 우선순위

**P0 — 반드시 완료**

- 고정 validation split
- 재현 가능한 baseline
- Verified-CoT QLoRA
- 답 추출기와 self-consistency
- 최종 오프라인 리허설

**P1 — P0 안정 후**

- 외부 데이터 curriculum
- 길이 제어 DPO
- adaptive early stopping calibration

**P2 — 시간과 자원이 남을 때**

- GRPO
- 모델 출력만 사용하는 Best-of-N
- 고급 confidence calibration

**금지 또는 추가 승인 전 보류**

- 테스트 타임 TIR·Program-of-Thought
- Python·SymPy·solver 기반 답 계산
- 계산 verifier
- 문제별 동적 retrieval

## 15. 최종 준비 완료 정의

다음 항목이 모두 충족돼야 최종 시스템을 준비 완료로 간주한다.

- [ ] 모델과 tokenizer revision이 고정돼 있다.
- [ ] 훈련 데이터와 split의 SHA-256이 기록돼 있다.
- [ ] Random·Template·Hard set 결과가 저장돼 있다.
- [ ] Base, SFT, DPO, adaptive inference의 ablation이 있다.
- [ ] pass@k, majority@k, parse failure와 runtime이 측정돼 있다.
- [ ] 최종 추론이 모델 출력만으로 답을 도출한다.
- [ ] 전체 오프라인 리허설을 두 번 통과했다.
- [ ] 중단·재시작과 fallback 절차를 검증했다.
- [ ] 제출 형식 검증기를 통과했다.
- [ ] 코드, 가중치, 데이터와 환경 문서가 재현 가능하다.
- [ ] 최종 테스트 runbook을 팀 전체가 확인했다.

## 16. 로드맵 유지 규칙

- 체크박스는 실제 산출물이나 검증 로그가 있을 때만 완료 처리한다.
- 일정이 지연되면 P2, P1 순서로 범위를 줄이고 P0를 보호한다.
- 공식 규칙이나 운영진 답변이 바뀌면 먼저 [규칙 문서](../information/rules.md)를 수정한 뒤 이 로드맵을 갱신한다.
- 한 단계가 실패하면 결과를 숨기지 않고 실패 원인과 복귀 checkpoint를 기록한다.
- 리더보드 점수만으로 로컬 평가 계약을 변경하지 않는다.
