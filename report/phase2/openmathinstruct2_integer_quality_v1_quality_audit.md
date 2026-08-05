# OpenMathInstruct-2 integer-quality v1 감사 보고서

생성 시각(UTC): `2026-08-05T17:31:46.956617+00:00`

대상 버전: `openmathinstruct2_integer_quality_v1`

## 기술 요약

원본 50,000행을 수정하지 않고 판정했으며 canonical 정수 답 40,563행 중 39,450행을 integer-quality 데이터셋에 유지했다. 정수 형식만 통과한 행을 `verified`로 부르지 않았고, source·풀이 길이·기존 단순 등식 verifier 결과·coverage를 조합해 high/medium/low tier를 부여했다. 첫 사전 SFT 후보는 `high` tier 460행으로 제한했으며 canonical 정수행 중 40,103행은 첫 후보에서 제외했다.

## 답 형식과 포함 결과

| 분류 | 행 수 | 처리 |
|---|---|---|
| canonical integer | 40,563 | 품질 검사 후 tier 부여 |
| decimal | 3,165 | 변환·반올림 없이 제외 |
| fraction | 6,236 | 변환·반올림 없이 제외 |
| other | 36 | 비정규 형식으로 제외 |

## source와 quality tier 분포

| source | tier | 행 수 | F1 후보 | verifier not_checked |
|---|---|---|---|---|
| augmented_gsm8k | high | 416 | 416 | 0 |
| augmented_gsm8k | medium | 6,264 | 0 | 5,696 |
| augmented_gsm8k | low | 775 | 0 | 774 |
| augmented_gsm8k | excluded | 1,225 | 0 | 1,125 |
| augmented_math | medium | 58 | 0 | 0 |
| augmented_math | low | 31,164 | 0 | 31,064 |
| augmented_math | excluded | 9,150 | 0 | 9,106 |
| gsm8k | high | 43 | 43 | 0 |
| gsm8k | medium | 268 | 0 | 267 |
| gsm8k | excluded | 1 | 0 | 0 |
| math | high | 1 | 1 | 0 |
| math | medium | 459 | 0 | 459 |
| math | low | 2 | 0 | 2 |
| math | excluded | 174 | 0 | 173 |

Tier 점수는 config에 고정돼 있다. 원본 benchmark source(`gsm8k`, `math`)에 가장 높은 source 점수를, `augmented_gsm8k`에 중간 점수를, `augmented_math`에 가장 보수적인 점수를 부여한다. 60~450단어 풀이를 preferred, 35~650단어를 acceptable로 분류한다. verifier는 안전한 단순 이항 등식만 검사하며 복합식은 `not_checked`로 남긴다.

## 제외 사유와 탐지 사례

Primary exclusion 기준 10,550행을 제외했다. 한 행에 여러 사유가 있을 수 있으므로 아래 all-reason 합계는 제외 행 수보다 클 수 있다.

| 제외 사유 | primary 행 수 | all-reason 행 수 |
|---|---|---|
| non_integer_fraction_answer | 6,236 | 6,236 |
| non_integer_decimal_answer | 3,165 | 3,165 |
| external_visual_or_problem_code_dependency | 705 | 891 |
| detectable_self_contradiction | 287 | 324 |
| truncated_solution | 61 | 67 |
| non_integer_other_answer | 36 | 36 |
| simple_equation_verifier_failed | 29 | 50 |
| final_marker_inside_solution | 23 | 30 |
| tool_or_code_dependent_solution | 7 | 8 |
| manual_quality_audit_failed | 1 | 4 |
| abnormal_control_or_replacement_character | 0 | 0 |
| duplicate_or_blank_id | 0 | 0 |
| empty_solution | 0 | 0 |
| id_provenance_mismatch | 0 | 0 |
| messages_or_final_line_inconsistent | 0 | 0 |
| schema_error | 0 | 0 |

## verifier coverage는 품질 보증이 아니다

| verifier 상태 | 전체 행 수 |
|---|---|
| passed_full | 136 |
| passed_partial | 1,148 |
| not_checked | 48,666 |
| failed | 50 |

`not_checked`는 실패가 아니다. 복합식 또는 안전하게 독립 계산으로 해석할 수 없는 식은 계산하지 않았고, 행별 audit에 checked/not-checked 식 수와 coverage를 기록했다. 반대로 `passed_full`도 풀이 전체의 의미적 정확성을 증명하지 않는다.

## source × tier stratified audit

`data\phase2\openmathinstruct2_integer_quality_v1\openmathinstruct2_integer_quality_v1_stratified_quality_audit.csv`는 포함 행을 source × tier로, 제외 행을 source × primary reason으로 결정적 표본 추출한다. 수동 표본 판정은 pass 11건, fail 4건, uncertain 0건이다. 이 중 최종 high stratified 표본은 pass 11건, fail 0건, uncertain 0건이다.

| ID | tier | 판정 | 오류 유형 | 메모 |
|---|---|---|---|---|
| omi2-000004030 | excluded | fail | unsupported_final_answer | 전개는 1023과 1024만 계산한 뒤 근거 없이 boxed 342로 점프함; source 점수 하향의 근거 |
| omi2-000055022 | excluded | fail | detectable_self_contradiction | 같은 실수를 반복했다고 명시한 뒤 주어진 투영면 넓이를 만족하지 않는 치수를 채택함 |
| omi2-000015561 | excluded | fail | external_visual_or_problem_code_dependency | 문제가 shown here 및 Asymptote 코드에 의존해 텍스트 단독 학습 입력으로 부적합함 |
| omi2-000005831 | excluded | fail | detectable_self_contradiction | 총 지급액 1650과 450이 같아야 한다고 한 뒤 계산을 무시하고 1650을 채택함 |

## contamination provenance와 원본 보호

입력 provenance는 대회 train 17,000행과 leaderboard 원본 1,000행에 대해 exact, normalized-template, token-trigram Jaccard 근접 중복 검사를 기록하며 accepted match는 exact/template 0건, near 0건이다. 새 데이터셋은 이 입력의 행 부분집합이고 문제·풀이·답을 변경하지 않으므로 기존 decontamination 조건을 상속하며 새 오염을 만들지 않는다.

## 재현 명령과 산출물

```powershell
python scripts/refine_openmathinstruct2_integer_quality.py --config configs/openmathinstruct2_integer_quality_v1.json
```

- 입력: `data\phase2\phase2_verified_cot_luna_budget5_v1\openmathinstruct2_curriculum_v1.jsonl` (`0e61b2abef33fa303f98474830ad28f8e3ed32e3c4ab94412ac71934de9ae8c8`)
- 데이터: `data\phase2\openmathinstruct2_integer_quality_v1\openmathinstruct2_integer_quality_v1.jsonl`
- 전체 audit: `data\phase2\openmathinstruct2_integer_quality_v1\openmathinstruct2_integer_quality_v1_row_audit.csv`
- tier 통계: `data\phase2\openmathinstruct2_integer_quality_v1\openmathinstruct2_integer_quality_v1_tier_stats.csv`
- stratified audit: `data\phase2\openmathinstruct2_integer_quality_v1\openmathinstruct2_integer_quality_v1_stratified_quality_audit.csv`
- F1 후보 ID: `data\phase2\openmathinstruct2_integer_quality_v1\openmathinstruct2_integer_quality_v1_f1_candidate_ids.txt`
- manifest: `data\phase2\openmathinstruct2_integer_quality_v1\openmathinstruct2_integer_quality_v1_manifest.json`

## 한계와 다음 검증

- 자동 검사는 표면적 일관성, 강한 truncation·tool 의존 신호, 명시적 자기모순과 제한된 단순 등식만 탐지한다. 문제를 다시 풀거나 복합식을 계산하지 않는다.
- OpenMathInstruct-2의 expected answer 자체가 틀렸거나 자연어 논리가 미묘하게 잘못된 경우는 남을 수 있다.
- 단순 나눗셈의 반올림 표기처럼 수학적으로 의도된 근삿값도 exact verifier에서 실패할 수 있어 보수적 false positive가 가능하다.
- high tier도 correctness proof가 아니다. 첫 학습 전에 high tier의 추가 수동 표본과 짧은 SFT ablation으로 일반화 손실을 확인해야 한다.
