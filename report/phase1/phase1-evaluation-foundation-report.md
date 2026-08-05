# Phase 1 평가 기반 구축 결과 보고서

실행 환경: Vast.ai RTX 4090 원격 인스턴스  
실제 작업 경로: `/workspace/university-deep-learning-challenge-2026`  
artifact: `artifacts/experiments/p1_20260803T110900Z_eval-foundation_aa8e7253_s42`  
Git HEAD: `f7ba0809bf18697ee9d3f8f563c7796fab26fc75`

## 결론

Phase 1의 고정 split, 형식적 답 추출, B0/B1/B2 Base baseline, metric 집계,
offline reload 및 재현 검증을 완료했다. 최종 verification은 **PASS**이며 Phase 2
학습이나 Verified-CoT 생성은 수행하지 않았다.

## 데이터와 provenance

| 자산 | 경로 | 행 수 | schema | SHA-256 |
|---|---|---:|---|---|
| `leaderboard` | `data/deep_chal_math_leaderboard.csv` | 1000 | `id,question, answer` | `f00b83805479140fb4d59fedb01c092e16c6cd35ac588f387b281ffea55eb2d7` |
| `leaderboard_filtered` | `data/deep_chal_math_leaderboard_filtered.csv` | 831 | `id,question` | `032333a1361c8083093674ad19817e024c38dc7c9f4bdf05c0c9b0c71940dcf1` |
| `train` | `data/deep_chal_math_train.csv` | 17000 | `id,question,answer` | `94f3302a6240b91b6fb3d093696b898750b8c4ca1d8ae1eb54210358664af9df` |
| `train_filter_audit` | `data/deep_chal_math_train_filter_audit.csv` | 17000 | `id,question,answer,decision,primary_reason,reason_codes,reason_descriptions,confidence,question_length,has_visual_signal,evidence,question_sha256` | `3275fc5ad3dfc7ef6f3e699714c1e38302642040c88ab2c414c930b669b949dd` |
| `train_filtered` | `data/deep_chal_math_train_filtered.csv` | 16528 | `id,question,answer` | `2844386e4d5c7355f773ac58f4b735be4e9ce1caa70b4b1a3576244e1346ff98` |

- 원본 train과 leaderboard의 실행 전후 SHA-256은 동일하다.
- filtered train 16,528행은 versioned 임시 경로에서 행 내용과 byte hash가 모두 동일하게 재현됐다.
- 기존 train audit의 정책 필드와 newline 정규화 질문은 재현본과 동일하다. 기존 snapshot의
  embedded newline encoding 때문에 1977행의 질문 길이·hash 계열
  metadata는 byte-level로 달랐으며 기존 파일은 덮어쓰지 않았다.
- 기존 leaderboard 831행 파생본은 ID 순서를 재현했지만 원본과 질문 내용이 다른 행이
  62개다. 역사적 semantic 제외 정책이 없으므로
  이를 추측하지 않고 행별 원본/파생 hash와 legacy membership을 audit에 기록했다.

## 고정 split

| split | train | validation/diagnostic | seed | 누수 검사 |
|---|---:|---:|---:|---|
| Random | 14875 | 1653 | 42 | ID overlap 0 |
| Template-group | 14875 | 1653 | 42 | ID overlap 0, group leakage 0 |
| Hard diagnostic | — | 553 | 42 | deterministic category cap |
| Format diagnostic | — | 256 | 42 | actual-label cases; synthetic parser tests separate |

동일 입력으로 split 생성기를 두 번 실행했으며 12개 deterministic 출력의 SHA-256이 모두 일치했다.
Template 정규화는 NFKC와 숫자·인명 후보·통화·단위의 표면 치환만 수행하며 문제 풀이 또는
정답 계산을 하지 않는다. 숫자만 바뀐 문제를 같은 group으로 묶기 때문에 Random 대비
Template 결과가 일반화 위험을 더 보수적으로 나타낸다.

## 모델과 생성 설정

- model/tokenizer: `Qwen/Qwen2.5-3B-Instruct`
- pinned revision: `aa8e72537993ba99e69dfaafa59ed015b17504d1`
- `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `local_files_only=True`
- dtype: BF16
- ground truth는 생성 종료 후 metric 계산에만 로드했다.

| baseline | sampling | seeds | max new tokens | temperature | top-p | 최대 batch | token budget |
|---|---|---|---:|---:|---:|---:|---:|
| B0 | false | `[42]` | 1024 | None | None | 256 | 294912 |
| B1 | false | `[42]` | 1024 | None | None | 256 | 294912 |
| B2 | true | `[42, 2026, 3407]` | 1024 | 0.7 | 0.9 | 256 | 294912 |

B0는 최소 풀이·최종 marker 지시, B1은 단계별 풀이와 모델 내부 검산 지시,
B2는 B1 prompt에 세 seed sampling을 적용했다. 전체 prompt는 `configs/phase1.json`과
각 generation row에 보존했다.

배치는 라벨이나 모델 출력과 무관한 입력 token 길이로 정렬하고, 최대 256개 및
`(batch rows) × (max input tokens + max new tokens) <= 294,912` 제약으로 결정했다.
초기 혼합 순서 batch 128은 긴 prompt가 한 배치에 섞이며 OOM이 발생했고 그 log를 보존했다.
고정 batch 96 혼합 실행은 약 0.89 generation/s였지만, 최종 적응형 실행은 B0 중간 측정에서
약 1.77 generation/s, GPU 98~100%, VRAM 약 22.46/24.56 GiB를 기록했다.

## 결과

| baseline | scope | 문항 | greedy acc. | sample acc. | pass@k | majority@k | agreement@k | invalid | median tokens | p95 latency(s) | 1,000문항 예상(h) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | random | 1653 | 0.6213 | 0.6213 | 0.6213 | 0.6213 | 0.8693 | 0.1307 | 355.0 | 1.380 | 0.18 |
| B0 | template | 1653 | 0.6407 | 0.6407 | 0.6407 | 0.6407 | 0.8566 | 0.1434 | 339.0 | 1.380 | 0.19 |
| B0 | hard | 553 | 0.2532 | 0.2532 | 0.2532 | 0.2532 | 0.6908 | 0.3092 | 657.0 | 1.395 | 0.21 |
| B0 | format | 256 | 0.4023 | 0.4023 | 0.4023 | 0.4023 | 0.7695 | 0.2305 | 617.0 | 0.918 | 0.19 |
| B1 | random | 1653 | 0.6140 | 0.6140 | 0.6140 | 0.6140 | 0.8361 | 0.1639 | 509.0 | 0.668 | 0.17 |
| B1 | template | 1653 | 0.6261 | 0.6261 | 0.6261 | 0.6261 | 0.8312 | 0.1688 | 504.0 | 0.668 | 0.17 |
| B1 | hard | 553 | 0.2495 | 0.2495 | 0.2495 | 0.2495 | 0.6872 | 0.3128 | 714.0 | 2.113 | 0.21 |
| B1 | format | 256 | 0.4414 | 0.4414 | 0.4414 | 0.4414 | 0.7305 | 0.2695 | 657.0 | 0.754 | 0.17 |
| B2 | random | 1653 | — | 0.5787 | 0.6951 | 0.6449 | 0.6983 | 0.1712 | 506.0 | 0.724 | 0.56 |
| B2 | template | 1653 | — | 0.5977 | 0.7030 | 0.6594 | 0.7139 | 0.1760 | 497.0 | 0.724 | 0.56 |
| B2 | hard | 553 | — | 0.2411 | 0.3363 | 0.2948 | 0.4671 | 0.3002 | 704.0 | 2.155 | 0.67 |
| B2 | format | 256 | — | 0.3880 | 0.5039 | 0.4492 | 0.5573 | 0.2786 | 662.0 | 0.809 | 0.57 |

### Random–Template 차이

| baseline | metric | Random | Template | Random - Template |
|---|---|---:|---:|---:|
| B0 | `greedy_accuracy` | 0.6213 | 0.6407 | -0.0194 |
| B1 | `greedy_accuracy` | 0.6140 | 0.6261 | -0.0121 |
| B2 | `sample_accuracy` | 0.5787 | 0.5977 | -0.0190 |

### 답 추출 실패 유형

| baseline | 유형 | 건수 |
|---|---|---:|
| B0 | `conflicting_explicit_answers` | 15 |
| B0 | `no_supported_answer_marker` | 593 |
| B1 | `conflicting_explicit_answers` | 16 |
| B1 | `no_supported_answer_marker` | 687 |
| B2 | `conflicting_explicit_answers` | 43 |
| B2 | `no_supported_answer_marker` | 2124 |

Extractor는 마지막 `FINAL_ANSWER:`, 마지막 `\boxed{...}`, 명시적인 마지막 답 문장,
제한적인 독립 숫자 마지막 줄 순서만 지원한다. 상충 답·빈 출력·지원 marker 없음은 실패로
분류하며 쉼표·공백·부호·TeX fraction 표기만 정규화한다. 산술, 방정식 풀이, solver,
외부 서비스 또는 동적 retrieval은 없다.

## 재현·무결성 검증

- 독립 greedy 재실행: 16 generation, raw text 일치
  16/16.
- 독립 seeded sampling 재실행: 48 generation, 추출 답 일치
  48/48, accuracy 차이
  0.0000.
- Random ID overlap 0, Template ID overlap 0, Template group leakage 0.
- leaderboard audit는 1,000개 ID를 각각 한 번 포함한다.
- B0/B1/B2 generation은 누락·중복 없이 모든 Random·Template·Hard·Format ID를 포함한다.
- Phase 0 verification은 계속 PASS이며 Phase 0 artifact는 수정하지 않았다.
- focused test와 통합 verification log, raw generation, metric, source hash는 artifact에 보존했다.

## Git 상태와 변경 범위

초기 상태:

```text
 M README.md
 M docs/information/README.md
 M docs/information/data.md
 M docs/information/rules.md
 M docs/strategy/winning-strategy.md
?? AGENTS.md
?? artifacts/
?? configs/
?? data/deep_chal_math_leaderboard_filtered.csv
?? docs/strategy/phase0-rules-checklist.md
?? docs/strategy/roadmap.md
?? report/phase0/
?? scripts/build_phase1_report.py
?? scripts/collect_environment.py
?? scripts/create_evaluation_splits.py
?? scripts/evaluate_generations.py
?? scripts/extract_answers.py
?? scripts/phase0_smoke_inference.py
?? scripts/phase1_common.py
?? scripts/run_baseline.py
?? scripts/verify_data_provenance.py
?? scripts/verify_phase0.py
?? scripts/verify_phase1.py
?? tests/
```

최종 상태:

```text
 M README.md
 M docs/information/README.md
 M docs/information/data.md
 M docs/information/rules.md
 M docs/strategy/winning-strategy.md
?? AGENTS.md
?? artifacts/
?? configs/
?? data/deep_chal_math_leaderboard_filtered.csv
?? data/splits/
?? docs/strategy/phase0-rules-checklist.md
?? docs/strategy/roadmap.md
?? report/phase0/
?? report/phase1/
?? scripts/build_phase1_report.py
?? scripts/collect_environment.py
?? scripts/create_evaluation_splits.py
?? scripts/evaluate_generations.py
?? scripts/extract_answers.py
?? scripts/phase0_smoke_inference.py
?? scripts/phase1_common.py
?? scripts/run_baseline.py
?? scripts/verify_data_provenance.py
?? scripts/verify_phase0.py
?? scripts/verify_phase1.py
?? tests/
```

기존 modified/untracked 파일을 보존했다. stage, commit, branch 생성, push는 수행하지 않았다.

## 로드맵 반영

Phase 1 작업 10개는 split manifest, unit/integration test, raw generation, metric 및 최종
verification 근거로 완료 처리했다. 종료 조건 네 항목도 충족했다. Phase 2는 시작하지 않았다.

## 알려진 제약

- legacy leaderboard filtered 파일의 원래 semantic 필터 정책은 복원할 수 없다. 현재 audit는
  기존 membership과 원본 대비 질문 변환 차이를 정확히 고정한다.
- `agreement@k`는 정답성 증명이 아니라 모델 출력 문자열의 일치도다.
- runtime 추정은 이 RTX 4090, 적응형 token-budget batch와 현재 split에서 측정한 값이며
  다른 GPU에서는 달라진다.
