# Phase 0 규칙 확인 체크리스트

확인일: 2026-08-03 (KST)

근거 문서:

- `AGENTS.md`
- `docs/information/rules.md`
- `docs/strategy/roadmap.md` 단계 0 및 종료 조건
- `docs/strategy/winning-strategy.md`의 평가 환경, QLoRA, 재현성 지침
- 2026-08-01 Discord 운영진 답변(추론 중 코드 실행·도구 호출 금지)

## 모델 및 재현성

- [x] 유일한 베이스 모델은 `Qwen/Qwen2.5-3B-Instruct`이다.
- [x] 모델과 tokenizer는 같은 전체 commit SHA `aa8e72537993ba99e69dfaafa59ed015b17504d1`를 사용한다.
- [x] 모든 `from_pretrained` 호출은 명시적 `revision`을 요구한다.
- [x] 모델 cache는 저장소 밖 `/workspace/.cache/huggingface`에 둔다.
- [x] 공통 seed는 `[42, 2026, 3407]`이다.
- [x] 실험 ID와 산출물 경로 규칙은 `configs/phase0.json`에 고정했다.
- [x] 실험 manifest는 로드맵의 experiment contract 필드를 포함한다.
- [x] 외부 데이터 manifest는 출처, 라이선스, revision, 접근일, 변환, 행 수, hash와 오염 검사를 요구한다.

## 테스트 타임 금지 사항

- [x] 모델 생성 Python 또는 다른 코드를 실행하지 않는다.
- [x] TIR·Program-of-Thought 실행 피드백을 사용하지 않는다.
- [x] Python, SymPy, SAT/SMT, 수치 solver 또는 참가자 작성 수학 함수를 사용하지 않는다.
- [x] 계산 verifier로 후보를 교정하거나 재순위화하지 않는다.
- [x] 외부 계산 결과를 모델에 되먹이지 않는다.
- [x] 인터넷, 웹 검색, 외부 API 또는 외부 모델을 사용하지 않는다.
- [x] 별도 운영진 허용 전에는 문제별 동적 BM25·embedding retrieval을 사용하지 않는다.
- [x] 답 추출은 모델이 출력한 `FINAL_ANSWER:` 표기를 읽을 뿐 새 계산을 하지 않는다.

## 허용 범위와 데이터 보호

- [x] 모델 출력 여러 개의 majority voting과 self-consistency는 허용된다.
- [x] 답 표기의 형식적 추출·정규화와 비수학적 제출 검사는 허용된다.
- [x] 리더보드와 최종 테스트 질문을 외부 서비스로 전송하지 않는다.
- [x] 리더보드 원본 1,000행은 학습 데이터로 사용하지 않는다.
- [x] `deep_chal_math_leaderboard_filtered.csv`를 보호 또는 제출 범위를 축소하는 근거로 사용하지 않는다.
- [x] 원본 train/leaderboard CSV는 불변 자산이며 전후 SHA-256을 검증한다.

Phase 0 smoke inference는 짧은 synthetic 문제만 사용하고, 모델 출력 외의 계산·solver·retrieval 경로를 구현하지 않는다.
