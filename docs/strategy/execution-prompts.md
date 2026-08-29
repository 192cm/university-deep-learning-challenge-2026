# 원격 실행 프롬프트 모음

[수상 전략](winning-strategy.md)을 vast.ai 원격 서버에서 실행하기 위한 작업별 프롬프트다. 각 프롬프트는 **새 세션에 그대로 붙여넣어 단독 실행**할 수 있도록 작성했다. 앞의 「공통 컨텍스트」를 먼저 붙이고 그 뒤에 해당 작업 프롬프트를 붙인다.

원칙 하나: **한 작업이 끝날 때마다 완료 조건을 실제로 확인하고 다음으로 넘어간다.** 완료 조건을 못 채우면 다음 작업을 시작하지 않는다.

---

## 공통 컨텍스트 (모든 프롬프트 앞에 붙일 것)

```text
[프로젝트]
Kaggle "아주소중한딥러닝챌린지 2026" — Qwen2.5-3B-Instruct를 미세조정해 수학 문제의 정수 답을 맞히는 대회.
평가: Accuracy (Exact Match), 정답은 항상 정수. 제출은 submission.csv (모든 문항, 빈 값은 오답).

[원격 환경]
접속: ssh -p 41829 root@84.67.29.50 -L 8080:localhost:8080
작업 디렉터리: /workspace
로컬 저장소: C:\Users\kyle0\Develops\university-deep-learning-challenge-2026
파일 전송은 tar로 묶어 scp -P 41829 <archive.tar.gz> root@84.67.29.50:/workspace/ 로 보내고 원격에서
푼다. (포트 플래그가 scp는 -P, ssh는 -p 로 다르다.)

[베이스 모델 — 변경 불가]
Qwen/Qwen2.5-3B-Instruct
revision: aa8e72537993ba99e69dfaafa59ed015b17504d1 (tokenizer도 동일 revision으로 고정)
다른 모델을 베이스로 쓰거나 가중치를 병합하는 것은 실격 사유다. 사전학습은 금지, 미세조정만 허용.

[절대 규칙 — 추론 시]
운영진 답변: "추론 시에는 모델의 추론 출력만으로 답을 도출하는 것을 원칙으로 하며, 코드 실행 및 도구 호출은
허용하지 않습니다. Majority Voting, Self-Consistency 등 다중 샘플링 기반의 test-time 기법은 자유롭게 활용
하실 수 있습니다."

따라서 최종 추론 파이프라인에서 금지:
  - 모델이 만든 코드를 실행하는 TIR / Program-of-Thought
  - Python, SymPy, solver, 수치 해석기, 사전 작성 계산 함수
  - 계산 결과를 모델 입력으로 되먹임하는 방식
  - 계산 verifier로 답을 고치거나 후보 순위를 바꾸는 방식
  - 문항별 동적 BM25/embedding retrieval

허용:
  - 다중 샘플링, Majority Voting, Self-Consistency, Best-of-N
  - 답 일치도만 보는 adaptive sampling / early stopping
  - 같은 모델·같은 어댑터가 자기 후보를 읽고 고르는 GenSelect
  - 동일 베이스에서 학습한 별도 verifier/ORM 어댑터로 후보를 채점·선별하는 Best-of-N
  - 모델이 이미 출력한 문자열을 읽고 표기만 정규화하는 답 추출기

답 추출기 철칙: 추출기는 문자열을 읽기만 한다. 어떤 산술도 하지 않는다.
  금지 예: eval(), sympy 파싱, "12 + 5"를 보고 17을 만들기, 분수를 정수로 환산하기
  허용 예: FINAL_ANSWER: 뒤 정수 읽기, \boxed{-3} 에서 -3 읽기, "1,234" → "1234", U+2212 → "-"

[테스트 데이터 보호]
- 리더보드/최종 테스트 문항을 외부 API·검색엔진·외부 모델에 절대 보내지 않는다.
- 리더보드 문항은 라벨이 없어도 학습 데이터로 쓰지 않는다.
- 외부 학습 데이터는 리더보드 원본 1,000행 전체와 오염 검사를 한다. (831행 필터본 기준으로 하면 169문항이 검사에서 누락된다)
- 추론·제출 범위는 항상 원본 전체 행이다.

[데이터 — 원본 4종이 전부]
data/deep_chal_math_train.csv              17,000행  id,question,answer
data/deep_chal_math_leaderboard.csv         1,000행  id,question," answer"  ← 세 번째 열 이름 앞에 공백이 있다
data/train_filtered_ids.csv                   627행  대회 측 훈련 제외 목록
data/deep_chal_math_leaderboard_filtered.csv  831행  리더보드 169행 제외본 (사용하지 않음)

canonical train = train.csv(17,000) − train_filtered_ids.csv(627) = 16,373행

CSV를 읽을 때는 항상 헤더를 strip해서 컬럼명 공백에 당하지 않게 한다.

[train_filtered_ids.csv의 성격 — 2026-08-20 재검증, 중요]
구조: 627개 id가 전부 train에 존재(누락 0). answer는 627행 모두 train과 동일(정정본이 아니다).
      question 차이 352건은 전량 공백·개행 정규화. 리더보드와의 질문·템플릿 일치 0건(오염 제거 목적이 아니다).
      즉 순수한 제외 목록이므로 id 기준 뺄셈이 맞다.

성격: 이미지 필터가 아니다. 결함 표지 분해가 이렇다.
      이미지/URL/[asy] 174 (27.8%) / 그림 참조 149 / 빈칸 placeholder 34 / 표지 없음 270 (43.1%)
      표지 없는 270행은 텍스트 파손과 라벨 오류다. 예:
        train-000189  "(n + 5)(n - 5)(n-15) 7$?"     부등호가 유실된 파손 문항
        train-000222  질문 전문이 "Defeated number solution.", answer=0
        train-000176  제약식 중복 서술, 답 0이 성립하지 않음

따라서: canonical 16,373에도 같은 계열의 파손 문항·오답 라벨이 남아 있다. 비율은 아직 모른다.
        이것을 사람 눈이나 휴리스틱으로 잡으려 하지 말 것 — 시간만 쓰고 재현도 안 된다.
        대신 RFT의 문항별 정답 일치 샘플 수 c 를 라벨 신뢰도 대리 지표로 쓴다 (T5, T7).
        exact match 검증기는 라벨이 틀린 문항에서 같이 틀린다. 위험한 쪽은 하나다:
          틀린 풀이가 틀린 라벨과 우연히 일치해 오답 풀이가 SFT에 채택되는 경우.

[기존 실측 베이스라인 — 이전 인프라에서 측정, 참고용]
같은 base 모델, greedy, max_new_tokens=1024 기준
  random holdout greedy accuracy   62.5%
  template holdout                 63.6%
  hard holdout                     24.5%
  invalid_output_rate (random)     14.7%
  majority@3 (T=0.7)               65.6%
  pass@3                           71.0%
  처리량 (HF generate, RTX 4090)   1.66 generations/sec
  문제 유형별: 산술문장제 70.9% / 조합확률 43.1% / 정수론 41.0% / 대수 39.2% / 기하 30.7%
새 split은 seed가 달라 정확히 일치하지 않는다. ±2pp 범위면 정상이다.

[작업 규율]
- 한 번에 변수 하나만 바꾼다.
- 모든 산출물 디렉터리에 manifest.json을 남긴다: 입력 파일 SHA-256, config SHA-256, seed, 환경 정보, 지표.
- 원시 생성 결과(generations.jsonl)는 절대 지우지 않는다. 점수 변화의 원인을 되짚을 유일한 근거다.
- 개선을 주장하기 전에 고정 holdout에서 수치로 확인한다.
- 작업 완료 시 완료 조건을 하나씩 실제로 검증하고 결과를 보고한다.
```

---

## T0 — 원격 환경 부트스트랩 · 데이터 전송 · GPU 프로파일

```text
[목표]
원격 서버를 작업 가능 상태로 만들고, GPU의 실제 성능 한계를 측정해 이후 모든 작업의 배치 설정 근거를 만든다.

[작업]
1. 로컬에서 data/ 와 docs/ 를 tar.gz로 묶어 원격 /workspace 로 전송하고 푼다.
   전송 전후로 각 파일의 SHA-256과 행 수를 비교해 무결성을 확인한다.
   참고: docs/information/data.md에 적힌 train/leaderboard 해시는 현재 파일과 다르다. 문서 쪽이 오래된 값이므로
   원격에서 실제 측정한 해시를 새 기준으로 기록하고, 이 불일치를 manifest에 명시한다.
   검증은 해시 일치가 아니라 "전송 전후 동일" + "행 수 17,000 / 1,000 / 627 / 831"로 한다.

2. 원격 환경을 조사해 environment.json에 기록한다.
   - GPU 모델명, 개수, VRAM 총량 (nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv)
   - CUDA 버전, 드라이버 버전
   - CPU 코어 수, RAM, 디스크 여유 공간
   - Python 버전, torch/transformers/peft/trl/datasets/accelerate/bitsandbytes 버전
   - bf16 지원 여부, GPU compute capability
   GPU가 RTX 4090 24GB라고 가정하지 말고 실제로 측정한다. 이후 모든 배치 설정은 이 값에서 역산한다.

3. 필요한 패키지를 설치하고 버전을 requirements.lock 으로 고정한다.
   torch, transformers, peft, trl, datasets, accelerate, bitsandbytes 는 필수.
   vllm은 T3에서 판단하므로 여기서는 설치만 시도하고 실패해도 진행한다.

4. 베이스 모델을 고정 revision으로 다운로드하고 로컬 캐시에 둔다.
   Qwen/Qwen2.5-3B-Instruct revision aa8e72537993ba99e69dfaafa59ed015b17504d1
   다운로드 후 실제 로드된 revision을 확인해 기록한다.

5. GPU 메모리 상한을 실측한다.
   - bf16으로 모델만 로드했을 때 점유 VRAM
   - 4-bit NF4로 로드했을 때 점유 VRAM
   - 각각에서 남는 VRAM = KV 캐시와 배치에 쓸 수 있는 예산
   이 세 수치를 environment.json에 넣는다. T3(생성)과 T6(학습)이 이 값을 근거로 배치를 정한다.

6. 짧은 스모크 추론: 문제 8개를 같은 seed로 두 번 생성해 결과가 바이트 단위로 동일한지 확인한다.
   환경변수 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 를 기본으로 설정한다.

[완료 조건]
- /workspace 에 data/ 4개 파일이 있고 행 수가 17,000 / 1,000 / 627 / 831 이다
- environment.json에 GPU 모델·VRAM·bf16/4bit 로드 시 잔여 VRAM이 기록되어 있다
- 모델이 지정 revision으로 로드되고, 같은 seed의 두 번 생성이 동일하다
- requirements.lock 이 존재한다

[산출물]
/workspace/environment.json, /workspace/requirements.lock, /workspace/artifacts/t0_bootstrap/manifest.json
```

---

## T1 — 답 추출기와 평가기

```text
[목표]
src/extract.py 와 src/evaluate.py 를 만든다. 이 둘이 이후 모든 단계의 채택/기각 판단 근거이므로 가장 먼저 만든다.
GPU를 쓰지 않으므로 로컬에서 작성해 전송해도 되고 원격에서 바로 작성해도 된다.

[src/extract.py]
모델 출력 문자열 하나를 받아 정수 문자열 하나 또는 실패 사유를 반환한다.

추출 우선순위:
  1. 마지막 "FINAL_ANSWER:" 뒤의 정수
  2. 마지막 \boxed{...} 안의 정수
  3. 마지막 줄이 정수 하나로만 이루어진 경우 그 정수
  4. 본문에서 마지막으로 등장하는 정수
  5. 전부 실패 → 실패 사유와 함께 반환 (제출 단계에서만 "0"으로 대체)

정규화(표기만, 계산 금지):
  - 천단위 콤마 제거: "1,234" → "1234"
  - 유니코드 마이너스 U+2212, 전각 하이픈 → ASCII "-"
  - 선행 "+" 제거, 전각 숫자 → 반각
  - 후행 마침표·단위·문장부호 제거
  - 최종 형태는 정규식 ^-?(?:0|[1-9][0-9]*)$ 를 만족해야 한다
  - "-0"은 "0"으로 정규화

절대 금지: eval, sympy, 산술 연산, 분수/소수 → 정수 환산, 여러 숫자를 조합해 새 값 만들기.
소수점이나 분수 형태만 있는 경우는 "정수 아님"으로 실패 처리한다. 반올림하지 않는다.

반환 정보: 추출된 정수, 사용된 경로(final_answer_marker / boxed / standalone_last_line / last_integer / none),
실패 사유(no_supported_answer_marker / conflicting_explicit_answers / non_integer_only).

[src/evaluate.py]
생성 결과 jsonl과 정답 라벨을 받아 다음 지표를 계산한다.
  greedy_accuracy, sample_accuracy, pass@k, majority@k, agreement@k, tie_rate
  invalid_output_rate, 파싱 경로별 분포, 실패 사유별 건수
  median/p95 output_tokens, hit_max_new_tokens_rate
  문항 유형별 accuracy, 문항 길이 구간별 accuracy
  처리량(generations/sec), 1,000문항 추정 소요 시간

계약:
  - 정답 라벨은 지표 계산에만 쓴다. 후보 선택에는 절대 쓰지 않는다.
  - majority 동점 시 "먼저 생성된 답"을 택한다. 정답을 참조하지 않는다.
  - 수식 동치 판정이나 계산 verifier를 쓰지 않는다. 정규화 후 문자열 exact match만 한다.

[테스트]
tests/test_extract.py 에 최소 다음 케이스를 넣고 전부 통과시킨다.
  음수, 0, "-0", 10자리 초과 대정수(예: 3431577212128939), 천단위 콤마,
  \boxed{} 안의 음수, FINAL_ANSWER 뒤에 단위가 붙은 경우, 마커가 여러 번 등장하는 경우,
  마커 없이 본문만 있는 경우, 소수/분수만 있는 경우(실패해야 함), 빈 문자열,
  출력이 중간에 잘린 경우, "FINAL_ANSWER: 42." 처럼 마침표가 붙은 경우

[완료 조건]
- tests/test_extract.py 전부 통과
- evaluate.py가 더미 jsonl에 대해 위 지표를 모두 산출한다
- 추출기 코드 어디에도 산술 연산이 없음을 확인했다 (검색으로 eval/sympy/+ 연산 확인)

[산출물]
src/extract.py, src/evaluate.py, tests/test_extract.py
```

---

## T2 — 데이터 재구축 (canonical · holdout · RFT pool)

```text
[목표]
원본 4종에서 이후 모든 실험이 공유할 고정 데이터셋을 만든다. GPU 불필요.

[작업]
1. canonical train 생성
   deep_chal_math_train.csv(17,000) − train_filtered_ids.csv(627 id) = 16,373행
   원본 train.csv는 절대 수정하지 않는다.

   제외 목록이 "제외 목록"이 맞는지 세 가지를 assert한다. 하나라도 깨지면 멈추고 보고한다.
     a) 627개 id가 전부 train에 존재하고 중복이 없다
     b) 627행의 answer가 train의 answer와 전부 동일하다  ← 다르면 이 파일은 정정본이고 전략이 바뀐다
     c) question 차이는 공백·개행 정규화뿐이다 (re.sub(r'\s+',' ',s).strip() 비교로 확인)
   기대값: a) 627/627 존재, b) 차이 0건, c) 차이 352건이 전부 공백 정규화

2. 이미지 의존 문항 표시
   canonical에 남은 이미지·URL·[asy] 포함 문항(약 42개)에 플래그를 단다.
   삭제하지 말고 플래그만 단다. SFT 대상에서만 제외하고 분석용으로는 남긴다.

3. holdout 4종을 고정 seed로 생성한다. 서로 겹치는 id는 audit에 기록한다.
   a) random holdout — canonical의 약 10%. 질문 길이, 답 부호, 답 크기 구간을 층화한다.
   b) template-group holdout — 숫자/인명/단위를 정규화해 유사 템플릿을 그룹화하고,
      같은 그룹이 train과 validation에 동시에 들어가지 않게 한다. 약 10%.
   c) hard diagnostic — 기하/정수론/조합, 긴 문장제, 큰 정수 답 문항을 모은다. 약 550개.
   d) format diagnostic — 음수, 0, 10자리 초과 대정수, LaTeX 많은 출력 유도 문항. 약 256개.

4. RFT pool = canonical − (위 4종 holdout에 속한 모든 id). 약 12,600행 예상.

5. answer-only 대조군 데이터셋 생성 (T6의 대조군 2번용)
   RFT pool과 동일 범위, assistant 타겟은 "FINAL_ANSWER: <answer>" 한 줄.

6. 전체 행별 판정 audit CSV와 manifest.json을 남긴다.
   같은 config로 두 번 생성했을 때 산출물 SHA-256이 동일해야 한다. 실제로 두 번 돌려 확인한다.

[완료 조건]
- canonical 16,373행, 627개 제외 id 전량 확인
- holdout 4종의 id가 RFT pool과 교집합이 없음
- 동일 config 2회 생성 시 모든 산출물 해시 일치
- audit CSV가 canonical 전체 행을 커버

[산출물]
data/canonical/, data/splits/, data/rft_pool_ids.txt, data/answer_only/, 각 디렉터리의 manifest.json
```

---

## T3 — 생성기 · GPU 처리량 캘리브레이션 · 베이스라인 재측정

```text
[목표]
src/generate.py 를 만들고, 이 GPU에서 낼 수 있는 최대 처리량을 실측으로 찾은 뒤, 새 holdout에서 B0 베이스라인을 다시 찍는다.
이 작업의 캘리브레이션 결과가 T5(20만 건 생성)의 소요 시간을 좌우한다. 여기서 대충 하면 뒤에서 몇 배로 갚는다.

[1단계 — 엔진 선택]
vLLM과 HF generate 둘 다 짧게 벤치마크하고 빠른 쪽을 택한다.
  - 같은 문항 200개, 같은 설정으로 generations/sec를 잰다
  - vLLM이 HF 대비 5배 미만이면 설정이 잘못된 것이다. 아래 항목을 점검한 뒤 재측정한다
  - vLLM 설치나 구동이 30분 안에 안 되면 미련 없이 HF로 확정하고 진행한다
기존 실측 기준선: HF generate, RTX 4090, batch 256에서 1.66 generations/sec

[2단계 — GPU 활용 극대화]
목표는 생성 중 GPU SM 사용률 90% 이상을 지속하는 것이다.
별도 셸에서 nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 2 로 관찰하며 튜닝한다.
사용률이 60% 아래로 머물면 배치가 작거나 패딩 낭비가 크거나 CPU 전처리가 병목인 것이다.

vLLM을 쓰는 경우:
  - gpu_memory_utilization: 0.85에서 시작해 0.90 → 0.92로 올리며 OOM 직전까지 밀어붙인다
  - max_model_len: 실제 필요한 값(입력 2048 + 출력 2048 = 4096)으로 낮게 잡는다. 크게 잡으면 KV 캐시가 낭비된다
  - max_num_seqs: 256에서 시작해 올린다. VRAM이 남으면 계속 올린다
  - enable_prefix_caching=True: 같은 문제에 k개 샘플을 뽑을 때 프롬프트 prefix가 공유되어 큰 이득이 있다
  - 한 문제에 k개 샘플이 필요하면 SamplingParams(n=k)로 한 번에 요청한다. k번 따로 호출하지 않는다
  - dtype은 bfloat16

HF generate를 쓰는 경우:
  - tokenizer.padding_side="left" (디코더 모델의 배치 생성에 필수)
  - 입력을 토큰 길이로 정렬해 배치를 구성한다. 길이가 비슷한 것끼리 묶어야 패딩 낭비가 없다
  - 고정 batch_size가 아니라 토큰 예산으로 배치를 만든다:
    batch_tokens = batch_size × (max_input_len + max_new_tokens) 가 예산을 넘지 않게
    기존 실적 참고값: batch_size 256, max_batch_tokens 294912
  - attn_implementation="sdpa" (또는 flash_attention_2 설치 가능하면 그쪽)
  - use_cache=True, dtype=bfloat16
  - 배치 크기 이진 탐색: 64 → 128 → 256 → 512로 올리며 OOM 나기 직전 값의 90%를 채택
  - OOM이 나면 torch.cuda.empty_cache() 후 배치를 절반으로 줄여 자동 재시도하고, 이 사건을 로그에 남긴다

공통:
  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  - 생성 중 CPU 전처리(토크나이즈)를 미리 일괄 수행해 GPU가 놀지 않게 한다
  - 결과는 스트리밍으로 jsonl에 append한다. 중간에 죽어도 이어서 할 수 있게 resume을 지원한다

[3단계 — 캘리브레이션 결과 기록]
calibration.json에 다음을 남긴다:
  선택한 엔진과 이유, 시도한 배치 설정별 (배치크기, generations/sec, peak VRAM, GPU 사용률),
  채택한 최종 설정, 20만 건 생성 예상 소요 시간

[4단계 — B0 베이스라인 재측정]
설정: greedy(do_sample=False), max_new_tokens=1024, max_input_tokens=2048, seed 42
프롬프트: "Solve the following problem. Write the final answer on the last line exactly as
FINAL_ANSWER: <answer>. Do not write anything after that line.\n\nProblem:\n{question}"
대상: 새 holdout 4종 전부

[완료 조건]
- random holdout greedy accuracy가 60~65% 범위에 들어온다 (이전 실측 62.5% ±2pp)
- invalid_output_rate가 12~18% 범위로 재현된다 (이전 14.7%)
- calibration.json에 배치 스윕 결과와 채택 근거가 기록되어 있다
- 채택 설정에서 GPU 사용률이 90% 이상 지속됨을 확인했다
- 같은 seed로 두 번 돌린 greedy 결과가 동일하다

[산출물]
src/generate.py, artifacts/t3_baseline/{generations.jsonl, metrics.json, calibration.json, manifest.json}
```

---

## T4 — [A] 출력 계약 수리

```text
[목표]
수학 실력을 전혀 올리지 않고 얻는 점수를 회수한다. 전략 문서에서 가장 저렴한 레버다.

[근거 — 이전 실측]
B0 생성 3,719건 중 답 추출 실패 625건(16.8%).
그중 61.4%가 1024 토큰 상한에 걸려 잘린 경우.
토큰 상한에 도달한 396건 중 384건(97.0%)이 추출 실패 = 사실상 확정 0점.
나머지 38.6%는 EOS로 정상 종료했는데 지정 형식을 안 지킨 경우.

[작업]
1. max_new_tokens를 1024 → 2048로 올리고 T3와 동일 조건으로 재측정한다.
   토큰 상한 도달률과 invalid_output_rate가 얼마나 떨어지는지, accuracy가 얼마나 오르는지 각각 기록한다.
   출력이 길어지므로 T3의 배치 설정을 그대로 쓰면 OOM이 날 수 있다.
   토큰 예산 기준으로 배치를 다시 잡고(같은 batch_tokens 예산 유지, batch_size 축소),
   재캘리브레이션 결과를 calibration.json에 추가한다.

2. T1의 다단계 fallback 추출기를 적용해 재채점한다.
   중요: 이미 저장된 T3 generations.jsonl을 재파싱하는 것만으로 fallback 효과를 먼저 측정한다.
   생성을 다시 하지 않고 얻는 이득이 얼마인지 분리해서 봐야 한다.

3. 두 변경의 기여를 분해해 표로 만든다.
   (a) 원래 설정 + 원래 추출기       ← T3 결과
   (b) 원래 설정 + fallback 추출기   ← 재파싱만
   (c) 2048 토큰 + fallback 추출기   ← 최종
   각 행에 accuracy, invalid_rate, 토큰 상한 도달률, 평균 출력 토큰, 소요 시간을 넣는다.

4. format diagnostic 256문항으로 회귀 검증한다. 음수/0/대정수 처리가 깨지지 않았는지 본다.

[완료 조건]
- format split invalid rate < 3%
- random split invalid rate < 5%
- (c)가 (a)보다 random holdout accuracy에서 개선되었다. 개선폭을 pp로 보고한다
- 분해 표 3행이 모두 채워졌다
- 2048 토큰 설정에서 OOM 없이 GPU 사용률 90% 이상 유지

[산출물]
artifacts/t4_output_contract/{metrics_a.json, metrics_b.json, metrics_c.json, ablation.md, manifest.json}
```

---

## T5 — RFT R1 대량 생성 + 외부 CoT 재구축 (GPU/CPU 병렬)

```text
[목표]
유료 teacher 없이 SFT 데이터를 만든다. 이 프로젝트에서 GPU를 가장 오래 쓰는 작업이다.
과거에 유료 API teacher로 같은 데이터를 만들려다 3회 모두 실패하고 산출 0행으로 끝났다. 그 경로는 폐기한다.

[GPU 작업 — RFT R1]
1. RFT pool 약 12,600문제에 대해 문제당 k=16 샘플을 생성한다. 총 약 20만 건.
   설정: T=0.8, top_p=0.95, max_new_tokens=2048, T4에서 확정한 프롬프트
   정답 라벨을 프롬프트에 절대 넣지 않는다. 답을 알려주면 틀린 라벨에 억지로 끼워 맞추는 풀이가 나온다.

2. GPU 활용:
   - T3 calibration.json의 채택 설정에서 시작한다
   - vLLM이면 SamplingParams(n=16)으로 한 문제당 한 번만 요청한다. prefix 공유로 큰 이득이 있다
   - HF면 같은 프롬프트를 16번 복제해 한 배치에 넣지 말고, 서로 다른 문제를 길이순으로 묶어 배치를 채운다
   - 20만 건은 중간에 끊길 수 있다. 문제 id 단위 체크포인트와 resume을 반드시 구현한다
   - 진행률과 남은 시간 추정을 주기적으로 출력한다
   - 시작 후 10분 시점에 실제 처리량을 재고, calibration 예상치와 2배 이상 차이 나면 중단하고 설정을 다시 잡는다

3. 채택 필터 (src/build_rft.py)
   - 추출된 정수가 canonical 라벨과 exact match일 때만 채택
   - 문항별 정답 일치 샘플 수를 c 로 정의하고 audit에 반드시 기록한다.
     c 는 라벨 신뢰도의 대리 지표다. T7이 이 값을 그대로 받아 쓴다. 계산 비용은 0이다.
   - c 에 따라 채택 개수를 다르게 한다:
       c >= 2  → 최대 2~4개 채택. 쉬운 문제가 데이터를 지배하지 않게 상한을 둔다
       c == 1  → 1개만 채택. 틀린 풀이가 틀린 라벨과 우연히 일치했을 수 있는 구간이다
       c == 0  → 채택 없음. T7의 R2 대상이 된다
   - 채택 후보 중 짧은 것을 우선한다
   - 마지막 줄이 정확히 "FINAL_ANSWER: <integer>" 가 되도록 정리한다. 뒤에 아무것도 붙이지 않는다
   - 이미지 의존 플래그가 붙은 문항은 제외한다
   - 오답 풀이(= 틀린 답을 낸 생성)도 버리지 말고 따로 저장한다. T9 GenSelect 학습 데이터가 된다

4. 수확량 점검
   예상: 약 83%인 10,400문제 내외에서 최소 1개 채택 (pass@16 추정치 기반)
   실제 수확률이 70% 아래면 온도나 프롬프트를 점검한다. 90% 위면 예상보다 좋은 것이니 그대로 간다.

   단, 수확률 부족분 전부를 설정 탓으로 돌리지 말 것. canonical에는 파손 문항과 오답 라벨이
   남아 있고(공통 컨텍스트 참조), 그런 문항은 어떤 온도·프롬프트로도 c=0 이다.
   온도를 두 번 조정해도 수확률이 안 오르면 더 만지지 말고 T7으로 넘어간다.
   c=0 집합의 성격은 T7에서 k=32 재샘플로 판정한다.

   c 분포(c=0 / c=1 / c=2~3 / c>=4 문항 수)를 metrics.json에 남긴다. 발표 자료에 쓴다.

[CPU 병렬 작업 — 외부 CoT]
GPU가 도는 동안 CPU에서 동시에 진행한다. 서로 자원을 다투지 않는다.
1. OpenMathInstruct-2 (nvidia/OpenMathInstruct-2, CC-BY-4.0)를 HF에서 스트리밍으로 받는다.
   전체 14M행을 다 받지 말고 5만~10만 행만 스트리밍한다.
2. 필터: 최종 답이 정수인 것만 남긴다. 소수/분수/기타 형태는 제외.
   이전 실적: 5만 행 입력 → 정수 답 39,450행(78.9%) 잔존
   추가 제외: 시각 자료 의존, 코드 의존 풀이, 잘린 풀이, 자기모순 풀이, 지나치게 긴 풀이
3. 형식 정규화: assistant 타겟의 마지막 줄을 "FINAL_ANSWER: <integer>" 로 통일한다.
4. 오염 검사 (필수): 리더보드 원본 1,000행 전체와 대조한다.
   - 질문 문자열 exact match
   - 숫자 정규화 후 템플릿 match
   - 근접 중복 (정규화 후 토큰 유사도)
   하나라도 걸리면 제외하고 제외 목록을 audit에 남긴다.
5. 최종 15,000행을 층화 샘플링한다. 대회 데이터의 길이/주제 분포에 가깝게 뽑는다.

[완료 조건]
- RFT R1 채택 문제 수와 총 샘플 수가 기록되었다. 수확률이 70% 이상이다
- 오답 풀이가 별도로 보존되었다 (T9용)
- 외부 CoT 15,000행이 오염 검사를 통과했다. 제외 건수와 사유가 audit에 있다
- 두 데이터셋 모두 마지막 줄 형식이 정규식 ^FINAL_ANSWER: -?(?:0|[1-9][0-9]*)$ 를 100% 만족한다
- 생성 원본 jsonl이 보존되어 있다

[산출물]
data/rft_r1/{sft.jsonl, rejected.jsonl, audit.csv, manifest.json}
data/external_cot/{sft.jsonl, contamination_audit.csv, manifest.json}
artifacts/t5_rft_r1/{generations.jsonl, metrics.json, manifest.json}
```

---

## T6 — SFT-v1 학습 및 대조군

```text
[목표]
QLoRA로 verified-CoT SFT를 수행하고, 대조군으로 "CoT 학습이 실제로 필요했다"를 수치로 증명한다.
대조군은 발표 평가(배점 50%)의 핵심 슬라이드가 되므로 생략하지 않는다.

[학습 설정 출발점]
방식: QLoRA 4-bit NF4
LoRA rank 64 / alpha 128, target modules: attention + MLP projection 전체
seq len 2048, epoch 2, LR 1e-4 cosine, warmup 0.03
loss: assistant 응답 토큰에만 적용 (프롬프트 토큰은 마스킹)
출력 형식: 마지막 줄 정확히 "FINAL_ANSWER: <integer>"

[GPU 활용 극대화]
T0의 4-bit 로드 시 잔여 VRAM 값에서 역산한다.
  - per_device_train_batch_size를 1부터 2배씩 올리며 OOM 직전 값을 찾는다. 그 값의 90%를 채택한다
  - 목표 유효 배치(예: 32~64)를 gradient_accumulation_steps로 맞춘다
    유효 배치 = per_device_batch × grad_accum
  - gradient_checkpointing=True: 메모리를 크게 아끼는 대신 속도가 20~30% 느려진다.
    끄고 배치를 키우는 쪽이 더 빠른지 실제로 두 설정을 비교해보고 정한다
  - packing=True 검토: 짧은 샘플이 많으면 시퀀스를 이어붙여 패딩 낭비를 없앤다.
    단 assistant-only loss 마스킹이 packing과 함께 정확히 동작하는지 반드시 확인한다. 안 되면 packing을 끈다
  - optim="paged_adamw_8bit" 로 옵티마이저 메모리를 줄인다
  - bf16=True, group_by_length=True (길이 비슷한 것끼리 묶어 패딩 축소)
  - dataloader_num_workers를 CPU 코어 수에 맞춰 올려 GPU가 데이터를 기다리지 않게 한다
  - 학습 중 nvidia-smi로 사용률을 확인한다. 90% 아래면 배치를 키우거나 dataloader를 늘린다
  - 학습 시작 전 max_steps=10 으로 짧게 돌려 스텝당 시간과 peak VRAM을 재고 전체 소요를 추정한다

[돌려야 할 실험 — 한 번에 변수 하나만]
1. base (학습 없음) ← T4 결과 재사용
2. answer-only SFT (T2의 answer_only 데이터)
3. 외부 CoT만 (T5의 external_cot 15,000행)
4. RFT만 (T5의 rft_r1)
5. RFT + 외부 CoT ← 본안

각 실험마다 holdout 4종 전부에서 평가한다. greedy accuracy, invalid_rate, 평균 출력 토큰을 기록한다.
평가 시 생성 설정은 T4에서 확정한 것으로 통일한다.

[2번 대조군에 대한 해석 주의 — 반드시 comparison.md에 각주로 남길 것]
2번 answer-only는 RFT pool 전체 범위(약 12,600문제)를 쓰므로 파손 문항·오답 라벨을 그대로 타겟으로 학습한다.
4·5번 RFT 계열은 c=0 문항에서 채택 풀이가 0개라 그 문항들이 자동으로 빠진다.
즉 2번은 데이터 품질 면에서 불리한 조건에서 뛴다. 이 비대칭은 5번 > 2번 결론에 유리한 방향이다.
대조군 범위는 바꾸지 않는다(설계를 지금 바꾸면 GPU 예산이 늘고 비교 기준이 흔들린다).
대신 이 비대칭을 명시하고, 가능하면 c>=1 문항으로 한정한 2번의 부분 지표도 같이 계산해 각주에 넣는다.
(추가 학습 없이 기존 holdout 결과를 문항 단위로 재집계하면 되므로 비용이 없다.)

[완료 조건]
- 5개 실험의 holdout 4종 지표표가 완성되었다
- 5번(본안)이 1번(base)보다 random holdout에서 개선되었다. 개선폭을 pp로 보고한다
- 5번이 2번(answer-only)보다 명확히 우수하다. 이것이 CoT의 가치 증명이다
- invalid_output_rate가 base 대비 크게 떨어졌다 (SFT의 부수 효과)
- 학습 로그에 스텝당 시간, peak VRAM, GPU 사용률이 기록되어 있다
- 만약 5번이 1번보다 나쁘면 어댑터를 채택하지 않고 원인을 분석한다. 나쁜 모델로 다음 단계에 가지 않는다

[산출물]
src/train_sft.py, artifacts/t6_sft_v1/{adapters/, 실험별 metrics.json, comparison.md, manifest.json}
```

---

## T6-1 — RFT 데이터 재설계와 SFT-v1 재학습

T6 본안(RFT + 외부 CoT)은 base 대비 미채택으로 끝났다. T6-1은 그 사후 분석을 반영한 재실행이다.
(누적 기록표의 `T6-2`~`T6-5`는 T6 내부 대조군 번호이고, `T6-1`은 T6에 이어지는 별도 작업이다.)

```text
[전제 — T6 사후 분석. 이 프롬프트의 근거이므로 먼저 읽는다]

T6는 실패한 게 아니라 무효였다. 아래 7가지가 확정된 원인이고 T6-1은 이것을 고친다.

1. 통계적으로 차이가 없었다.
   random holdout N=1637에서 RFT는 base 대비 +0.86pp였으나 McNemar p=0.363이다.
   base→오답 95문항 / base→정답 109문항, 즉 204문항(12.5%)이 뒤집히고 순증은 14문항이다.
   95% CI [-0.85, +2.57]pp. 이 설계의 최소 검출 가능 효과(80% power)는 약 2.4pp다.
   → 2pp 미만 개선은 이 평가로 볼 수 없다. 정확도 pp 단독으로 채택 판정을 하면 안 된다.

2. 학습 데이터에 새 정보가 없었다 (self-distillation).
   학습 행의 94.7%가 c>=4 문항에서 나왔다. RFT pool 평균 샘플 정답률 66.6% ≈ base greedy 67.99%.
   rft_r1 학습 첫 스텝 loss가 이미 0.2035였고 grad_norm은 20스텝 만에 0.52 → 0.14로 떨어졌다.
   3.5시간 학습으로 줄인 loss 총량이 0.06 nat다. 모델이 이미 쓰는 문장을 다시 가르친 것이다.

3. 학습 가중치가 난이도와 역상관이다.
   src/build_rft.py 의 cap = min(c, 4) 는 쉬운 문제일수록 trace를 많이 뽑는다.
   RFT의 이론적 이득이 있는 c=1~3 구간이 전체의 5.3%, c=13~16 구간이 71.9%였다.
   pool의 pass@16은 85.7%인데 greedy는 68%다. 이 17.7pp 갭이 실제 헤드룸인데 거기에 가중치가 없다.
   난이도 대리지표 어느 쪽으로 봐도 같은 편향이다:
     정답 6자리 이상 — harvest 54.1% / 학습비중 0.6%
     문항 513자 이상 — harvest 59.8% / 학습비중 1.7%

4. 정답 필터가 사실상 짧은 추론 필터로 작동했다.
   오답 샘플 평균 707 토큰, 정답인데 cap에 밀린 샘플 348 토큰.
   correctness로 거르는 순간 모델 출력 분포의 짧은 절반만 남는다.
   여기에 build_rft.py가 output_tokens 오름차순 상위 4개를 뽑아 한 번 더 짧게 만들었다
   (선택된 것 중앙값 721자 vs 탈락한 정답 857자).
   결과: 학습 타깃 assistant 토큰이 median 245 / p95 818 / max 1969.
   그런데 base가 hard split에서 실제로 쓰는 길이는 평균 737 / p95 2023이다.
   긴 추론 예시가 학습 데이터에 사실상 없다. hard -1.09/-3.09pp, format -5.08/-6.25pp 회귀가 여기서 나온다.

5. packing에 attention 격리가 없다 — 구현 결함이다.
   src/train_sft.py 콜레이터가 내보내는 attention_mask는 pack 전체가 1이고 position_ids가 없다.
   모델은 attn_implementation="sdpa" 로 로드된다.
   → 한 pack 안의 각 샘플이 앞의 무관한 샘플 전부를 attention으로 본다. position도 연속으로 흐른다.
   pack당 샘플 수: rft_r1 4.57 / external_cot 4.87 / rft_external 4.65 / answer_only 15.11.
   학습 샘플의 약 78%가 "앞에 무관한 문제 4개가 붙어 있고 position 1500쯤부터 시작"하는 조건으로 학습됐다.
   추론은 항상 문제 하나가 position 0부터 단독으로 들어온다.
   manifest의 assistant_only_mask_preserved=true 는 label 마스킹만 검증한 값이고,
   attention 격리는 애초에 검증 대상이 아니었다.

6. 학습과 추론의 base 정밀도가 다르다.
   학습은 load_in_4bit=true (nf4), 평가는 vLLM dtype=bfloat16 + enable_lora 다.
   어댑터가 학습 중 보상한 양자화 오차가 추론 시점에는 존재하지 않는다.

7. 하이퍼파라미터 탐색과 체크포인트 선택이 없었다.
   4개 arm 전부 LR 1e-4 / 2 epoch / cosine-to-zero 고정, validation 평가 없음,
   save_total_limit=2 로 최종 체크포인트를 무조건 채택했다.
   정보량이 거의 없는 데이터에 2 epoch를 태우면 쉬운 모드로 더 뾰족해지는 것 말고 할 수 있는 게 없다.

참고 — 파이프라인 자체는 정상이다. answer-only 대조군이 67.99% → 23.21%, 출력 491 → 8.2 토큰으로
움직였다. 가르치면 확실히 바뀐다는 뜻이고, 학습 코드·마스킹·chat template·평가 경로는 정상이다.
학습(src/train_sft.py)과 추론(src/generate.py)이 같은 chat template를 쓰는 것도 확인됐다.
문제는 코드가 아니라 데이터였다.

[목표]
RFT 학습 데이터를 난이도 쪽으로 재설계하고, 학습 구현 결함 두 가지를 고친 뒤 SFT를 다시 돌린다.
그리고 이번에는 "개선했다"를 pp가 아니라 검정 결과로 말한다.

[작업 순서 — 앞 단계 완료 조건을 못 채우면 다음으로 가지 않는다]

[0단계] 정밀도 불일치 크기 측정 (학습 0회, 약 30분)
기존 artifacts/t6_sft_v1/adapters/rft_r1 어댑터를 그대로 쓴다.
random holdout 1,637문항을 (a) HF 4-bit nf4 + 어댑터로 새로 생성하고,
(b) 이미 있는 vLLM bf16 결과와 비교한다. 새로 생성할 것은 (a)뿐이다.
  차이 1pp 미만 → 부차 요인으로 확정하고 기록만 남긴다. 이후 학습은 4-bit 유지.
  차이 1pp 이상 → 이후 모든 학습을 bf16 LoRA로 전환한다 (1-2에서 반영).
어느 쪽이든 숫자를 남긴다. 발표 자료 항목이다.

[1단계] 구현 결함 수정 (src/train_sft.py)
1-1. packing 격리
  기본값을 packing=False 로 바꾼다. group_by_length=True 가 이미 패딩 낭비를 줄인다.
  packing을 유지하려면 block-diagonal attention mask와 샘플별 position_ids 리셋을 둘 다 구현하고,
  다음 단위 테스트를 통과해야만 켠다:
    같은 샘플을 (a) 단독으로, (b) 앞에 무관한 샘플 3개를 붙인 pack 안에서 forward 했을 때
    해당 샘플 구간의 logits가 허용 오차 내에서 일치할 것.
  테스트가 없으면 켜지 않는다. 이번 실행은 packing=False 로 간다.
  packing을 끄면 스텝당 시간이 늘어난다. 실측해서 기록한다.
1-2. 정밀도 일치
  0단계 결과를 따른다. bf16 LoRA로 전환하는 경우 OOM이면 per_device_batch를 줄이고
  grad_accum으로 유효 배치 32를 유지한다. 4-bit를 유지하는 경우 그 사실과 0단계 측정치를
  metrics.json에 명시한다. "몰라서 안 맞춘 것"과 "재고 나서 유지한 것"은 다르다.
1-3. 체크포인트 선택
  validation 500문항을 RFT pool에서 c 층화로 뽑고 학습에서 완전히 제외한다.
  holdout 4종과 교집합 0임을 검증한다.
  save_steps와 save_total_limit을 조정해 epoch 0.25 / 0.5 / 0.75 / 1.0 / 1.5 / 2.0 체크포인트를 남기고,
  각 지점에서 이 500문항 greedy accuracy를 잰다.
  최종 체크포인트가 아니라 이 곡선의 최고점을 채택한다.
  곡선 전체를 artifacts에 남긴다. "2 epoch는 과학습이다"의 직접 증거가 된다.

[2단계] 어려운 구간 표적 재생성 (약 4시간)
c=1~3 (1,207문항) + c=4~7 (918문항) = 2,125문항에만 k=48 을 추가 생성한다.
설정은 T5와 동일하게 둔다 (T=0.8, top_p 0.95, max_new_tokens 2048, 동일 프롬프트, 정답 라벨 미포함).
  왜 필요한가: c=1~3 문항은 16샘플에서 정답 trace가 최대 3개밖에 안 나온다.
  재가중만으로는 이 구간의 비중을 물리적으로 올릴 수 없다. 데이터를 더 만드는 것 외에 방법이 없다.
  2,125 × 48 = 102,000건이고 T5 실측 7.13 gen/s 기준 약 4.0시간이다. T5 전체(202,176건)의 절반이다.
기존 T5 생성 결과는 지우지 않는다. 새 생성은 별도 파일에 쓰고 병합 시점에만 합친다.
c 는 원본 16샘플 기준값을 유지한다. T7이 이 정의를 그대로 받아 쓰므로 바꾸면 안 된다.
재생성분은 trace 공급용이지 c 재정의용이 아니다.

[3단계] 데이터 재구축 (data/rft_r1_v2/)
3-1. 채택 개수를 난이도에 비례시킨다. min(c, 4) 를 버린다.
     c = 0      → 0개 (T7 대상, 변경 없음)
     c = 1      → 최대 4개
     c = 2~3    → 최대 6개
     c = 4~7    → 최대 4개
     c = 8~12   → 2개
     c = 13~16  → 1개, 그리고 문항 자체를 2,500개로 무작위 서브샘플 (seed 42)
     c=13~16은 7,309문항이라 1개씩만 넣어도 전체를 지배한다. 이미 푸는 문제이므로
     형식·문체 앵커 용도로만 남기고 문항 수를 제한한다.
     예상: 약 15,100행, c=1~3 비중 5.3% → 40.8%, c=13~16 비중 71.9% → 16.5%
     실제 수치는 2단계 수확량에 따라 달라진다. 확정치를 manifest에 남긴다.
3-2. 길이 편향 제거
     output_tokens 오름차순 상위 N개 선택을 버린다.
     문항별 정답 trace를 길이순으로 정렬한 뒤 균등 간격으로 N개를 뽑는다 (층화 추출).
     선택 전에 정규화된 풀이 서명(등장 숫자 시퀀스)으로 중복을 제거한다.
     완료 기준: 학습 타깃 assistant 토큰 p95가 1,500 이상이어야 한다 (현행 818).
     이 값이 안 나오면 층화가 동작하지 않은 것이다. 넘어가지 말고 고친다.
3-3. 나머지 계약은 T5와 동일하게 유지한다.
     마지막 줄 정확히 "FINAL_ANSWER: <integer>", 이미지 의존 문항 제외,
     오답 풀이 별도 보존 (T9 GenSelect용).

[4단계] 하이퍼파라미터 스윕 (짧게, 3회)
LR {1e-5, 3e-5, 1e-4} × 1 epoch 로 3회 학습하고 1-3의 validation 500문항으로만 비교한다.
holdout은 이 단계에서 쳐다보지 않는다. holdout으로 HP를 고르면 채택 판정이 오염된다.
T6의 1e-4 / 2 epoch는 loss 0.20 → 0.146 곡선으로 볼 때 과학습 쪽이다.
최고 validation 점수의 (LR, 체크포인트)를 5단계 설정으로 확정한다.

[5단계] 본 실험 — 한 번에 변수 하나만
  A. RFT-v2 데이터 + 1단계 수정 + 4단계 HP          ← 데이터·구현 합산 효과
  B. A + 외부 CoT 15,000행                          ← 본안 후보
외부 CoT를 다시 섞기 전에 T6의 종료 실패부터 확인한다.
external_cot는 hit_max_new_tokens가 random 2.81% → 8.00%, format 5.08% → 22.66% 로 뛰었다.
답을 쓰기 전에 잘려서 틀린 것이므로, 재사용한다면 max_solution_words 상한(현행 700)을 낮추고
종료 토큰이 실제로 학습되는지 먼저 확인한다. 확인이 안 되면 B를 돌리지 않고 A만 판정한다.

[6단계] 평가와 채택 판정 — 규칙을 실행 전에 고정한다
6-1. 1차 판정 통계는 holdout 4종의 합집합 3,737문항이다 (artifacts/t3_baseline/holdout_union_ids.txt).
     T6 평가도 이미 3,737문항 전부를 생성하고 있었으므로 추가 생성 비용이 0이다.
     N=1637 대신 N=3737을 쓰면 최소 검출 가능 효과가 약 2.4pp → 약 1.6pp로 내려간다.
     이 합집합에서의 base 정확도는 62.70%다.
6-2. base와 문항 단위로 짝지어 McNemar 검정을 한다.
     보고 형식: Δpp / base→오답 수 / base→정답 수 / 뒤집힌 총 문항 수 / 95% CI / p.
     정확도 숫자만 적지 않는다. T6의 +0.86pp가 p=0.363이었던 것을 반복하지 않기 위해서다.
6-3. 사전 등록 채택 규칙 — 실행 전에 고정하고 사후에 바꾸지 않는다:
     채택   Δ >= +1.5pp 이고 p < 0.05
     보류   Δ > 0 이지만 p >= 0.05 → 채택하지 않고 데이터·HP를 더 본다
     기각   Δ <= 0
     추가 게이트: hard 또는 format split에서 -2pp를 넘는 하락이 있으면 합집합이 통과해도 채택하지 않는다.
     T6에서 이 두 split이 먼저 무너졌으므로 조기 경보로 쓴다.
6-4. split 4종 개별 지표는 진단용으로 전부 기록하되 채택 근거로 쓰지 않는다.
     16개 비교 중 가장 좋은 하나를 고르는 것이 T6의 +0.86pp를 "개선"으로 읽게 만든 경로다.

[T7으로 넘기는 것 — T6-1에서 하지 않는다]
c=0 문항 1,801개는 채택 풀이가 0개라 어떤 재가중으로도 살릴 수 없다. T7 범위다.
정답을 프롬프트에 넣고 풀이를 생성한 뒤 정답 부분만 지우는 방식(hint-conditioned)이 후보로 있으나,
이는 T5의 "정답 라벨을 프롬프트에 절대 넣지 않는다" 규칙과 정면으로 충돌한다.
c=0 집합은 파손 문항·오답 라벨 의심 집합과 겹치므로 위험이 가장 큰 구간이기도 하다.
쓴다면 T7의 의심 집합 판정이 끝나고 라벨 신뢰가 확인된 문항으로만 한정해야 한다. 여기서는 하지 않는다.

[완료 조건]
- 0단계에서 4-bit / bf16 정밀도 차이가 pp로 측정되어 기록됐다
- packing=False로 학습했거나, block-diagonal + position_ids 리셋 단위 테스트를 통과했다
- validation 500문항이 학습 데이터·holdout 4종과 교집합 0임이 검증됐다
- 체크포인트별 validation 곡선이 남았고, 채택 체크포인트가 그 곡선의 최고점이다
- rft_r1_v2의 c=1~3 학습 비중이 30% 이상이다 (현행 5.3%)
- rft_r1_v2의 assistant 토큰 p95가 1,500 이상이다 (현행 818)
- 합집합 3,737 기준 McNemar 결과(Δ, 뒤집힌 문항 수, 95% CI, p)가 arm별로 보고됐다
- 사전 등록 채택 규칙에 따라 채택/보류/기각이 결정됐고, 규칙을 사후에 바꾸지 않았다
- 기각이면 T4 base를 유지하고 T7으로 간다. 나쁜 어댑터를 다음 단계로 넘기지 않는다

[산출물]
data/rft_r1_v2/{sft.jsonl, rejected.jsonl, audit.csv, manifest.json}
data/splits/rft_validation_500.csv
artifacts/t5_rft_targeted/{generations.jsonl, run-metadata.json, manifest.json}
artifacts/t6_1_sft_v1r/{adapters/, precision-probe.json, hp-sweep.json, checkpoint-curve.json,
                        실험별 metrics.json, comparison.md, manifest.json}
```

---

## T7 — c=0 재생성과 의심 집합 확정 (데이터 품질 감사)

**2026-08-22 범위 축소.** 원안의 5단계(SFT-v2 학습)를 잘라내고 1~4단계만 남겼다.
근거는 아래 「SFT-v2 학습을 잘라낸 이유」다. 남은 범위에는 중간 판정 게이트가 없으므로
T6/T6-1처럼 나누지 않고 한 세션에서 통으로 실행한다.

```text
[전제 — 이 프롬프트의 성격이 바뀐 이유. 먼저 읽는다]

T6-1은 A 보류(합집합 Δ+0.27pp, p=0.679) · B 기각(-8.30pp)으로 끝났고 채택된 어댑터가 없다.
따라서 이 라운드는 self-improvement가 아니다. T4 base로 생성하며, 그 사실을 manifest에 명시한다.
(T6의 rft_r1·rft_external, T6-1의 A·B 어댑터는 전부 미채택 산출물이므로 여기서 쓰지 않는다.)

같은 base로 다시 뽑는 것이므로 기대 수확은 pass@48 − pass@16 의 꼬리 확률뿐이다.
원안이 적고 있던 "실패분의 15~25% 추가 수확"은 개선된 SFT-v1을 전제로 한 숫자라 지금은 성립하지 않는다.
수확이 적게 나오는 것은 실패가 아니라 예상된 결과다. 다만 수확량은 아래 [목표] (2)의 값을 좌우하므로 반드시 센다.

[목표 — SFT-v2를 안 할 건데 왜 생성하는가]
이 질문에 먼저 답한다. 소비처가 두 개이고, 둘 다 SFT-v2와 무관하다.

(1) 의심 집합의 판별력. — 이것이 본 산출물이다.
    R1(16샘플) + R2(32샘플) = 48샘플 전부에서 정답이 한 번도 안 나온 문항을 확정하고 규모를 기록한다.
    이 숫자가 canonical 16,373에 남은 파손 문항·오답 라벨 비율의 하한 추정치이고,
    winning-strategy 2.4가 열어 둔 "그 비율을 모른다"에 숫자를 붙이는 유일한 값싼 수단이며,
    발표 자료(9장)에 그대로 들어간다.
    왜 0/16으로 끝내면 안 되는가: base가 greedy 68%인데 0/16은 그냥 어려운 정상 문항도 쉽게 걸린다.
    R1 pool의 pass@16이 85.7%였고 1,801은 그 잔여분이라 운 나쁜 정상 문항이 섞여 있다.
    0/48로 가면 그것이 걸러진다. 이 숫자의 가치는 크기가 아니라 "48번 시도해도 0번"이라는 신뢰도에 있다.

(2) T9 GenSelect 학습 데이터의 최난도 구간. — 2026-08-22에 새로 생긴 소비처다.
    GenSelect가 회수하는 것은 pass@k − maj@k 격차이고, 그 격차는 정답이 후보 중 소수인 문항에만 존재한다.
    c>=13 문항은 다수결이 이미 맞히므로 GenSelect가 회수할 것이 없다. 그런데 R1은 c=13~16이 71.9%다.
    R2 수확분(48샘플 중 1~2개만 정답)은 정확히 다수결이 지고 선택이 이겨야 하는 분포이고,
    R1이 가장 못 주는 구간이다. 이 데이터는 T9로 넘어간다.

수확이 적게 나오는 것 자체는 실패가 아니다. 다만 (2)의 값은 수확량에 비례하므로 수확 문항 수를 반드시 기록한다.
수확이 100문항 미만이면 (2)는 사실상 성립하지 않는다. 그 경우 T9는 R1의 c=1~3 (1,211문항)만으로 간다.
그때도 (1)은 그대로 성립하므로 이 작업 자체는 값을 한다. 어느 쪽인지 T9 시작 전에 확정해 넘긴다.

[작업]
1. T5에서 c=0 이었던 문항만 추린다. 확정 수는 1,801문항이다 (T5 metrics.json, pool 12,636 기준).
   원안의 "약 2,200문제 예상"은 추정치였다. 실측값으로 대체한다.

2. T4 base로 그 문항들에만 k=32 샘플을 생성한다.
   설정은 T5와 동일하게 둔다 (T=0.8, top_p 0.95, max_new_tokens 2048, 동일 프롬프트, 정답 라벨 미포함).
   1,801 × 32 = 57,632건, T5 실측 7.13 gen/s 기준 약 2.2시간이다.
   시작 후 짧게 처리량을 재측정해 추정과 어긋나면 기록한다.
   기존 T5 생성 결과는 지우지 않는다. 새 생성은 별도 파일에 쓴다.

3. 정답 일치분을 채택해 T5 데이터에 추가한다.
   채택 개수는 T5와 같은 c 규칙을 쓴다. R2의 c 는 32샘플 기준이므로 c==1 은 여기서 더 약한 신호다.
   c<=1 이면 1개만 채택한다.
   수확 문항 수와 행 수를 기록한다. SFT 학습에는 쓰지 않는다. 소비처는 T9다 ([목표] (2)).
   T9가 바로 쓸 수 있게 문항별로 정답 후보와 오답 후보를 함께 보존한다.
   문항당 정답 샘플 수(48 기준)를 필드로 남긴다. T9가 "정답이 소수인 후보 집합"을 이 값으로 골라낸다.

4. 의심 집합 확정 — 이 작업의 본 산출물이다.
   R1(16) + R2(32) 를 합쳐 0/48 인 문항을 "의심 집합"으로 분리한다.
   - 이 집합에 추가 샘플링 예산을 더 쓰지 않는다. 이후 어떤 라운드에서도 다시 건드리지 않는다.
     안 풀리는 문제에 compute를 더 붓는 것이 기본 반응이지만, 상당 부분은 애초에 풀 수 없는 문항이다.
   - SFT에서 "제외"할 필요는 없다. 채택 풀이가 0개라 이미 자동으로 빠져 있다.
   - 집합의 크기와 id 목록을 반드시 파일로 남긴다.
   - 무작위 20개를 샘플해 질문 원문을 직접 눈으로 보고, 파손/오답/단순 고난도로 분류해 기록한다.
     20개면 충분하다. 전수 검토하지 않는다.
   - 분류 비율을 1,801에 곱해 canonical 잔존 결함 문항 수의 점추정을 함께 적는다.
     n=20의 이항 신뢰구간이 넓다는 점을 각주로 남긴다. 하한 추정치라는 성격을 흐리지 않는다.

[하지 않는 것]
- SFT-v2 학습을 하지 않는다. 이유는 프롬프트 밖 「SFT-v2 학습을 잘라낸 이유」에 있다.
- hint-conditioned 생성(정답을 프롬프트에 넣고 풀이를 만든 뒤 정답 부분만 지우는 방식)을 하지 않는다.
  T5의 "정답 라벨을 프롬프트에 절대 넣지 않는다" 규칙과 정면으로 충돌하고,
  대상이 되는 c=0 집합이 곧 의심 집합이라 라벨을 믿을 수 없는 구간이기도 하다.
  T6-1이 "T7의 의심 집합 판정 후 라벨 신뢰가 확인된 문항으로 한정하면 후보"라고 넘겼으나,
  남은 일정에서 그 판정 후 재학습까지 갈 여유가 없다. 후보로만 기록하고 실행하지 않는다.

[완료 조건]
- c=0 대상 1,801문항 전부에 k=32 생성이 끝났고 generations.jsonl이 보존되어 있다
- R2 추가 수확 문항 수와 행 수가 기록되었다. 100문항 미만이면 그 사실을 T9로 명시해 넘겼다 ([목표] (2))
- 의심 집합의 크기와 id 목록이 파일로 남았다
- 20개 표본의 파손/오답/고난도 분류와 canonical 잔존 결함 점추정이 남았다
- manifest에 "T6-1 미채택으로 base 생성" 사실이 명시되어 있다
- 누적 기록표의 데이터 품질 감사 행 2개(R2 추가 수확, 의심 집합 크기)가 채워졌다

[산출물]
data/rft_r2/{generations.jsonl, candidates.jsonl, audit.csv, manifest.json}
  candidates.jsonl — 문항별 정답·오답 후보와 정답 샘플 수(48 기준). T9 GenSelect 학습 입력이다
data/suspect_set/{ids.txt, sample20_review.md, manifest.json}
```

### SFT-v2 학습을 잘라낸 이유 (2026-08-22)

원안 5단계는 "3번 추가분을 합쳐 SFT-v2를 학습한다"였다. 세 가지 이유로 실행하지 않는다.

1. **기대 효과가 사전 등록 게이트를 넘지 못한다.**
   SFT-v2의 학습 데이터는 rft_r1_v2 + R2 수확분 수백 행이다.
   rft_r1_v2 단독(T6-1 A)이 이미 합집합 Δ+0.27pp / p=0.679 / 95% CI [-0.87,+1.41]pp였고,
   format split -3.91pp로 추가 게이트도 실패했다.
   합집합 N=3,737의 최소 검출 가능 효과가 약 1.6pp인데, 수백 행 추가로 채택 규칙
   (Δ >= +1.5pp 이고 p < 0.05)을 넘길 근거가 없다.

2. **수확분이 가장 신뢰가 낮은 구간에서 나온다.**
   R2 수확 문항은 정의상 base가 16/16 실패했다가 48샘플째에 한 번 맞은 문항이다.
   그 "정답 일치"에는 틀린 풀이가 틀린 라벨과 우연히 맞은 경우가 섞여 있고,
   c=0 집합은 의심 집합과 겹치는 구간이라 그 비율이 다른 어느 구간보다 높다.
   4단계 판정이 끝나기 전에는 이 데이터의 품질을 알 수 없다.

3. **남은 예산의 기대값이 T8·T9 쪽이 명확히 높다.**
   2026-08-22 기준 개발 마감(08-30)까지 8일이다.
   방치된 헤드룸은 pass@32 86.3% vs greedy 68.0% = 약 18pp이고, 운영진이 다중 샘플링을
   명시적으로 허용했으므로 규칙 리스크도 없다. SFT는 T6·T6-1 두 번 모두 무효로 끝났다.

**T7 이후 풀이 패스의 채택 모델은 T4 base를 유지한다.** T8·T9는 T4 base 위에서 진행한다.

**단, R2 생성 자체를 잘라내지는 않는다.** SFT-v2를 안 하면 R2가 소비처 없는 데이터가 되는 것 아니냐는
지적이 정당하므로 여기에 답을 박아 둔다. 소비처는 두 개이고 둘 다 SFT와 무관하다 — T7 [목표] (1)(2) 참조.
요약하면 (1) 의심 집합을 0/16이 아니라 0/48로 확정해야 그 숫자가 신뢰를 얻고,
(2) R2 수확분은 "정답이 48개 중 1~2개"인 문항이라 T9 GenSelect가 가장 필요로 하는 학습 구간이다.
R1은 c=13~16이 71.9%라 이 구간을 거의 못 준다. 2.2 GPU-시간의 값은 여기서 나온다.
만약 T9까지 잘라내는 결정이 나오면 (2)가 사라지므로 그때는 k=32를 k=16으로 줄여 (1)만 남긴다.

---

## T8 — [C] adaptive self-consistency

```text
[목표]
다중 샘플링으로 정확도를 올리되, 24시간 추론 예산 안에 들어오는 k를 실측으로 정한다.
운영진이 명시적으로 자유 사용을 허용한 기법이다.

[근거 — 이전 실측]
base 기준 majority@3가 greedy보다 +3.1pp. pass@3는 71.0%로 maj@3(65.6%)보다 5.4pp 높다.
pass@k 추정: k=8에서 0.794, k=16에서 0.832, k=32에서 0.863

[작업]
1. 채택 모델로 k = 4 / 8 / 16 / 32 스윕. T=0.7~0.8, top_p=0.9~0.95.
   2026-08-22 기준 채택 모델은 T4 base다. T6·T6-1 어댑터는 전부 미채택이고 T7에서 SFT-v2를 실행하지 않는다.
   따라서 이 스윕은 어댑터 없이 base로 돌린다. 위 [근거]의 base 실측치가 그대로 출발점이 된다.
   각 k에서 majority@k, pass@k, agreement@k, tie_rate, 소요 시간을 기록한다.
   정확도 대비 시간 곡선을 그린다.

2. adaptive early stopping 구현
   - 처음 4개 샘플의 추출 답이 전부 일치하면 중단한다
   - 절약한 예산을 불일치 문항에 재배분해 그런 문항만 k=32까지 올린다
   - 규칙 준수: 중단 판단에 답의 일치도만 쓴다. 정답이나 계산 검증기를 절대 쓰지 않는다
   - 고정 k와 비교해 같은 총 생성 수에서 accuracy가 더 높은지 확인한다

3. 동점 처리 규칙 고정
   최빈 답이 동점이면 먼저 생성된 답을 택한다. 정답을 참조하지 않는다.
   이전 실측 tie_rate가 16.3%로 낮지 않으므로 이 규칙이 점수에 실제로 영향을 준다.

4. GPU 활용
   - 여러 문제 × 여러 샘플을 한 배치에 섞어 GPU를 채운다. 문제 단위로 순차 처리하면 사용률이 떨어진다
   - adaptive 방식은 배치 구성이 동적이 되므로, 진행 중인 문항 풀에서 계속 배치를 채우는 구조로 만든다
   - vLLM이면 continuous batching이 이걸 자동으로 해준다

5. 최종 k 확정
   1,000문항 기준 총 소요 시간을 계산하고, 24시간 예산에서 역산한다.
   여유를 최소 6시간 남긴다. 최종일에 재시도할 시간이 필요하다.

[완료 조건]
- k 스윕 결과표와 정확도-시간 곡선이 있다
- adaptive 방식이 같은 예산의 고정 k보다 우수함을 확인했거나, 아니라면 고정 k를 택한 근거가 있다
- 최종 k와 1,000문항 예상 소요 시간이 확정되었다
- 확정 설정이 greedy 대비 몇 pp 개선인지 기록되었다

[산출물]
artifacts/t8_self_consistency/{sweep.json, curve.md, final_config.json, manifest.json}
```

---

## T8-1 — `t6_4_rft_sft` 기반 T8 동등 조건 재검증

```text
[왜 지금 하는가 — 2026-08-23 추가]
로컬 합집합 3,737문항에서 T6-4 RFT SFT는 T4c greedy 대비 +0.40pp(p=0.517)로
사전 등록 채택 게이트를 통과하지 못했다. hard는 -1.09pp, format은 -5.08pp 하락했다.
반면 실제 리더보드에서는 t6_4_rft_sft가 T4c보다 높은 점수를 기록했다.

Public 신호만 보고 모델을 교체하거나 제출을 반복 튜닝하지 않는다. 대신 이 불일치를 해소하기 위해
풀이 모델만 T6-4로 바꾼 T8을 한 번, 아래에 고정한 동등 조건으로 재실행한다.
최종 비교 단위는 greedy끼리가 아니라 T4c + T8과 T6-4 + T8-1의 end-to-end 성능이다.

기존 T8의 채택 결과는 fixed majority@32, 합집합 69.31%다.
split별로 random 74.28%, template 73.98%, hard 39.64%, format 50.39%였고,
pass@32 84.40%, agreement@32 70.44%, tie 4.66%, 1,000문항 예상 시간 0.840h였다.
T8-1은 이 수치를 동일 생성 예산의 직접 대조군으로 사용한다.

[목표]
T8에서 채택한 self-consistency 계약을 그대로 유지한 채 풀이 모델에 T6-4 RFT LoRA만 적용한다.
어댑터가 샘플 다양성, pass@k, majority@k, agreement, tie rate를 어떻게 바꾸는지 측정하고,
동일 생성 예산에서 기존 T4c fixed majority@32를 실제로 상회하는지 판정한다.

[모델 신원 — 실행 전에 고정]
- base: Qwen/Qwen2.5-3B-Instruct
- base revision: aa8e72537993ba99e69dfaafa59ed015b17504d1
- tokenizer revision: aa8e72537993ba99e69dfaafa59ed015b17504d1
- adapter: artifacts/t6_sft_v1/adapters/rft_r1
- adapter SHA-256: c5351995b9874fa27778d564e0748b6e694a26936b1372711535bc28b7c38bd1
- vLLM bf16 LoRA 추론을 쓴다. 병합 모델이나 NF4 추론으로 조건을 바꾸지 않는다.

[T8에서 그대로 고정할 조건]
- 평가 집합: 기존 T8과 동일한 고정 holdout 합집합 3,737문항, 동일 ID와 동일 순서
- 프롬프트와 답 추출기: 기존 T8과 바이트 단위로 동일
- do_sample=true, temperature=0.8, top_p=0.95
- max_input_tokens=2048, max_new_tokens=2048
- fixed pool: k=32, seed=42
- k 스윕: 하나의 불변 k=32 생성 pool에서 paired prefix k=4/8/16/32를 평가
- adaptive: initial_k=4, max_k=32, stage1_seed=42004, stage2_seed=42032
- adaptive 중단 조건: 최초 4개의 추출 답이 전부 유효하고 동일할 때만 중단
- 다수결 동점: 최빈 동점 답 중 먼저 생성된 답
- 예산: 총 24시간, 최소 예비 시간 6시간, 채택 가능 런타임 상한 18시간
- 정답은 생성, adaptive 중단, 후보 배분, 투표에 절대 사용하지 않는다. 평가 완료 뒤 지표에만 쓴다.

[비파괴 구현]
1. 기존 파일과 산출물을 보존한다.
   - configs/t8_self_consistency.json
   - scripts/run_t8.sh
   - artifacts/t8_self_consistency/**
   위 경로를 수정하거나 덮어쓰지 않는다.

2. T8-1 전용 경로를 만든다.
   - configs/t8_1_rft_self_consistency.json
   - scripts/run_t8_1.sh
   - artifacts/t8_1_rft_self_consistency/**

3. src/generate.py의 기존 --adapter 기능을 사용한다.
   src/self_consistency.py의 현재 base-only 검증은 단순 삭제하지 않는다.
   실행별 기대 adapter path와 SHA-256을 검증하는 모델 신원 계약으로 일반화한다.
   - 기존 T8: adapter=null만 허용
   - T8-1: 위에 고정한 T6-4 adapter만 허용

4. 다음 실패 조건을 테스트한다.
   - T8-1에서 adapter 누락
   - 다른 adapter 경로 또는 SHA-256 불일치
   - 기존 T8에 adapter가 적용됨
   - T8과 T8-1의 generation/metadata가 섞임
   기존 T8 테스트와 결과 계약도 계속 통과해야 한다.

5. resume은 같은 run fingerprint의 완성 행만 재사용한다.
   base revision, adapter identity, config, 입력 ID 중 하나라도 다르면 기존 캐시를 재사용하지 않는다.

[실행]
1. 전용 config와 script를 만든 뒤 관련 단위 테스트를 먼저 통과시킨다.

2. T6-4 adapter를 적용한 불변 k=32 생성 pool을 만든다.
   이 pool의 prefix로 k=4/8/16/32를 평가해 다음을 기록한다.
   - majority@k, pass@k, agreement@k
   - tie rate, invalid rate, hit-max rate
   - 평균/중앙/p95 출력 토큰
   - 처리량, GPU 활용률, peak VRAM, OOM, wall time

3. 기존 T8과 동일한 adaptive 4→32 경로를 별도로 실행한다.
   adaptive 결과는 같은 실제 총 생성 수의 T6-4 fixed 대조군과 비교한다.
   fixed k=32보다 정확도가 낮으면 기존 T8과 마찬가지로 adaptive를 채택하지 않는다.

4. T6-4 자기 greedy 대비 개선 폭을 계산한다.
   greedy 기준 파일은 다음 두 개다.
   - artifacts/t6_sft_v1/rft_r1/evaluation/generations.jsonl
   - artifacts/t6_sft_v1/rft_r1/evaluation/run-metadata.json

5. 최종 주 비교는 다음 두 정책의 문항별 paired 비교다.
   - reference: artifacts/t8_self_consistency의 T4c fixed majority@32
   - candidate: artifacts/t8_1_rft_self_consistency의 T6-4 fixed majority@32
   합집합 3,737문항에서 Δpp, reference→오답 수, reference→정답 수, discordant 수,
   paired McNemar exact p-value와 95% CI를 함께 기록한다.

6. random/template/hard/format 4개 split의 지표를 모두 기록한다.
   split별 최고값을 골라 채택 근거로 쓰지 않는다. hard와 format은 퇴행 guardrail로만 쓴다.

7. 1,000문항 예상 시간을 계산하고 24시간 중 남는 시간을 기록한다.
   기존 T8과 T8-1의 정확도-시간 곡선을 한 표에서 직접 비교한다.

[사전 등록 판정 — 실행 후 바꾸지 않는다]
- 채택:
  기존 T4c T8 fixed majority@32 대비 합집합 Δ >= +1.5pp이고 p < 0.05이며,
  hard 또는 format 어느 쪽도 기존 T8 대비 2pp를 초과해 하락하지 않는다.
- 보류:
  합집합 Δ > 0이지만 효과 크기 또는 유의성 기준을 못 넘는다.
  T4c T8을 교체하지 않고 T8-1을 후보 산출물로만 보존한다.
- 기각:
  합집합 Δ <= 0이거나 hard/format 하락 guardrail을 위반한다. T4c T8을 유지한다.
- adaptive 채택:
  같은 T6-4 모델의 fixed k=32보다 정확도가 높고 18시간 안에 들 때만 채택한다.
- 리더보드 점수는 위 판정 규칙을 사후 변경하는 데 사용하지 않는다.

[범위 밖]
- T5/R1 또는 T7 데이터를 다시 생성하지 않는다.
- T9 selector를 재학습하거나 T6-4 후보에 적용하지 않는다.
- 리더보드 추론, submission.csv 생성 또는 제출을 하지 않는다.
- T8-1이 채택되면 T9-1의 필요성을 별도 handoff로 기록하고 여기서는 실행하지 않는다.

[완료 조건]
- 기존 T8 파일과 산출물 해시가 보존되어 있다.
- 모든 T8-1 generation metadata에 base revision, adapter path, adapter SHA-256이 기록되어 있다.
- 동일 조건의 k 스윕과 adaptive 대조가 완료됐다.
- T6-4 greedy 대비 및 기존 T4c T8 대비 paired 비교가 모두 완료됐다.
- 합집합 통계와 4개 split guardrail, 정확도-시간 곡선, 최종 채택/보류/기각 판정이 남았다.
- 테스트가 통과하고 ground-truth-free 생성·중단·투표 계약이 manifest로 검증됐다.
- 이 문서의 발표 자료용 누적 기록표에 T8-1 행과 판정 근거가 추가됐다.

[산출물]
configs/t8_1_rft_self_consistency.json
scripts/run_t8_1.sh
artifacts/t8_1_rft_self_consistency/
  generations.jsonl
  run-metadata.json
  sweep.json
  curve.md
  comparison.json
  comparison.md
  final_config.json
  manifest.json
  tests.xml
  adaptive/
```

---

## T8-2 — 답 불일치 기반 명시적 CoT 프롬프트 라우팅

```text
[왜 지금 하는가 — 2026-08-24 추가]
현재 최종 채택안은 T4 base의 T8 fixed majority@32이며, 합집합 3,737문항 정확도는 69.31%다.
T6·T6-1의 CoT/RFT 어댑터는 채택 게이트를 통과하지 못했으므로 이 실험에서도 풀이 어댑터를 쓰지 않는다.

초기 Phase 1의 B1은 기본 프롬프트에 단계별 풀이와 독립 검산을 추가한 정식 prompt ablation이었다.
greedy·max_new_tokens=1024 조건에서 B0→B1은 random 62.47%→61.25%(-1.22pp),
template 63.63%→63.39%(-0.24pp)였지만 hard는 24.55%→27.44%(+2.89pp)였다.
동시에 random 중앙 출력 토큰이 352→506.5, invalid가 14.73%→17.48%로 늘었다.
이후 T4에서 max_new_tokens=2048과 fallback 추출기로 길이·종료 실패를 크게 수리했으므로,
그 1024-token greedy 결과만으로 현재 T8의 sampled majority@32에서도 명시적 CoT가 불리하다고 결론내릴 수 없다.

기존 T8 k=32 pool의 사전 진단에서는 첫 4개가 모두 유효하고 같은 문항이 합집합의 약 49.6%였다.
hard split에서는 118/550(21.45%)만 첫 4개가 일치했고 432/550(78.55%)가 불일치했다.
hard 불일치 문항의 majority 정확도는 k=4 19.9%에서 k=32 28.0%로 올랐지만,
첫 4개 일치 문항은 k=4와 k=32가 모두 82.2%였다.
따라서 첫 4개 답 불일치는 정답을 보지 않고도 추가 추론이 필요한 문항을 찾는 실전 대리지표다.
단, 이 수치는 명시적 CoT의 효과를 증명하지 않는다. 아래 paired 실험으로 그 효과를 새로 검증한다.

[규정 확인]
운영진은 동일한 Qwen2.5-3B-Instruct 베이스 위의 복수 LoRA 앙상블과,
동일 베이스의 verifier 어댑터를 이용한 Best-of-N 선별을 합리적인 범위에서 허용한다고 답변했다.
이 실험은 그보다 보수적으로 어댑터 0개, 동일 base/revision 1개, 사전 고정 프롬프트 2개만 사용한다.
운영진 답변의 질문·답변 원문, 날짜, 스크린샷을 실행 전 원격 밖에도 보관한다.

규정 계약은 다음과 같다.
- 첫 4개 출력에서 src.extract로 읽은 답이 모두 유효하고 문자열 정규화 후 같은지만 본다.
- 정답 라벨, 문제 유형 라벨, 리더보드 점수, 계산 결과는 생성·라우팅·투표에 사용하지 않는다.
- 외부 모델/API/인터넷/retrieval, Python·SymPy·solver·계산기, 코드 실행, 계산 verifier를 사용하지 않는다.
- strong-CoT의 "검산"은 동일 모델이 자연어 출력 안에서 수행하는 self-check이며 도구 호출이 아니다.
- 최종 답은 32개 모델 출력의 사전 고정 동일가중치 majority vote로만 결정한다.

[목표]
쉬운 문항에서는 기존 최소 프롬프트의 짧고 안정적인 출력을 유지하고,
첫 4개 답이 갈리는 문항에서만 남은 28회를 단계별 풀이·독립 검산 프롬프트로 생성한다.
총 생성 예산을 문제당 정확히 32회로 고정한 채 기존 T8 fixed majority@32를 상회하는지 판정한다.

[모델·생성 계약 — 실행 전에 고정]
- base: Qwen/Qwen2.5-3B-Instruct
- base revision: aa8e72537993ba99e69dfaafa59ed015b17504d1
- tokenizer revision: aa8e72537993ba99e69dfaafa59ed015b17504d1
- adapter: null
- do_sample=true, temperature=0.8, top_p=0.95
- max_input_tokens=2048, max_new_tokens=2048
- 총 생성 수: 문항당 32
- seed: 42. sample_index 0..31을 보존한다
- 답 추출기와 정규화: 기존 T8의 src.extract와 바이트 단위로 동일
- 다수결 동점: 최빈 동점 답 중 먼저 생성된 답
- 평가 집합: 기존 T8과 동일한 고정 holdout 합집합 3,737문항, 동일 ID와 동일 순서
- 예산: 총 24시간, 최소 예비 시간 6시간, 채택 가능 런타임 상한 18시간

[프롬프트 계약 — 실행 전에 바이트와 SHA-256 고정]
A. base prompt는 configs/t8_self_consistency.json의 prompt_template과 바이트 단위로 동일하다.

Solve the following problem. Write the final answer on the last line exactly as
FINAL_ANSWER: <answer>. Do not write anything after that line.

Problem:
{question}

B. strong-CoT prompt는 아래 문구 하나만 사용한다. holdout 결과를 본 뒤 문구를 고치지 않는다.

Solve the following problem carefully.
1. Identify all relevant quantities and constraints.
2. Derive the answer step by step.
3. Independently verify the reasoning and calculation.
Write the final answer on the last line exactly as FINAL_ANSWER: <answer>.
Do not write anything after that line.

Problem:
{question}

두 프롬프트 모두 config에 그대로 저장하고 UTF-8 SHA-256을 run metadata와 manifest에 기록한다.
strong-CoT prompt에 첫 4개 답, 정답 후보, 불일치 사실, 문제 유형, 외부 예시를 넣지 않는다.
분기는 어떤 프롬프트로 새 샘플을 생성할지만 결정하며 모델에 중간 결과를 피드백하지 않는다.

[사전 고정 비교군]
A. reference — 기존 T8 base prompt fixed majority@32
   artifacts/t8_self_consistency/generations.jsonl을 그대로 재사용한다. 재생성하거나 덮어쓰지 않는다.

B. ablation — strong-CoT prompt fixed majority@32
   같은 3,737 ID에 strong-CoT prompt로 sample_index 0..31을 생성한다.
   이 arm은 프롬프트 자체의 평균 효과와 길이·종료 비용을 진단하기 위한 대조군이다.
   최종 채택의 primary candidate로 사용하지 않는다.

C. primary candidate — disagreement-routed CoT, 문제당 정확히 32회
   1) reference A의 base sample_index 0..3을 읽는다.
   2) 4개 추출 답이 모두 유효하고 동일하면 reference A의 base sample_index 4..31을 붙인다.
   3) 하나라도 invalid이거나 답이 둘 이상이면 arm B의 strong-CoT sample_index 4..31을 붙인다.
   4) 결합한 32개 답에 동일가중치 majority vote를 적용한다.

이 구성은 C를 만들기 위한 추가 생성 없이 A와 B의 불변 pool을 조합한다.
최종 테스트에서는 base 4회 뒤 분기하여 base 또는 strong-CoT 28회만 생성하므로 문제당 총 32회를 유지한다.

[비파괴 구현]
1. 다음 기존 파일과 산출물은 수정하거나 덮어쓰지 않는다.
   - configs/t8_self_consistency.json, scripts/run_t8.sh
   - configs/t8_1_rft_self_consistency.json, scripts/run_t8_1.sh
   - artifacts/t8_self_consistency/**
   - artifacts/t8_1_rft_self_consistency/**
   - artifacts/t9_genselect/**

2. T8-2 전용 경로를 만든다.
   - configs/t8_2_cot_routing.json
   - scripts/run_t8_2.sh
   - artifacts/t8_2_cot_routing/**

3. src/generate.py가 기존 T8 프롬프트를 강제로 검증하는 계약을 전역에서 느슨하게 만들지 않는다.
   T8-2 task에만 base/strong-CoT 두 개의 사전 고정 prompt hash를 허용하고,
   기존 T8·T8-1·T9의 prompt/model/adapter 검증과 테스트는 그대로 통과시킨다.

4. reference 재사용 전에 다음을 검증한다.
   - T8 generations, run metadata, final config의 SHA-256이 기존 manifest와 일치
   - 3,737개 ID마다 sample_index 0..31이 정확히 한 번씩 존재
   - model/revision/tokenizer/adapter/generation settings가 위 계약과 일치
   - first-4 valid unanimity와 hard 118/550 진단이 재현됨

5. router는 raw model text와 구문적 답 추출 결과만 입력받는다.
   ground-truth CSV를 읽을 수 없는 별도 함수/CLI로 만들고,
   evaluation 단계에서만 정답을 로드하도록 모듈 경계를 테스트한다.

6. resume은 prompt hash와 run fingerprint가 같은 완성 행만 재사용한다.
   prompt/model/revision/config/input ID 중 하나라도 다르면 캐시를 거부한다.

[실행]
1. 전용 config·script·router·테스트를 먼저 작성하고 CPU 단위 테스트를 통과시킨다.

2. 32문항 smoke test로 두 prompt hash, sample_index, 답 추출, 라우팅, 투표, resume을 검증한다.
   smoke에서는 정답 정확도로 프롬프트나 threshold를 고르지 않는다.

3. strong-CoT fixed k=32 불변 pool을 합집합 3,737문항에 한 번 생성한다.
   GPU 사용률, wall time, generations/sec, OOM, 입력 truncation을 기록한다.

4. A/B pool에서 primary candidate C를 label-free로 구성하고 다음을 먼저 기록한다.
   - first-4 valid unanimous / invalid / disagreement 문항 수
   - base 32회와 strong-CoT 28회로 간 문항 수
   - 각 문항의 route, 사용한 sample provenance, normalized answers, vote count, tie

5. 라우팅과 예측 파일을 동결하고 SHA-256을 기록한 뒤에만 정답을 로드해 평가한다.

[평가]
1. primary 비교는 C disagreement-routed CoT와 A 기존 T8 fixed majority@32다.
   합집합 3,737문항에서 Δpp, A→오답 수, A→정답 수, discordant 수,
   paired exact McNemar p-value와 paired bootstrap 95% CI를 기록한다.

2. B strong-CoT fixed majority@32도 A와 paired 비교하되 필수 ablation으로만 읽는다.
   B가 가장 높았다는 이유로 사후에 primary candidate를 B로 바꾸지 않는다.

3. random/template/hard/format 4개 split을 모두 기록한다.
   hard는 geometry/number_theory/combinatorics_probability/long_question/large_integer_answer
   5개 selection category도 별도로 보고한다. 하위 범주는 exploratory이며 채택 근거로 단독 사용하지 않는다.

4. first-4 valid unanimous와 나머지 문항을 나눠 A/B/C 정확도와 뒤집힘을 기록한다.
   이 분석으로 router가 실제로 이득이 필요한 문항을 잡았는지 확인하되 threshold는 바꾸지 않는다.

5. 각 arm에서 invalid rate, hit-max rate, 평균/중앙/p95 출력 토큰,
   FINAL_ANSWER marker 비율, tie rate, agreement@32, pass@32, 1,000문항 예상 시간을 기록한다.

6. 최종 테스트용 실제 staged 경로(base 4→base/strong-CoT 28)의 처리량을 별도로 측정한다.
   offline pool 조합의 정확도만 보고 채택하지 않고 18시간 런타임 상한을 실제 staged 측정으로 확인한다.

[사전 등록 판정 — 실행 후 바꾸지 않는다]
- 채택:
  primary C가 A 대비 합집합 Δ >= +1.5pp이고 exact McNemar p < 0.05이며,
  hard 또는 format 어느 쪽도 A 대비 2pp를 초과해 하락하지 않고,
  합집합 invalid가 1pp를 초과해 증가하지 않으며 실제 staged 추론이 18시간 안에 든다.
- 보류:
  합집합 Δ > 0이지만 효과 크기·유의성·guardrail 중 하나를 충족하지 못한다.
  T8을 교체하지 않고 T8-2 산출물과 후속 가설만 보존한다.
- 기각:
  합집합 Δ <= 0이거나 hard/format/invalid/runtime guardrail을 위반한다.
  기존 T8 fixed majority@32를 유지한다.
- B strong-CoT fixed arm과 hard split의 결과만으로 C를 채택하지 않는다.
- holdout 또는 리더보드 결과를 보고 프롬프트, first_k=4, threshold, sample 배분, 투표 규칙을 바꾸지 않는다.

[범위 밖]
- CoT/RFT/verifier/selector 어댑터를 새로 학습하거나 로드하지 않는다.
- 문제 유형 키워드, 질문 길이, 예상 정답 크기로 라우팅하지 않는다.
- 첫 4개 답을 strong-CoT prompt에 넣어 비평·재선택하게 하지 않는다.
- GenSelect, Best-of-N selector, 계산 verifier, TIR을 결합하지 않는다.
- leaderboard submission을 만들거나 제출하지 않는다.

[완료 조건]
- 기존 T8·T8-1·T9 산출물과 설정 파일의 해시가 보존되어 있다.
- 두 prompt의 정확한 UTF-8 바이트와 SHA-256, 모델 신원, generation settings가 manifest에 있다.
- A/B/C가 동일 3,737 ID와 문제당 32회 예산으로 문항별 paired 비교되었다.
- 라우팅·투표는 ground-truth-free이고 정답은 예측 동결 뒤 평가에서만 사용됐음이 테스트로 검증됐다.
- 합집합 primary 통계, 4개 split guardrail, hard 5범주 진단, 길이·종료·runtime 지표가 모두 남았다.
- 사전 등록 규칙에 따라 채택/보류/기각이 결정되고 최종 전략이 final_config와 manifest에 기록됐다.
- 실행 뒤 이 문서의 발표 자료용 누적 기록표에 T8-2 행과 판정 근거를 추가했다.
- 채택이면 T13 runbook과 제출 리허설을 T8-2 staged 경로로 갱신하고, 아니면 T8 경로를 유지한다.

[산출물]
configs/t8_2_cot_routing.json
scripts/run_t8_2.sh
src/cot_routing.py
artifacts/t8_2_cot_routing/
  strong_cot/generations.jsonl
  strong_cot/run-metadata.json
  routes.jsonl
  predictions.jsonl
  comparison.json
  comparison.md
  runtime.json
  final_config.json
  manifest.json
  tests.xml
```

---

## T8-3 — 추출 경로 기반 투표 품질 필터

```text
[왜 지금 하는가 — 2026-08-24 추가]
현재 최종 채택안은 T4 base의 T8 fixed majority@32이고, 합집합 3,737문항 정확도는 69.31%(2,590문항)다.
T8은 32개 출력에서 src.extract가 읽어낸 답을 전부 동일가중치 1표로 센다.
그런데 그 표의 신뢰도는 추출 경로에 따라 균일하지 않다. T8 pool 119,584건을 재집계한 결과는 다음과 같다.

  경로/조건                     비중       그 표 하나의 정답률
  final_answer_marker          82.06%     69.1%
  boxed                         6.57%     37.4%
  last_integer                 10.46%     14.4%
  standalone_last_line          0.14%     12.4%
  none(무효)                    0.77%      0.0%
  hit_max_new_tokens=True       1.22%      1.9%
  본문 내 상충 답 2개 이상       0.71%      0.0%

last_integer는 모델이 FINAL_ANSWER도 oxed{}도 쓰지 않은 출력에서 본문 마지막 정수를 주워온
폴백 경로다. T4에서 출력 계약을 수리할 때 이 폴백은 invalid 회수 목적으로 도입했고,
그 판단은 greedy k=1에서 옳았다. 그러나 k=32 투표에서까지 정답률 14.4%인 표를
69.1%인 표와 동일가중치로 세도 되는지는 T4에서 검토하지 않았다. 이 작업이 그 누락을 메운다.

[한계 정답률과 문제 내 판별력을 혼동하지 않는다]
위 표는 한계(marginal) 정답률이며, 그것만으로 가중치를 정하면 틀린 방향으로 간다.
투표를 바꾸려면 같은 문항의 32개 후보 안에서 정답을 오답보다 높게 매겨야 하므로,
문제 내 pairwise concordance(AUC, 265,583쌍)를 따로 측정했다.

  강한 추출 경로 우선            AUC 0.5965  (동점 68.4%, 비동점 구간 실질 0.81)
  짧은 output_tokens 우선        AUC 0.5587  (동점 0.4%)
  절단 안 된 것 우선             AUC 0.5088  (동점 97.9%)
  상충 답 적은 것 우선           AUC 0.4047  (역방향)

boxed가 이 구분의 실례다. 한계 정답률은 37.4%로 전체 평균 60.7%보다 훨씬 낮지만,
가중치를 낮추면 성능이 떨어진다. boxed가 나오는 문항 자체가 어려운 것이지
같은 문항 안에서 boxed 표가 나쁜 것이 아니다. 따라서 boxed 가중치는 1.0으로 유지한다.

[정직성 계약 — 이 섹션의 어느 부분이 사전 등록인지 명시한다]
이 섹션은 통상적인 사전 등록이 아니다. 필터 정책은 홀드아웃 라벨을 본 뒤에 고른 사후(post-hoc) 발견이다.
문서 나머지 섹션과 같은 형식으로 위장하지 않기 위해 범위를 다음과 같이 쪼갠다.

- 사전 등록 아님(사후 발견): 합집합 3,737문항에서 측정한 Δ +1.47pp.
  라벨을 보고 정책을 선택했으므로 이 수치는 채택 근거로 쓰지 않는다. 기록과 진단용이다.
  과적합 여부만 template_group 기준 5-fold 교차검증으로 별도 확인했다(아래).
- 사전 등록임(진짜): 홀드아웃에서 동결한 정책을 리더보드 831행에 그대로 적용하는 것.
  리더보드 라벨은 본 적이 없고 정책은 이 문서 작성 시점에 이미 고정됐다.
  따라서 리더보드 적용은 정당한 사전 등록 대상이다.
- 어떤 경우에도 홀드아웃 또는 리더보드 결과를 보고 필터 조건이나 판정 임계값을 바꾸지 않는다.

[사후 측정 결과 — 진단용, 채택 근거 아님]
정책: last_integer 경로 · standalone_last_line 경로 · hit_max_new_tokens=True ·
본문 내 상충 답 2개 이상인 생성물의 표를 버리고 남은 표로 동일가중치 majority.
전부 버려지면 무필터 majority로 폴백한다(상수 fallback "0"으로 떨어뜨리지 않는다).

  합집합         69.31%(2,590) -> 70.78%(2,645)   Δ +1.47pp
  McNemar        회수 69 / 파손 14, exact two-sided p = 6.8e-10, 회수:파손 4.9:1
  5-fold CV      5개 폴드 전부 같은 정책 선택, 5개 폴드 전부 양수(+0.80 ~ +2.52pp)
                 out-of-fold 정확도 = in-sample과 동일 -> 과적합 신호 없음
  격자 확대       boxed 0/0.25/0.5, standalone 0, 길이 페널티 0.5/0.75를 추가하면
                 CV가 +1.45pp로 소폭 하락. 효과는 last_integer 단일 레버에서 나온다.
  split          random +1.28pp, template +1.53pp, hard +1.09pp, format +2.73pp (회귀 0건)

비교 대상으로, 지금까지 T8 이후 후보들의 McNemar p는 T8-1이 0.2276, T8-2가 1.0이었고
T9는 회수 60 / 파손 191로 방향이 반대였다. 이 후보는 그것들과 성질이 다르다.

[리더보드 풀에서는 이득이 더 작을 수 있다]
필터가 당길 표의 비중이 리더보드 쪽에서 더 낮다.

  last_integer 비중   홀드아웃 10.46%  vs  리더보드 831행 7.49%
  답이 바뀌는 문항     홀드아웃 83/3,737(2.2%)  vs  리더보드 34/831(4.1%)

홀드아웃의 4.9:1 회수비가 그대로 유지되면 34건 중 순 +23문항(+2.7pp)이지만,
831문항 표본에서 나온 수치이므로 구간 추정이 넓다. 기대값은 양수, 점추정 +1 ~ +2.7pp로 본다.

[목표]
T8이 이미 생성해 둔 32개 출력을 재생성 없이 다시 집계해,
정답 라벨을 쓰지 않고 관측 가능한 신호만으로 저품질 표를 투표에서 제외한다.
새 GPU 생성 0건, 새 학습 0건.

[규정 확인]
이 작업은 모델 출력의 후처리이며 운영진이 자유 사용으로 명시한
"다중 샘플링 기반 test-time 기법" 범주 안에 있다. 계약은 다음과 같다.
- 필터 입력은 src.extract의 path/explicit_candidates와 생성 메타데이터의 hit_max_new_tokens뿐이다.
- 정답 라벨, 문제 유형 라벨, 리더보드 점수, 계산 결과는 필터·투표에 사용하지 않는다.
- 외부 모델/API/인터넷/retrieval, Python·SymPy·solver·계산기, 코드 실행을 사용하지 않는다.
- 어댑터 0개. 동일 base/revision 1개(Qwen2.5-3B-Instruct @ aa8e7253).
- 최종 답은 필터 통과 표의 사전 고정 동일가중치 majority vote로만 결정한다.

[모델·생성 계약 — 실행 전에 고정]
- 생성물을 새로 만들지 않는다. 기존 두 pool의 SHA-256을 실행 전후로 검증한다.
    artifacts/t8_self_consistency/generations.jsonl        (홀드아웃 3,737문항 x 32)
    artifacts/submissions/t8_majority_k32/generations.jsonl (리더보드 1,000문항 x 32)
- 동점 처리는 T8과 동일하다. 최초 생성 순서가 앞선 답을 고른다.
- k=32, 문항별 표본 인덱스 0..31 완전성을 기존 로더 검증으로 유지한다.

[필터 계약 — 실행 전에 바이트로 고정]
제외 조건 (OR):
  1. extraction.path in {"last_integer", "standalone_last_line"}
  2. generation["hit_max_new_tokens"] is true
  3. len(set(extraction.explicit_candidates)) >= 2
유지: final_answer_marker 가중치 1.0, boxed 가중치 1.0.
폴백: 어떤 문항에서 1~3으로 모든 표가 제거되면 그 문항만 무필터 majority로 되돌린다.
       상수 fallback으로 떨어뜨리지 않으며, 해당 문항 id를 audit에 남긴다.

[비파괴 구현]
- src/submit.py에 --filter-low-quality-votes 플래그를 추가한다. 기본값은 off다.
- 플래그 off일 때 기존 submission.csv를 바이트 동일하게 재현하는 회귀 테스트를 먼저 통과시킨다.
  (2026-08-24 예비 확인: 831행 0건 불일치. 정식 실행에서 tests.xml로 재검증한다.)
- 기존 artifacts/submissions/t8_majority_k32/ 를 덮어쓰지 않는다. 새 디렉터리에 출력한다.
- 기존 T8·T8-1·T8-2·T9 산출물과 설정 파일의 해시를 보존한다.

[실행]
1. 회귀 테스트: 필터 off로 t8_majority_k32 재집계 -> 기존 submission.csv와 0건 불일치 확인.
2. 홀드아웃 재집계: t8_self_consistency pool에 필터 적용 -> 합집합·4 split 지표를 src/evaluate.py로 산출.
3. 교차검증 재현: template_group 기준 sha256('t8-vote-cv-v1:'+group) mod 5로 5-fold,
   폴드별 선택 정책과 out-of-fold 정확도를 기록한다.
4. 리더보드 재집계: 필터 on으로 submission.csv를 새 디렉터리에 생성. 34건 변경 예상.
5. 두 제출본(무필터/필터)의 diff 목록과 문항별 표 구성을 audit에 남긴다.

[평가]
- primary: 합집합 3,737문항 짝지은 exact McNemar (기존 T8 fixed majority@32 대비).
- guardrail: random / template / hard / format 4 split 정확도와 invalid rate.
- 진단: 제거된 표 수를 조건별로 분해, 폴백 발동 문항 수, 득표 지지 밴드별 변화.
- 리더보드 제출본은 라벨이 없으므로 정확도를 계산하지 않는다. 변경 건수와 감사 기록만 남긴다.

[사전 등록 판정 — 실행 후 바꾸지 않는다]
기존 게이트를 그대로 적용한다. 사후에 임계값을 낮추지 않는다.
- 채택:
  합집합 Δ >= +1.5pp이고 exact McNemar p < 0.05이며,
  hard 또는 format 어느 쪽도 기존 T8 대비 2pp를 초과해 하락하지 않고,
  합집합 invalid가 1pp를 초과해 증가하지 않는다.
- 보류:
  합집합 Δ > 0이지만 효과 크기 또는 유의성 기준을 못 넘는다.
  T8을 교체하지 않고 T8-3 산출물을 후보로만 보존한다.
- 기각:
  합집합 Δ <= 0이거나 guardrail을 위반한다. 기존 T8 fixed majority@32를 유지한다.

이 규칙에 사후 측정치(Δ +1.47pp, p=6.8e-10)를 대입하면 판정은 "보류"다.
효과 크기가 +1.5pp에 0.03pp 미달하기 때문이다. 유의성·guardrail·회수비는 모두 통과한다.
따라서 기본 결정은 "T8 유지, T8-3은 후보 보존"이며, 이를 뒤집으려면
게이트 자체를 바꾸는 별도 결정이 필요하고 그 결정은 이 섹션 밖에서 근거와 함께 기록한다.
+1.5pp 임계값의 원래 취지는 "합집합 최소 검출 가능 효과 약 1.6pp 미만의 잡음을 채택하지 말라"였는데,
p=6.8e-10과 회수:파손 4.9:1은 이 후보가 그 잡음 범주가 아님을 보여준다.
이 긴장은 사실로 기록하고, 해소는 사람의 판단으로 남긴다.

[범위 밖]
- 새 생성, 새 학습, 새 어댑터, maj@64로의 k 확대.
- 필터 조건의 추가·삭제, 경로별 연속 가중치 탐색, 길이 기반 가중.
  (격자 확대가 CV에서 이득이 없음을 이미 확인했다. 재탐색은 과적합만 늘린다.)
- 홀드아웃 또는 리더보드 결과를 보고 필터·임계값·투표 규칙을 바꾸는 일.
- 제출 파일을 실제로 Kaggle에 올리는 일. T13에서 다룬다.

[완료 조건]
- 필터 off 회귀 테스트가 기존 submission.csv를 0건 불일치로 재현했음이 tests.xml에 있다.
- 기존 T8·T8-1·T8-2·T9 산출물과 설정 파일의 해시가 보존되어 있다.
- 필터 조건이 config에 바이트로 고정되고 SHA-256이 manifest에 있다.
- 합집합 primary McNemar, 4개 split guardrail, 5-fold out-of-fold 수치가 모두 남았다.
- 필터·투표가 ground-truth-free이고 정답은 예측 동결 뒤 평가에서만 사용됐음이 테스트로 검증됐다.
- 폴백 발동 문항 수와 id, 조건별 제거 표 수가 audit에 남았다.
- 사전 등록 규칙에 따라 채택/보류/기각이 결정되고 최종 전략이 final_config와 manifest에 기록됐다.
- 실행 뒤 이 문서의 발표 자료용 누적 기록표에 T8-3 행과 판정 근거를 추가했다.
- 채택이면 T13 runbook과 제출 리허설을 필터 경로로 갱신하고, 아니면 무필터 T8 경로를 유지한다.

[산출물]
configs/t8_3_vote_filter.json
scripts/run_t8_3.sh
artifacts/t8_3_vote_filter/
  holdout/predictions.jsonl
  holdout/comparison.json
  holdout/comparison.md
  cross-validation.json
  vote-filter-diagnostics.json
  final_config.json
  manifest.json
  tests.xml
artifacts/submissions/t8_3_filtered_k32/
  submission.csv
  submission-prepared.json
  submission-audit.json
  diff-vs-t8-unfiltered.json
```

### 실행 결과 — 2026-08-24

- 필터 off 통합 회귀를 포함한 집중 테스트 30개가 전부 통과했다. 831행 무필터 재집계 CSV는 기존
  T8 제출본과 행 불일치 0건·바이트 동일이었고 SHA-256도
  `e13f933abd73017a845cf683ed3df6d91f34c7769dabe7d5bc7c1cbb13ca236b`로 일치했다.
- 합집합 3,737문항은 69.31%(2,590)에서 70.78%(2,645)로 +1.47pp 상승했다. 회수 69 / 파손 14,
  95% CI `[+1.00,+1.95]pp`, exact McNemar `p=6.80e-10`이다. 예측 문자열 자체는 220건 바뀌었고,
  그중 정오 여부가 바뀐 discordant가 83건이다. split 변화는 random +1.28pp, template +1.53pp,
  hard +1.09pp, format +2.73pp로 네 split 모두 양수였다.
- template-group 5-fold CV는 5개 학습 fold가 모두 동결 필터를 선택했고 검증 fold도 전부 양수였다
  (`+0.80~+2.52pp`). OOF는 70.78%, 무필터 대비 +1.47pp로 전체 재집계와 일치했다.
- 홀드아웃 119,584생성 중 조건에 걸린 후보는 중복 제거 13,544건, 실제 유효 표 제거는 12,702건이었다.
  약한 추출 경로 12,670건, hit-max 1,452건이 유효 표 제거에 기여했고 상충 explicit 후보 842건은
  원래부터 무효 표였다. 문항별 무필터 폴백은 46건이다.
- 라벨이 없는 831행 리더보드 재집계에서는 유효 표 2,017건을 제거했고 폴백 5건, 최종 답 변경
  34건이었다. 정확도는 계산하지 않았다. 새 CSV는 831행·ID 중복/누락 0·전량 정수이며 SHA-256은
  `1a99b8eb0797fd4699a65b392cae2286c29647354cdfa56cde39dc6a71ecdb63`이다.
- T8/T8-1/T8-2/T9 보호 파일 166개의 실행 전후 해시가 전부 일치했고 두 원시 generation pool도
  불변이었다. 정답은 두 예측 맵을 동결한 뒤 평가와 CV 진단에서만 불러왔다.
- **판정: 보류.** 유의성·guardrail은 통과했지만 합집합 효과가 사전 고정 +1.5pp 게이트에
  0.028pp 미달했다. T8-3 제출본은 후보로 보존하고 최종 전략은 T8 fixed majority@32를 유지한다.

---

## T8-4 — T8-1 RFT pool에 동결 T8-3 vote-quality filter 전이

```text
[왜 지금 하는가 — 2026-08-25 추가]
T8-1은 풀이 모델을 RFT LoRA로 바꿔 합집합 69.84%를 기록했고, T8 base보다 +0.54pp였지만
+1.5pp 채택 게이트를 넘지 못해 보류됐다. T8-3은 base T8의 같은 k=32 pool에 저품질 표 필터를
적용해 70.78%(T8 대비 +1.47pp)를 기록했지만 역시 +1.5pp 게이트에 0.028pp 미달해 보류됐다.

두 레버는 서로 다른 층에 있다. T8-1은 생성 분포를 바꾸고 T8-3은 생성 뒤 투표만 바꾼다.
따라서 둘을 조합했을 때 효과가 더해지는지, 혹은 RFT의 출력 계약 준수율이 높아 필터의
작동 여지가 줄어드는지를 새 생성 없이 확인한다.

[정직성 계약]
- 조합 결과를 보기 전에 primary/secondary 비교와 +1.5pp 판정 게이트를
  analysis/t8_4_rft_vote_filter_preregistration.json에 고정했다.
- 그러나 T8-3 필터 자체는 같은 3,737문항 홀드아웃 라벨을 본 뒤 발견한 사후 정책이다.
  따라서 T8-4도 완전히 독립적인 검증이 아니라 "동결 정책의 cross-solver 전이·조합 진단"이다.
- T8-4 결과를 보고 필터 조건, k, 폴백, 동점 규칙 또는 판정 임계값을 바꾸지 않는다.
- 리더보드 831행에는 라벨을 사용하지 않으며 변경 건수와 투표 감사만 기록한다.

[목표]
T8-1이 이미 생성한 RFT k=32 pool에 T8-3의 동결 이진 필터를 그대로 적용해 다음을 측정한다.
1. 필터의 순수 전이 효과: RFT filtered vs RFT unfiltered.
2. 최종 후보 효과: RFT filtered vs 현재 최종안 T8 base unfiltered.
3. 후보 간 순위: RFT filtered vs T8-3 base filtered.
새 GPU 생성 0건, 새 학습 0건.

[모델·생성 계약]
- base: Qwen/Qwen2.5-3B-Instruct @ aa8e72537993ba99e69dfaafa59ed015b17504d1
- adapter: artifacts/t6_sft_v1/adapters/rft_r1
- adapter SHA-256: c5351995b9874fa27778d564e0748b6e694a26936b1372711535bc28b7c38bd1
- k=32, seed=42, temperature=0.8, top_p=0.95, max_new_tokens=2048.
- 기존 두 pool만 읽고 실행 전후 SHA-256을 비교한다.
    artifacts/t8_1_rft_self_consistency/generations.jsonl
    artifacts/submissions/t8_1_rft_majority_k32/generations.jsonl
- 문항별 sample_index 0..31 완전성을 검사한다.

[필터 계약 — T8-3과 바이트 동일]
제외 조건 (OR):
  1. extraction.path in {"last_integer", "standalone_last_line"}
  2. generation["hit_max_new_tokens"] is true
  3. len(set(extraction.explicit_candidates)) >= 2
유지: final_answer_marker 1.0, boxed 1.0.
폴백: 남은 유효표가 없으면 같은 문항의 RFT 무필터 majority로 복귀한다.
동점: 최다 득표 답 중 먼저 생성된 답.

[비교·평가]
- primary transfer: T8-1 RFT filtered vs T8-1 RFT unfiltered의 합집합 paired exact McNemar.
- adoption reference: T8-1 RFT filtered vs 현재 최종 T8 base unfiltered.
- candidate ranking: T8-1 RFT filtered vs T8-3 base filtered.
- guardrail: random/template/hard/format 정확도, 합집합 invalid 증가.
- 진단: 조건별 제거 표, 최종 답 변경, 전표 제거 폴백, 추출 경로 분포.
- template_group_id 기준 T8-3과 동일한 5-fold를 재사용해 폴드별 전이 방향을 기록한다.

[실행]
1. 기존 submit/vote-filter 집중 테스트를 실행한다.
2. 라벨을 읽기 전에 RFT unfiltered/filtered와 T8 base unfiltered/filtered 예측 맵을 동결한다.
3. 동결 뒤 canonical 라벨을 읽어 합집합·4 split·McNemar·5-fold 진단을 계산한다.
4. 831행 리더보드 입력을 RFT pool에서 무필터/필터로 재집계한다.
5. 무필터 경로가 기존 T8-1 제출과 0건 불일치인지 확인하고 필터 변경 ID를 남긴다.
6. 두 원본 generation pool의 실행 전후 SHA-256이 같은지 검증한다.

[판정 게이트 — 현재 최종 T8 대비]
- 채택 후보:
  합집합 Δ >= +1.5pp이고 exact McNemar p < 0.05이며,
  hard 또는 format 어느 쪽도 T8 대비 2pp를 초과해 하락하지 않고,
  합집합 invalid가 T8 대비 1pp를 초과해 증가하지 않는다.
- 보류:
  합집합 Δ > 0이지만 효과 크기·유의성·guardrail 중 하나를 못 넘는다.
  T8-4를 후보로만 보존하고 현재 최종 T8을 유지한다.
- 기각:
  합집합 Δ <= 0이거나 guardrail을 위반한다.
- RFT unfiltered 대비 증분은 필터 전이 진단이며, 최종 채택 게이트의 reference를 대신하지 않는다.

[범위 밖]
- 새 생성, 새 학습, 다른 RFT adapter, k 확대.
- T8-3 필터 조건이나 경로별 가중치 변경.
- T10c 연속 가중 투표와의 중첩.
- 리더보드 결과를 보고 정책 또는 판정 임계값 변경.
- 후보 CSV의 실제 제출.

[산출물]
configs/t8_4_rft_vote_filter.json
scripts/run_t8_4.sh
analysis/t8_4_rft_vote_filter.py
analysis/t8_4_rft_vote_filter_preregistration.json
artifacts/t8_4_rft_vote_filter/
  experiment.json
  tests.xml
```

### 실행 결과 — 2026-08-25

- 집중 테스트 14개가 전부 통과했고, 두 T8-1 generation pool의 실행 전후 SHA-256이 일치했다.
  831행 필터-off 재집계도 기존 T8-1 제출과 불일치 0건이었다.
- RFT pool 자체에서 필터는 합집합 69.84%(2,610)에서 70.30%(2,627)로 +0.455pp 올랐다.
  회수 19 / 파손 2, 95% CI `[+0.215,+0.695]pp`, exact McNemar `p=2.21e-4`다.
  최종 답 문자열은 90건 바뀌었다.
- 현재 최종 T8 base unfiltered와 비교하면 +0.990pp(69.31%→70.30%), 회수 147 / 파손 110,
  95% CI `[+0.150,+1.830]pp`, `p=0.0245`다. 유의성은 통과했지만 +1.5pp 효과 게이트는 실패했다.
- T8-3 base filtered 70.78%와 비교하면 T8-4는 -0.482pp이며 `p=0.262`로 유의한 차이는 아니다.
- split은 RFT 무필터 대비 random +0.31pp, template +0.49pp, hard +0.91pp,
  format +1.17pp로 모두 양수였다. T8 대비 최종 정확도는 각각
  75.38% / 74.71% / 40.36% / 54.69%다.
- template-group 5-fold의 학습 fold는 모두 필터를 선택했다. 검증 delta는
  +0.41/+0.27/+0.66/+0.00/+0.92pp로 음수 폴드는 없었고, OOF는 전체와 같은 +0.455pp다.
- 홀드아웃 119,584생성에서 조건 후보는 중복 제거 5,415건, 실제 유효표 제거는 4,589건,
  폴백은 4문항이었다. final_answer_marker가 113,613건(95.01%)으로 base T8의 82.06%보다 높아
  필터가 제거할 약한 경로가 적었고, 이것이 T8-3보다 작은 증분의 주된 설명이다.
- 라벨 없는 831행 리더보드 재집계에서는 유효표 786개를 제거했고 폴백 0건, 최종 답 변경 17건이었다.
  정확도는 계산하지 않았다.
- **판정: 보류.** 유의성·hard/format·invalid guardrail은 통과했지만 현재 최종 T8 대비
  +0.990pp로 +1.5pp 효과 게이트를 넘지 못했다. T8-4 산출물은 후보로 보존하고
  최종 전략은 T8 fixed majority@32를 유지한다.

---

## T9 — [D] GenSelect

```text
[목표]
majority voting이 놓치는 점수를 회수한다. pass@k와 majority@k 사이 격차가 이 레버의 상한이다.
AIMO-2 보고서(arXiv 2504.16891)에서 제안했으나, 저자들이 최종 우승 파이프라인에는 포함하지 않은 기법이다
(논문 원문: "not use GenSelect training or inference for the Kaggle submission").
코드 실행 없이 모델 출력만 쓰므로 본 대회에서 사용 가능하다.

[규칙 확인 완료]
운영진: "Majority Voting, Self-Consistency 등 다중 샘플링 기반의 test-time 기법은 자유롭게 활용"
GenSelect는 이 범주에 속한다. 외부 도구 없음, 외부 모델 없음, 전부 모델 자신의 추론 출력.

[별도 어댑터에 대한 판단 — 2026-08-22 재검토]
원안은 "별도 어댑터를 로드하면 다른 모델 앙상블 해석 위험이 생기므로, 하나의 어댑터가
풀이 모드와 선택 모드를 모두 수행하도록 SFT 믹스에 함께 학습한다"였다. 이 제약을 완화한다.

규칙 문언을 다시 확인했다. rules.md의 금지 조항은 "다른 외부 모델을 호출해 앙상블하면 안 됩니다"이고,
LoRA·QLoRA 등 PEFT는 허용 학습 기법으로 명시되어 있다. 금지 대상은 다른 베이스 모델이지
허용된 base 하나에 얹은 어댑터가 아니다. 원안의 제약은 규칙 문언이 아니라 우리가 스스로 건 보수적 해석이었다.

[이번 실행의 구성]
선택 모드는 학습해서 간다. 무학습 few-shot으로 대체하지 않는다.
  - 풀이 패스: T4 base, 어댑터 없음. T8에서 확정한 k와 생성 설정을 그대로 쓴다.
    (T6·T6-1 어댑터는 전부 미채택이고 T7에서 SFT-v2를 실행하지 않으므로 풀이 채택 모델은 base다.)
  - 선택 패스: 같은 base + 선택 전용 LoRA 어댑터 1개.
  - base 가중치는 한 벌이고 revision도 동일하다. 병합하지 않고 어댑터만 붙였다 뗀다.
    manifest에 base revision 해시와 어댑터 해시를 나란히 기록해 "모델은 하나"임을 문서로 남긴다.
  - 풀이 패스에 어댑터를 얹지 않으므로 "풀이 모드 퇴행"이 구조적으로 발생하지 않는다.

왜 rft_r1_v2를 섞어 하나의 어댑터로 두 모드를 다 시키지 않는가:
  그 데이터로 학습한 T6-1 A가 합집합 Δ+0.27pp / p=0.679 이고 format split -3.91pp로 추가 게이트를 못 넘겼다.
  두 모드를 한 어댑터에 넣으면 풀이 패스가 그 어댑터로 바뀌고, 이미 기각선에 걸린 모델을 채택하는 셈이 된다.
  선택 모드만 학습하면 풀이 성능은 base 그대로다. 잃을 것이 없는 쪽을 택한다.

[작업]
1. 학습 데이터 구성 — T5/T7에서 이미 생긴 정답 풀이와 오답 풀이를 재활용한다. 추가 생성 비용이 0이다.
   입력: 문제 + 후보 풀이 요약 N개 (N=8~16, 정답 후보와 오답 후보를 섞는다)
   출력: 짧은 판단 근거 → 선택한 후보 번호 → 최종 답 (순서를 이대로 고정한다. 이유는 2-1)
   약 3,000샘플 목표. 정답 후보가 항상 특정 위치에 오지 않도록 순서를 무작위화한다.
   데이터에서 validation 분할을 따로 떼고, holdout 4종과 교집합 0을 검증한다.

1-1. 난이도 구성이 이 단계의 핵심이다. 아무 문항이나 3,000개 뽑으면 학습이 헛돈다.
   GenSelect가 회수하는 것은 pass@k − maj@k 격차이고, 그 격차는 정답이 후보 중 소수인 문항에만 있다.
   c>=13 문항은 다수결이 이미 맞히므로 그런 후보 집합을 학습해봐야 "다수 쪽을 고르라"만 배운다.
   그런데 R1 pool은 c=13~16이 71.9%다. 그대로 샘플링하면 정확히 그 헛도는 데이터가 만들어진다.
   따라서 문항당 정답 샘플 수를 기준으로 층화한다. 우선순위가 높은 순서로:
     (a) T7 R2 수확분 (data/rft_r2/candidates.jsonl) — 48샘플 중 정답 1~2개. 가장 값진 구간이다
     (b) R1 c=1~3 (534 + 677 = 1,211문항) — 16샘플 중 정답 1~3개
     (c) R1 c=4~7 (918문항) — 보조
     (d) c>=8 은 형식 앵커 용도로만 소량 넣는다. 전체의 20%를 넘기지 않는다
   실제 구성 비율을 manifest에 기록한다. T6가 난이도 역상관 가중치로 무효가 됐던 것과 같은 실패다.
   T7 R2 수확이 100문항 미만으로 보고되면 (a)를 빼고 (b)(c)만으로 간다. 그 사실을 manifest에 남긴다.

2. 선택 전용 LoRA 어댑터를 학습한다 (풀이 데이터를 섞지 않는다).
   학습 설정은 T6-1이 확정한 것을 승계한다: packing=False, bf16 LoRA(정밀도 프로브 1.65pp 근거),
   체크포인트는 validation 곡선의 최고점을 채택, holdout은 HP 선택에 쓰지 않는다.
   3,000샘플이면 1 epoch가 짧다. LR {1e-5, 3e-5, 1e-4} 스윕을 validation으로 돌린다.

2-1. 출력 붕괴 방지 — T6-2(answer-only)의 실패 모드가 여기서 재현될 수 있다.
   타깃이 "후보 번호"만이면 출력이 몇 토큰으로 붕괴하고(T6-2: 491 → 8.2 토큰),
   어댑터가 후보 내용을 읽지 않고 형식만 외운다. 그래서 1번의 타깃에 판단 근거를 먼저 쓰게 했다.
   근거는 후보들이 실제로 갈리는 지점을 인용하게 한다.
   학습 후 선택 출력의 평균 토큰 수를 기록한다. 20토큰 아래면 붕괴를 의심하고 타깃 형식부터 고친다.

2-2. 위치 암기 검사 — 학습 시 순서 무작위화만으로는 부족하다.
   평가에서 같은 문항의 후보 순서만 셔플해 두 번 돌리고 선택 결과의 일치율을 잰다.
   순서만 바꿨는데 정확도가 크게 흔들리면 내용이 아니라 위치를 학습한 것이다. 수치를 기록한다.

2-3. 무학습 few-shot 대조군을 함께 잰다 (추가 학습 0회).
   같은 base에 선택 프롬프트만 넣은 경로를 동일 예산에서 평가한다.
   학습이 값을 했는지 말하려면 이 비교가 필요하고, 비용은 생성 1회뿐이다.
   발표 자료의 "선택 모드는 학습이 필요했다" 슬라이드가 이 대조에서 나온다.

3. 추론 파이프라인
   후보 16개 생성 → GenSelect 4~8회 반복(매번 후보 부분집합을 다르게) → 선택된 답들에 다수결
   참고: AIMO-2는 64후보에서 16개씩 뽑아 64회 반복 후 다수결을 적용했다. 예산에 맞춰 축소한 형태다.

4. GPU 활용
   GenSelect 입력은 후보 여러 개를 담아 프롬프트가 매우 길다. 입력 토큰 예산을 다시 잡아야 한다.
   - 후보 풀이를 통째로 넣지 말고 요약(핵심 단계 + 최종 답)만 넣어 길이를 줄인다
   - max_model_len을 늘려야 할 수 있다. KV 캐시 여유를 확인하고 배치를 다시 캘리브레이션한다
   - 선택 출력은 근거 몇 문장 + 번호 + 답이므로 max_new_tokens를 작게(예: 256) 잡아 낭비를 없앤다
   - vLLM에 enable_lora로 선택 어댑터를 붙인다. 풀이 패스와 선택 패스의 어댑터 적용 여부를
     런 단위로 분리하고, 각 런의 manifest에 어댑터 적용 여부를 명시한다

[완료 조건]
- 동일한 총 생성 예산에서 GenSelect가 majority@k를 상회한다. 상회하지 못하면 채택하지 않고 T8 설정으로 되돌린다
- 선택 어댑터가 무학습 few-shot 대조군을 동일 예산에서 상회한다. 상회하지 못하면 어댑터를 채택하지 않고
  few-shot 경로를 쓴다. 어느 쪽이든 두 수치를 모두 기록한다
- 선택 출력 평균 토큰 수와 후보 순서 셔플 일치율이 기록되었다
- 학습 데이터의 난이도 구성 비율이 manifest에 기록되었고, c>=8 구간이 20%를 넘지 않는다 (1-1)
- 풀이 패스에 어댑터가 적용되지 않았음이 런 manifest로 확인된다 (풀이 성능은 T8 확정치와 동일해야 한다)
- manifest에 base revision 해시와 선택 어댑터 해시가 나란히 기록되어 "모델은 하나"임이 문서화되었다
- 추가된 추론 시간이 24시간 예산 안에 들어온다

[산출물]
data/genselect/{train.jsonl, validation.jsonl, manifest.json}
artifacts/t9_genselect/{adapters/, hp-sweep.json, metrics.json, comparison.md, manifest.json}
```

### 실행 결과 — 2026-08-23

- 데이터는 train 3,000행 / validation 320행으로 만들었고 holdout 4종과의 교집합은 0이다. 학습 구성은
  R2 hard tail 1,200(40.00%), R1 c=1~3 1,000(33.33%), c=4~7 600(20.00%), c>=8 앵커
  200(6.67%)이다. 정답 후보 위치는 16개 위치에 train 각 187~188회, validation 각 20회로 균형화했다.
- packing=False·bf16 LoRA로 LR 1e-5 / 3e-5 / 1e-4와 각 0.25 / 0.5 / 0.75 / 1.0 epoch
  체크포인트를 validation에서 비교했다. 최고점은 LR 1e-4, step 47(0.5013 epoch)의 25.31%였고,
  선택 어댑터 SHA-256은 `c6d4aa3fc53b93e5fadfa661cba8460cadf46d62689a5d8ee1e7463004add74d`다.
- 합집합 3,737문항에서 T8 majority@32는 69.31%, adapter GenSelect(32풀이+4선택)는 55.90%,
  few-shot GenSelect(32+4)는 65.40%였다. 엄격 동일 예산 대조인 few-shot 28풀이+4선택은 65.80%로
  T8보다 -3.51pp(McNemar p=4.22e-17)였다. adapter는 few-shot보다 -9.50pp라 두 채택 게이트를 모두 실패했다.
- 어댑터 선택 출력은 평균 125.85 tokens, 유효 후보 번호율 99.42%, 후보 답과 출력 최종 답 불일치율
  0.15%로 짧은 출력 붕괴는 없었다. 512문항 순서 셔플의 선택 답 일치율은 58.98%, 정확도 변화는
  0.00pp였다. 선택 입력 truncation은 0건이고 1,000문항 32풀이+4선택 예상 시간은 1.020시간이다.
- **결론: GenSelect와 선택 어댑터를 모두 미채택하고 T8 fixed majority@32로 복귀한다.** 학습·평가
  원시 출력은 삭제하지 않았으며 세부 수치와 해시는 `artifacts/t9_genselect/`에 보존했다.

## T10a — 프롬프트 개선 (CoT 지시 + 출력 형식 안내)

```text
[왜 지금 하는가]
현재 채택안(T8 fixed majority@32)의 프롬프트는 다음 한 줄뿐이다.

  Solve the following problem. Write the final answer on the last line exactly as
  FINAL_ANSWER: <answer>. Do not write anything after that line.

  Problem:
  {question}

이 프롬프트에는 단계별 풀이(Chain-of-Thought) 지시가 없고,
\boxed{} 마커 안내가 없으며, 검산(self-check) 유도가 없다.
T8 pool 119,584건의 추출 경로를 보면 final_answer_marker가 82.06%이고
boxed는 6.57%, last_integer 10.46%, standalone_last_line 0.14%, none 0.77%다.
last_integer 경로의 정답률은 14.4%로 final_answer_marker의 69.1%에 비해 현저히 낮다.

Phase 1 T4에서 B1(단계별 풀이 + 독립 검산) 프롬프트를 greedy·1024 토큰 조건에서 측정한 결과
random은 -1.22pp였지만 hard는 +2.89pp였고, 그때의 실패는 hit-max(random 14.73%→17.48%)와
invalid(14.73%→17.48%) 증가에 있었다. 이후 T4에서 max_new_tokens=2048으로 확대하고
fallback 추출기로 invalid를 크게 수리했으므로, T8의 sampled majority@32 환경에서
개선된 프롬프트를 재평가해야 한다.

T8-2에서 strong-CoT prompt를 ablation(arm B)으로 실험했지만, 그 결과는
disagreement-routed 구성(arm C)에만 사용되었고,
strong-CoT prompt 단독 fixed majority@32(arm B)의 채택 여부는 판정하지 않았다.
T10a는 T8-2 arm B 데이터를 재활용하면서, 프롬프트 자체를 더 정밀하게 개선한다.

[목표]
(1) CoT 지시와 출력 형식 안내(\boxed{} 또는 FINAL_ANSWER)를 포함한
    개선 프롬프트를 2~3개 설계하고, 기존 base prompt 대비 fixed majority@32에서
    합집합 정확도 향상을 측정한다.
(2) 프롬프트 개선의 효과를 기존 T8-2 arm B(strong-CoT) 데이터와 비교해
    어떤 지시 요소가 이득을 주는지 분해한다.
(3) 최종 채택 프롬프트를 고정하고, T10b 프롬프트 다양성의 기반 템플릿으로 넘긴다.

[규정 확인]
프롬프트 수정은 모델 출력의 사전 조건 변경이며 학습·외부 도구와 무관하다.
운영진이 허용한 "다중 샘플링 기반 test-time 기법"의 범주 안이다.
- 프롬프트에 정답, 유사 예제, 외부 지식, 문제 유형 라벨을 넣지 않는다.
- 프롬프트에 코드 실행, 도구 호출, 계산기 사용을 지시하지 않는다.
- 최종 답은 사전 고정 동일가중치 majority vote로만 결정한다.
- 프롬프트를 holdout/리더보드 정확도를 보고 반복 수정하지 않는다.

[모델·생성 계약 — 실행 전에 고정]
- base: Qwen/Qwen2.5-3B-Instruct
- base revision: aa8e72537993ba99e69dfaafa59ed015b17504d1
- adapter: null
- do_sample=true, temperature=0.8, top_p=0.95
- max_input_tokens=2048, max_new_tokens=2048
- fixed pool: k=32, seed=42
- 평가 집합: 기존 T8과 동일한 고정 holdout 합집합 3,737문항
- 예산: 총 24시간, 최소 예비 시간 6시간

[프롬프트 후보 — 실행 전에 바이트와 SHA-256 고정]
A. base (현 채택안, 대조군)
   기존 T8의 prompt_template과 바이트 동일. 재생성하지 않고 기존 pool을 재사용한다.

B. strong-CoT (T8-2 arm B, 기존 데이터 재사용)
   기존 T8-2의 strong_cot/generations.jsonl을 재사용한다. 재생성하지 않는다.

C. cot-boxed — 새 프롬프트. 이 작업의 핵심 후보.

Think through this problem step by step, showing your reasoning clearly.
After you reach the answer, write it inside \boxed{} and also on the last line as
FINAL_ANSWER: <answer>. Do not write anything after the FINAL_ANSWER line.

Problem:
{question}

   설계 근거:
   - "step by step"은 CoT 지시의 최소 형태이다. 과도한 지시(identify all quantities,
     independently verify)는 B1에서 출력 길이를 늘리고 hard 이외에서는 역효과였다.
   - \boxed{}를 명시적으로 안내한다. 현재 boxed 6.57%를 끌어올려 추출기의 고신뢰 경로를 늘린다.
   - FINAL_ANSWER도 병기해 기존 추출 파이프라인과의 호환을 유지한다.
   - 검산(verify)은 넣지 않는다. B1에서 검산 지시가 출력 길이 증가의 주범이었고,
     hard 이외에서의 이득이 없었다. 길이 증가는 hit-max 위험을 높인다.

D. cot-brief — 새 프롬프트. 최소 CoT 지시.

Solve the following problem step by step. Show your work, then state the final answer as
FINAL_ANSWER: <answer>.

Problem:
{question}

   설계 근거:
   - B와 C의 중간 지점. "show your work"만으로 CoT를 유도하되 \boxed{}는 빼서
     지시 복잡도를 낮춘다. C와 비교해 어떤 요소가 효과를 주는지 분리한다.

프롬프트를 이 이상 만들지 않는다. 4개(A/B/C/D)로 고정한다.
holdout 결과를 보고 프롬프트를 고치거나 추가하지 않는다.

[작업]
1. 기존 데이터 확인
   - A(base): artifacts/t8_self_consistency/generations.jsonl의 SHA-256 검증. 재생성 0건.
   - B(strong-CoT): artifacts/t8_2_cot_routing/strong_cot/generations.jsonl의 SHA-256 검증. 재생성 0건.

2. C(cot-boxed)와 D(cot-brief)의 k=32 pool을 합집합 3,737문항에 생성한다.
   두 프롬프트의 UTF-8 바이트와 SHA-256을 config와 manifest에 기록한다.
   3,737 × 32 × 2 = 239,168건. T8 실측 13.24 gen/s 기준 약 5.0시간이다.
   A/B 기존 pool과 별도 디렉터리에 저장하고 기존 산출물을 덮어쓰지 않는다.

3. A/B/C/D 각각의 fixed majority@32를 동일 추출기·동점 규칙으로 평가한다.
   문항별 paired 비교는 A를 대조군으로 삼아 B/C/D 각각과 McNemar를 돌린다.
   기록 항목:
   - 합집합 정확도, Δpp vs A, exact McNemar p, 95% CI
   - random/template/hard/format 4 split 정확도
   - 추출 경로 분포 (final_answer_marker / boxed / last_integer / standalone_last_line / none)
   - hit-max rate, invalid rate, 평균/중앙/p95 출력 토큰
   - tie rate, agreement@32, pass@32

4. 추출 경로 분해 분석
   C에서 boxed 비율이 A 대비 얼마나 올랐는지, 그로 인해 last_integer 비율이 줄었는지,
   그것이 T8-3 투표 필터의 제거 대상을 줄이는 효과가 있는지 연쇄 분석한다.
   T8-3 필터를 C/D pool에도 적용해 필터 전후 정확도를 비교한다.

5. 최종 채택 프롬프트 결정
   A/B/C/D 중 합집합 정확도가 가장 높은 것을 후보로 삼되, 사전 등록 판정 규칙을 적용한다.
   채택된 프롬프트는 T10b의 base template으로 넘긴다.

[T8-3 필터와의 상호작용]
C/D 프롬프트가 last_integer 비율을 줄이면 T8-3 필터의 실질 효과도 바뀐다.
이 상호작용을 측정한다:
- C/D + 무필터 majority vs C/D + T8-3 필터 majority
- A + T8-3 필터(기존 +1.47pp) vs C/D + T8-3 필터
둘 다 기록하되, 채택 판정은 무필터 기준으로 먼저 내리고
필터 적용은 독립된 후속 판단으로 분리한다.

[사전 등록 판정 — 실행 후 바꾸지 않는다]
- 채택:
  A 대비 합집합 Δ >= +1.5pp이고 exact McNemar p < 0.05이며,
  hard 또는 format 어느 쪽도 A 대비 2pp를 초과해 하락하지 않고,
  합집합 invalid가 A 대비 1pp를 초과해 증가하지 않는다.
- 보류:
  합집합 Δ > 0이지만 효과 크기·유의성·guardrail 중 하나를 충족하지 못한다.
  T8 base prompt를 유지하고 T10a 산출물을 후보로만 보존한다.
- 기각:
  합집합 Δ <= 0이거나 guardrail을 위반한다. 기존 base prompt를 유지한다.
- B/C/D 중 하나가 채택되고 다른 것이 더 높더라도, 사전 등록 순서(C→D→B)로
  첫 채택 후보를 primary로 삼는다. 사후에 더 높은 것으로 갈아타지 않는다.
- holdout 또는 리더보드 결과를 보고 프롬프트 문구, 추출기, 투표 규칙을 바꾸지 않는다.

[범위 밖]
- 문제 유형별 다른 프롬프트 사용 (T10b에서 다룬다).
- few-shot 예제를 프롬프트에 넣는 것.
- 프롬프트에 코드 실행, 도구 호출, 계산기 사용을 지시하는 것.
- 어댑터 학습이나 기존 어댑터 적용.
- T8-2처럼 불일치 기반 라우팅으로 두 프롬프트를 섞는 것 (T10b에서 다룬다).

[완료 조건]
- A/B/C/D 4개 프롬프트의 k=32 pool이 합집합 3,737문항에 완성되었다 (A/B는 기존 재사용)
- 4개 프롬프트의 UTF-8 SHA-256이 config와 manifest에 기록되었다
- A를 대조군으로 한 B/C/D 각각의 paired McNemar, split guardrail, 추출 경로 분포가 남았다
- T8-3 필터와의 상호작용 분석이 남았다
- 사전 등록 규칙에 따라 채택/보류/기각이 결정되고 채택 프롬프트가 T10b로 명시 전달되었다
- 이 문서의 발표 자료용 누적 기록표에 T10a 행과 판정 근거를 추가했다

[산출물]
configs/t10a_prompt_improvement.json
artifacts/t10a_prompt_improvement/
  cot_boxed/generations.jsonl
  cot_boxed/run-metadata.json
  cot_brief/generations.jsonl
  cot_brief/run-metadata.json
  comparison.json
  comparison.md
  extraction-path-analysis.json
  filter-interaction.json
  final_config.json
  manifest.json
  tests.xml
```

### 실행 결과 — 2026-08-25

- A(base)와 B(strong-CoT)는 기존 pool을 해시 검증 후 재사용했고, C(cot-boxed)와 D(cot-brief)는 각각 3,737문항 × 32 = 119,584건을 새로 생성했다. C/D 처리량은 9.34/9.17 gen/s, peak VRAM은 모두 23,315 MiB, OOM은 0건이었다. 네 pool과 기존 T8~T9 보호 파일의 실행 전후 해시는 모두 보존됐다.
- 무필터 majority@32 합집합 정확도는 A 69.31%, B 69.52%, C 69.36%, D 68.85%였다. A 대비 B는 +0.21pp(p=0.626, 95% CI [-0.54,+0.96]), C는 +0.05pp(p=0.942, 95% CI [-0.67,+0.78]), D는 -0.45pp(p=0.254, 95% CI [-1.20,+0.29])였다.
- C는 random +1.16pp·format +3.13pp였지만 template -0.79pp·hard -1.27pp로 상쇄됐다. `boxed` 경로는 6.57%→16.82%(+10.25pp)로 늘었으나 `final_answer_marker`는 82.06%→70.95%로 줄었고, `last_integer`도 10.46%→10.83%로 소폭 늘었다. 따라서 T8-3 필터 제거 대상은 A보다 1,142건 많아져 의도한 연쇄 개선이 일어나지 않았다.
- T8-3 필터 재현은 A에서 기존 예측과 완전히 일치했다. 필터 적용 정확도는 A/B/C/D 각각 70.78%/70.78%/70.54%/70.43%였고, C는 필터 후에도 A+필터보다 -0.24pp였다. 프롬프트 채택 판정은 사전 등록대로 무필터 기준에서 분리해 내렸다.
- **판정: 보류.** B와 C는 양의 변화였지만 +1.5pp 및 유의성 게이트를 통과하지 못했고 D는 음의 변화라 기각됐다. 기존 T8 base prompt를 유지하며, T10b의 base template도 SHA-256 `d5b3c274...b07687`인 기존 프롬프트로 명시 전달했다.

### T10a C-1 — C + 동결 vote-quality filter

- `C-1`은 T10a에서 이미 계획·측정한 필터 상호작용에 붙인 파생 arm 이름이다. T10a 사전등록 파일과 C의 119,584개 생성은 수정하지 않았고, T8-3의 `drop-low-quality-votes-v1` 정책을 그대로 재현했다. 새 생성·학습·필터 탐색은 모두 0건이다.
- 합집합 정확도는 C 69.36%에서 C-1 70.54%로 **+1.18pp** 올랐다(회수 61/파손 17, exact McNemar p=5.66e-7, paired bootstrap 95% CI [+0.72,+1.66]pp). 기존 T8-3과 동일한 template-group 5-fold에서도 필터 효과가 모두 양수(+0.79~+1.59pp)였고, 5개 훈련 fold 모두 동결 필터를 선택했다.
- C-1의 split 정확도는 random 76.05%, template 74.71%, hard 39.64%, format 54.30%다. 현재 T8 무필터 대비 합집합 **+1.23pp**(p=0.00256, 95% CI [+0.45,+2.03]pp)지만 +1.5pp 채택 게이트에는 미달했다.
- 같은 필터를 base prompt에 적용한 T8-3(70.78%)보다 C-1은 **-0.24pp** 낮았다(p=0.557, 95% CI [-0.96,+0.45]pp). 즉 필터는 C를 유의하게 수리하지만 `cot-boxed` 프롬프트 자체의 추가 이득은 없다.
- **판정: 보류.** 기존 T8 base majority@32를 유지하고 vote-quality filter의 기존 보류 상태도 바꾸지 않는다. 이 결과는 동일 3,737문항 합집합의 조합 진단이며 독립 검증으로 해석하지 않는다.

[산출물] `configs/t10a_c1_vote_filter.json`, `analysis/t10a_c1_vote_filter.py`, `artifacts/t10a_c1_vote_filter/`

---

## T10b — 프롬프트 다양성 단독 실험

```text
[왜 지금 하는가]
현재 k=32 생성은 단일 프롬프트 템플릿에서 sampling noise(temperature=0.8)만으로 다양성을 만든다.
이것은 같은 오류 패턴을 반복 생산하는 구조적 한계가 있다.
문제를 다른 각도로 접근하게 하는 복수 프롬프트를 사용하면 투표 풀의 다양성이 올라가고,
서로 다른 실수를 하는 출력끼리 상쇄되어 majority vote의 정확도가 올라갈 수 있다.

T10a에서 채택된 프롬프트를 base template으로 삼고, 여기서 변형을 만든다.
T10a에서 아무것도 채택되지 않았으면 기존 T8 base prompt를 base template으로 삼는다.

[이 실험이 답하는 질문]
"프롬프트 다양성이 majority 정확도를 높이는가?"

[이 실험이 답하지 않는 질문]
- max_new_tokens 2048→4096 효과
- 다양성 + 4096 조합 효과
- hit-max 감소 효과
위 세 가지는 별도 실험이 필요하며 이 작업의 범위 밖이다.

[목표]
4~8개의 다양한 프롬프트 변형을 설계하고, k=32를 프롬프트별로 균등 배분해
단일 프롬프트 k=32 대비 정확도 향상을 측정한다.

[규정 확인]
프롬프트 변형은 모델 출력 조건 변경이며 학습·외부 도구와 무관하다.
- 모든 변형 프롬프트에 정답, 유사 예제, 외부 지식, 문제 유형 라벨을 넣지 않는다.
- 프롬프트에 코드 실행, 도구 호출, 계산기 사용을 지시하지 않는다.
- 어댑터 0개. 동일 base/revision 1개.
- 최종 답은 사전 고정 동일가중치 majority vote로만 결정한다.

[모델·생성 계약 — 실행 전에 고정]
- base: Qwen/Qwen2.5-3B-Instruct
- base revision: aa8e72537993ba99e69dfaafa59ed015b17504d1
- adapter: null
- do_sample=true, temperature=0.8, top_p=0.95
- max_input_tokens=2048, max_new_tokens=2048
- fixed pool: k=32 (프롬프트별 균등 배분)
- 평가 집합: 기존 T8과 동일한 고정 holdout 합집합 3,737문항
- 예산: 총 24시간, 최소 예비 시간 6시간

[프롬프트 다양성 전략 — 실행 전에 전체 목록을 바이트로 고정]
base template(T10a 채택 프롬프트 또는 기존 T8 프롬프트)에서 다음 축을 변형한다.
변형은 풀이 전략(approach)을 달리 지시하되 모두 동일한 출력 형식을 유지한다.

변형 축:
(a) 풀이 방향 — 순방향(forward) vs 역방향(backward/work-from-answer)
(b) 구조화 정도 — 자유 서술 vs 번호 매긴 단계
(c) 검산 유무 — 검산 지시 포함 vs 미포함
(d) 언어 — 영어 vs 한국어 (한국어 프롬프트가 한국어 수학 문제에서 유리할 수 있다)

프롬프트를 최소 4개, 최대 8개 설계한다. k=32를 프롬프트 수 N으로 나눠
프롬프트당 32/N 개를 생성한다. N이 32를 나누지 않으면 앞쪽 프롬프트에 1개씩 더 배분한다.
모든 프롬프트의 UTF-8 바이트와 SHA-256을 config에 고정한다.
holdout 결과를 보고 프롬프트를 추가·삭제·수정하지 않는다.

한국어 변형 예시 (T10a 채택에 따라 조정):

다음 문제를 단계별로 풀어주세요. 풀이 과정을 자세히 보여주고,
최종 답을 마지막 줄에 FINAL_ANSWER: <answer> 형식으로 적어주세요.
FINAL_ANSWER 줄 이후에는 아무것도 적지 마세요.

문제:
{question}

[비교군 구성 — 3개 arm]
A. 기존 T8 단일 프롬프트, max_new_tokens=2048 — 기존 pool 재사용. 재생성 0건.
C. 다중 프롬프트(N개), max_new_tokens=2048 — 신규 생성. 이 작업의 유일한 생성 arm.
E. T10a 채택 프롬프트, max_new_tokens=2048 — T10a pool 재사용. 재생성 0건.

신규 생성은 C 하나뿐이다.
C: 3,737 × 32 = 119,584건. T8 실측 13.24 gen/s 기준 약 2.5시간(프롬프트 간 길이 차에 따라 3~4시간).

[작업]
1. 프롬프트 목록을 확정하고 config에 바이트 고정한다.
   T10a 채택 결과에 따라 base template을 결정한 뒤 변형을 만든다.
   T10a가 보류/기각이면 기존 T8 base prompt를 base로 쓴다.

2. 기존 데이터 확인
   - A(base): artifacts/t8_self_consistency/generations.jsonl의 SHA-256 검증. 재생성 0건.
   - E(T10a 채택): T10a 채택 pool의 SHA-256 검증. 재생성 0건.
     T10a가 보류/기각이면 E=A이므로 비교 대상에서 제외한다.

3. C(다중 프롬프트) k=32 pool을 합집합 3,737문항에 생성한다.
   프롬프트별로 균등 배분한 sample_index를 기록한다.
   각 프롬프트의 UTF-8 SHA-256을 run-metadata에 남긴다.

4. A를 대조군으로 C/E 각각의 fixed majority@32를 평가한다.
   기록 항목은 T10a와 동일하다.
   추가로:
   - 프롬프트별 정확도, 추출 경로 분포, hit-max rate (C에서)
   - 프롬프트 간 답 일치율(inter-prompt agreement): 같은 문항에서 서로 다른 프롬프트의
     답이 얼마나 갈리는지. 높으면 다양성이 부족한 것이고 낮으면 잡음만 늘린 것일 수 있다.
     적정 수준은 기존 intra-prompt agreement@32(70.44%)와 비교해 판단한다.

5. T8-3 필터와의 상호작용
   C에 T8-3 필터를 적용한 정확도를 추가 측정한다.
   프롬프트 변형이 추출 경로 분포를 바꿀 수 있으므로 필터의 순효과를 기록한다.

6. 1,000문항 예상 시간
   C arm의 처리량으로 1,000문항 wall time을 계산한다.
   18시간 상한을 넘기면 채택 불가다.

[사전 등록 판정 — 실행 후 바꾸지 않는다]
- 채택:
  A 대비 합집합 Δ >= +1.5pp이고 exact McNemar p < 0.05이며,
  hard 또는 format 어느 쪽도 A 대비 2pp를 초과해 하락하지 않고,
  합집합 invalid가 A 대비 1pp를 초과해 증가하지 않으며,
  1,000문항 예상 시간이 18시간 이내다.
- 보류:
  합집합 Δ > 0이지만 효과 크기·유의성·guardrail·시간 중 하나를 못 넘는다.
  기존 설정을 유지하고 후보로만 보존한다.
- 기각:
  합집합 Δ <= 0이거나 guardrail/시간을 위반한다. 기존 설정을 유지한다.
- E는 T10a의 확인 재현이므로 별도 판정하지 않는다.
- holdout 또는 리더보드 결과를 보고 프롬프트 목록, 배분, 투표 규칙을 바꾸지 않는다.

[범위 밖]
- max_new_tokens 확대 (2048→4096). hit-max 감소 효과. 다양성 + 4096 조합.
- 문제별로 다른 프롬프트를 선택하는 adaptive routing (T8-2에서 이미 실험하고 미채택).
- k를 32 이상으로 늘리는 것.
- few-shot 예제를 프롬프트에 넣는 것.
- 어댑터 학습이나 기존 어댑터 적용.
- temperature나 top_p를 프롬프트별로 다르게 하는 것.

[완료 조건]
- 프롬프트 목록이 config에 바이트 고정되고 SHA-256이 manifest에 있다
- A/C/E 3개 arm의 k=32 pool이 완성됐다 (A/E는 기존 재사용, 신규 생성은 C만)
- A를 대조군으로 한 C/E paired McNemar, split guardrail, 추출 경로 분포가 남았다
- 프롬프트 간 답 일치율(inter-prompt agreement)이 기록됐다
- T8-3 필터 상호작용이 기록됐다
- 1,000문항 예상 시간이 기록됐다
- 사전 등록 규칙에 따라 채택/보류/기각이 결정됐다
- 채택된 설정이 T10c 가중 투표의 입력으로 명시 전달됐다
- 이 문서의 발표 자료용 누적 기록표에 T10b 행과 판정 근거를 추가했다

[산출물]
configs/t10b_prompt_diversity.json
artifacts/t10b_prompt_diversity/
  arm_C/generations.jsonl, run-metadata.json
  comparison.json
  comparison.md
  inter-prompt-agreement.json
  extraction-path-analysis.json
  filter-interaction.json
  runtime.json
  final_config.json
  manifest.json
  tests.xml
```

### 실행 결과 — 2026-08-25

- T10a가 보류되어 SHA-256 `d5b3c274...b07687`인 기존 T8 base prompt를 기준으로 사용했다. 풀이 방향·구조화·검산·언어 축을 4:4로 균형화한 8개 프롬프트의 UTF-8 바이트/SHA-256과 프롬프트당 4개 sample index를 실행 전에 고정했다. E는 A와 바이트 동일하므로 사전 제외했고, 신규 생성은 C의 3,737×32=119,584건뿐이다. 모델/revision·sampling·2,048 input/output·무어댑터 계약을 유지했으며 예측과 프롬프트 일치율을 라벨 로드 전에 동결했다.

| 집합 | A: 단일 T8 prompt | C: 8-prompt diversity | C−A |
|---|---:|---:|---:|
| random | 74.28% | 74.22% | -0.06pp |
| template | 73.98% | 73.30% | -0.67pp |
| hard | 39.64% | 38.36% | -1.27pp |
| format | 50.39% | 50.78% | +0.39pp |
| 합집합 | 69.31% | 68.80% | **-0.51pp** |

- 합집합에서 C는 A의 오답 87개를 회수하고 정답 106개를 파손했다. 정확도 차이는 -0.508pp, exact McNemar p=0.19496, paired bootstrap 95% CI [-1.231,+0.214]pp다. hard/format 하락 상한, invalid 증가 상한, 시간 상한은 모두 통과했지만 합집합 변화가 음수라 사전등록 규칙상 기각이다.
- 다양성 자체는 분명히 늘었다. 전체 pool agreement@32는 70.44%→62.97%(-7.47pp), 프롬프트-majority 쌍간 exact agreement는 62.67%, raw 4×4 교차표 agreement는 50.17%였고 문항당 서로 다른 유효 프롬프트-majority 답은 평균 2.80개였다. 그러나 이 불일치가 오류 상쇄로 이어지지는 않았다. 프롬프트별 majority@4는 영어 backward-free-check 65.27%가 최고, 한국어 backward-numbered-check 54.54%가 최저였다.
- C는 pass@32가 84.40%→85.34%로 늘고 hit-max가 1.22%→1.07%, 합집합 invalid가 0.77%→0.38%로 줄었지만 majority 정확도는 하락했다. 추출 경로는 `boxed` 6.57%→2.50%, `final_answer_marker` 82.06%→82.53%, `last_integer` 10.46%→14.42%로 이동했다. 즉 더 다양한 풀에서 정답 후보의 존재와 출력 유효성은 개선됐지만 저신뢰 fallback 표도 늘었다.
- 동결 T8-3 필터는 A에서 기존 예측을 완전히 재현했다. C에 적용하면 68.80%→70.19%(+1.39pp, 78개 회수/26개 파손, p=3.28e-7)로 회복했지만 A+동일 필터 70.78%보다 -0.59pp(p=0.137, 95% CI [-1.31,+0.16]pp)였다. 필터 제거 표도 A 12,702개 대비 C 17,463개로 많아 프롬프트 다양성의 추가 이득은 확인되지 않았다.
- 생성은 12,503.5초, 9.56 gen/s, peak VRAM 23,315MiB, OOM 0건이었다. 1,000문항 예상 시간은 0.929시간으로 18시간 상한을 통과했다. focused test는 69 passed/1 skipped였고, T8~T10a 보호 파일 363개의 실행 전후 SHA-256 불일치는 0개였다. C raw pool SHA-256은 `3af3bdc3...4f2a`다.
- **판정: 기각.** 기존 T8 단일 prompt majority@32를 유지한다. T10c 입력은 `artifacts/t8_self_consistency/generations.jsonl`(SHA-256 `6ddf149f...db54`), k=32, 무필터 동일가중치 majority로 명시 전달했다.

[산출물] `configs/t10b_prompt_diversity.json`, `src/prompt_diversity.py`, `artifacts/t10b_prompt_diversity/`

---

## T10c — 가중 투표 (extraction-path confidence weighting)

```text
[왜 지금 하는가]
현재 majority vote는 모든 유효 표에 동일가중치 1.0을 부여한다.
T8-3에서 투표 품질 필터를 도입해 저신뢰 표를 이진적으로 제거했고,
합집합 +1.47pp(p=6.8e-10)를 기록했지만 +1.5pp 채택 게이트에 0.028pp 미달해 보류됐다.

이진 필터의 한계는 두 가지다.
(1) 경계 결정이 거칠다 — boxed 경로(한계 정답률 37.4%)를 1.0으로 유지하지만,
    같은 문항 안의 final_answer_marker 표와 동일가중치로 세도 되는지 검증하지 않았다.
(2) 표의 품질 차이를 연속적으로 반영하지 못한다 — 출력 길이, 종료 방식, 추출 경로 등
    관측 가능한 신호를 연속 가중치로 변환하면 이진 필터보다 정밀한 투표가 가능하다.

T8-3 진단에서 문제 내 pairwise concordance(AUC)를 측정한 결과:
  강한 추출 경로 우선            AUC 0.5965
  짧은 output_tokens 우선        AUC 0.5587
  절단 안 된 것 우선             AUC 0.5088
  상충 답 적은 것 우선           AUC 0.4047 (역방향)

이 중 추출 경로와 출력 길이는 random baseline(0.5)을 넘고,
경로 AUC의 비동점 구간 실질값은 0.81로 강하다.
가중치를 부여하면 T8-3의 +1.47pp를 넘길 가능성이 있다.

[주의 — 이진 필터와 가중 투표의 관계]
가중 투표는 T8-3 이진 필터를 대체한다. 둘을 중첩하지 않는다.
가중 투표에서 last_integer의 가중치를 0으로 설정하면 이진 필터의 효과를 포함한다.
따라서 비교 대상은 무필터 T8 majority@32(69.31%)이지 T8-3(70.78%)가 아니다.
T8-3 보류 결과를 가중 투표로 넘길 수 있는지 본다.

[목표]
(1) 추출 경로, 출력 길이, hit-max 여부 등 관측 가능한 신호를 연속 가중치로 변환한
    가중 majority vote를 구현하고, 무필터 동일가중치 majority 대비 정확도 향상을 측정한다.
(2) T8-3 이진 필터를 가중 투표의 특수 경우로 포섭하고, 이진 대비 연속의 차이를 정량화한다.
(3) T10a/T10b 채택 결과와 조합했을 때의 최종 end-to-end 정확도를 확정한다.

[규정 확인]
가중 투표는 모델 출력의 후처리이며 학습·외부 도구와 무관하다.
- 가중치 입력은 src.extract의 path/explicit_candidates와 생성 메타데이터뿐이다.
- 정답 라벨, 문제 유형 라벨, 리더보드 점수, 계산 결과는 가중치·투표에 사용하지 않는다.
- 외부 모델/API/인터넷/retrieval, Python·SymPy·solver·계산기, 코드 실행을 사용하지 않는다.
- 어댑터 0개.

[가중치 계약 — 실행 전에 고정]
다음 가중치 정책을 실행 전에 고정한다. holdout 결과를 보고 가중치를 수정하지 않는다.

정책 1: 경로 기반 이진 (T8-3 재현, 대조군)
  final_answer_marker: 1.0,  boxed: 1.0,
  last_integer: 0.0,  standalone_last_line: 0.0
  hit_max_new_tokens: 0.0,  explicit_candidates >= 2: 0.0
  (T8-3과 동일 — 구현이 일관됨을 검증)

정책 2: 경로 기반 연속
  final_answer_marker: 1.0,  boxed: 0.7,
  last_integer: 0.15,  standalone_last_line: 0.1
  hit_max_new_tokens: 0.05,  explicit_candidates >= 2: 0.0
  설계 근거: 한계 정답률 비율에서 출발하되, AUC 비동점 구간 실질값(0.81)을 반영해
  final_answer_marker와 boxed 사이의 간격을 줄이고, last_integer를 0이 아닌 소량으로 둔다.
  last_integer를 완전히 버리면 해당 문항에서 유효 표가 줄어 tie가 늘어나는 부작용이 있다.
  0.15는 "대부분 무시하되 동점 깨기에는 기여"하는 수준이다.

정책 3: 경로 + 길이 혼합
  정책 2의 경로 가중치에 출력 길이 보정을 곱한다.
  길이 보정 = min(output_tokens / 100, 1.0)
  설계 근거: 출력이 극단적으로 짧은 경우(< 100 tokens) 풀이가 불완전할 가능성이 높다.
  AUC 0.5587이 이를 뒷받침한다. 100 tokens 이상은 보정 1.0으로 차별하지 않는다.

정책 4: 경로 + 완료 상태
  정책 2의 경로 가중치를 쓰되, 다음 조건을 추가로 적용한다.
  - FINAL_ANSWER 마커로 끝난 출력: ×1.0
  - FINAL_ANSWER 마커 없이 정상 종료(EOS): ×0.8
  - hit_max_new_tokens로 절단: ×0.05
  설계 근거: 마커로 끝난 출력은 모델이 답을 의도적으로 마무리한 것이고,
  절단된 출력은 풀이가 미완성이다. 이 구분은 추출 경로와 독립적인 신호다.

이 4개 정책으로 고정한다. holdout 결과를 보고 가중치를 추가·수정하지 않는다.
동점 처리: 가중 투표에서 동점이면 가중치 합이 큰 답, 그래도 동점이면 먼저 생성된 답.
모든 유효 표의 가중치 합이 0이면 무필터 majority로 폴백한다.

[모델·생성 계약]
- 새 생성을 하지 않는다. 기존 pool을 재집계만 한다.
- 적용 대상 pool:
  (a) T8 pool (artifacts/t8_self_consistency/generations.jsonl)
  (b) T10a/T10b 채택 pool (있으면)
  기존 pool의 SHA-256을 실행 전후로 검증한다.

[작업]
1. 가중 majority vote 함수를 구현한다.
   기존 src/self_consistency.py의 select_majority_vote를 확장하되,
   기존 함수를 수정하지 않고 새 함수 weighted_majority_vote를 만든다.
   기존 테스트가 깨지지 않음을 확인한다.

2. 정책 1(T8-3 재현)을 먼저 실행하고 T8-3 결과와 바이트 동일한 예측을 재현한다.
   재현에 실패하면 구현 오류이므로 멈추고 디버그한다.

3. 정책 1/2/3/4 를 T8 pool에 적용해 각각의 합집합 정확도를 계산한다.
   A(무필터 T8)를 대조군으로 paired McNemar를 돌린다.
   기록 항목:
   - 합집합 정확도, Δpp vs A, exact McNemar p, 95% CI
   - random/template/hard/format 4 split 정확도
   - 가중 동점 발생 비율, 폴백 발동 건수
   - 정책 간 예측 일치율 (정책 2 vs 3 vs 4)

4. 교차검증
   template_group 기준 5-fold CV를 T8-3과 동일한 방법으로 실행한다.
   각 정책의 폴드별 out-of-fold 정확도를 기록한다.
   과적합 신호(in-sample >> out-of-fold)가 있으면 해당 정책을 기각한다.

5. T10a/T10b 채택 pool에도 적용
   T10a/T10b에서 프롬프트나 max_tokens가 바뀌면 추출 경로 분포가 달라지므로
   가중치의 효과도 바뀐다. 채택된 pool에 정책 1/2/3/4를 적용해 효과를 재측정한다.
   이것이 최종 end-to-end 정확도가 된다.

6. 최종 조합 확정
   T10a 채택 프롬프트 × T10b 채택 max_tokens × T10c 채택 가중 정책의 end-to-end 정확도를
   원래 T8(기존 프롬프트 × 2048 × 무필터 majority@32)과 비교한다.
   이것이 T10 전체의 최종 판정이다.

[사전 등록 판정 — 실행 후 바꾸지 않는다]
- 채택:
  A(무필터 T8 majority@32) 대비 합집합 Δ >= +1.5pp이고 exact McNemar p < 0.05이며,
  hard 또는 format 어느 쪽도 A 대비 2pp를 초과해 하락하지 않고,
  합집합 invalid가 A 대비 1pp를 초과해 증가하지 않으며,
  5-fold CV에서 과적합 신호가 없다.
- 정책 2/3/4 중 복수가 채택 게이트를 통과하면 합집합 Δ가 가장 큰 것을 채택한다.
  동점이면 가중치가 단순한(파라미터 수가 적은) 것을 택한다: 정책 2 > 4 > 3.
- 보류:
  합집합 Δ > 0이지만 효과 크기·유의성·guardrail·CV 중 하나를 못 넘는다.
  무필터 majority를 유지하고 가중 정책을 후보로만 보존한다.
- 기각:
  합집합 Δ <= 0이거나 guardrail/CV를 위반한다. 무필터 majority를 유지한다.
- 정책 1은 T8-3 재현 검증용이며 독립 채택 대상이 아니다.
  T8-3 결과(보류, +1.47pp)를 가중 투표에서 넘는 것이 이 작업의 존재 이유다.
- holdout 또는 리더보드 결과를 보고 가중치, 정책, 투표 규칙을 바꾸지 않는다.

[범위 밖]
- 학습 기반 가중치 (verifier, reward model, selector).
- 문제별 가중치 (문제 유형이나 난이도에 따라 다른 가중치).
- 표의 수치적 내용을 비교해 가중치를 매기는 것 (arithmetic verifier에 해당).
- k를 32 이상으로 늘리는 것.
- 새 생성이나 어댑터 적용.

[완료 조건]
- 정책 1이 T8-3 결과를 바이트 동일하게 재현했다
- 정책 1/2/3/4의 합집합 paired McNemar, 4 split guardrail, 가중 동점 비율이 남았다
- 5-fold CV가 정책별로 기록됐고 과적합 여부가 판정됐다
- T10a/T10b 채택 pool과의 조합 결과가 기록됐다
- T10 전체(T10a×T10b×T10c)의 end-to-end 정확도가 원래 T8과 비교됐다
- 사전 등록 규칙에 따라 채택/보류/기각이 결정됐다
- 기존 T8·T8-1·T8-2·T8-3·T8-4·T9 산출물과 설정 파일의 해시가 보존되어 있다
- 가중 정책 파라미터가 config에 고정되고 SHA-256이 manifest에 있다
- 이 문서의 발표 자료용 누적 기록표에 T10c 행과 판정 근거를 추가했다

[산출물]
configs/t10c_weighted_voting.json
src/weighted_vote.py
artifacts/t10c_weighted_voting/
  holdout/
    policy1_predictions.jsonl
    policy2_predictions.jsonl
    policy3_predictions.jsonl
    policy4_predictions.jsonl
  comparison.json
  comparison.md
  cross-validation.json
  end-to-end.json
  final_config.json
  manifest.json
  tests.xml
```

### 실행 결과 — 2026-08-26

- 실행 전에 4개 정책과 곱셈식 보정·가중합 우선·동률 시 첫 양수가중 표·전부 0일 때 무필터 majority 폴백을 `configs/t10c_weighted_voting.json`에 고정했다(SHA-256 `9871cc56...15ea`). 새 생성·학습·어댑터·문항별 특성·정답 기반 선택은 0건이다. T10a/T10b가 모두 기존 설정을 유지했으므로 입력은 T8 단일 프롬프트·2,048 tokens·k=32 pool이다.

| 정책 | 합집합 | Δ vs T8 | exact McNemar p | 95% CI | random | template | hard | format | 가중 동점 | 폴백 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 경로 이진(T8-3 재현) | 70.78% | +1.47pp | 6.80e-10 | [+1.00,+1.95]pp | 75.57% | 75.50% | 40.73% | 53.12% | 5.03% | 47 |
| 2 경로 연속 | 70.48% | **+1.18pp** | **6.21e-8** | [+0.75,+1.61]pp | 75.26% | 75.26% | 40.73% | 51.95% | 2.44% | 1 |
| 3 경로+길이 | 70.48% | +1.18pp | 6.21e-8 | [+0.75,+1.61]pp | 75.26% | 75.26% | 40.73% | 51.95% | 2.41% | 1 |
| 4 경로+완료 | 70.46% | +1.15pp | 8.91e-7 | [+0.69,+1.61]pp | 75.20% | 75.20% | 40.73% | 52.34% | 1.90% | 1 |

- 정책 1은 기존 T8-3 answer map 3,737개와 mismatch 0건이었고, canonical prediction bytes SHA-256도 양쪽 모두 `7414912f...e570`으로 같았다. 즉 새 가중 구현이 이진 필터를 정확히 특수 경우로 포섭했다.
- 채택 대상인 2/3/4 중 정책 2가 최고였다. T8 오답 56개를 회수하고 정답 12개를 파손해 +1.177pp였으며 유의했지만, 사전 채택선 +1.5pp에는 0.323pp 못 미쳤다. hard/format·invalid guardrail은 모두 통과했다. 정책 3은 정책 2와 예측 1개만 달랐고 정확도는 같았으며, 정책 4는 36개가 달라져 +1.151pp였다.
- 연속 가중은 이진 정책 1을 넘지 못했다. 정책 2/3은 정책 1 대비 -0.294pp(회수 11/파손 22, p=0.0801, 95% CI [-0.60,+0.01]pp), 정책 4는 -0.321pp(p=0.0730)였다. 저신뢰 표를 소량 남기는 방식보다 완전히 제거하는 이진 경계가 이 pool에서는 더 강했다.
- template-group 5-fold에서 정책 2의 고정 validation Δ는 +0.41/+0.93/+2.12/+1.36/+1.06pp였고, 5개 training fold 모두 정책 2를 선택했다. 정책 1/3/4도 각 5개 fold 모두 후보를 선택했으며 selected OOF Δ가 in-sample Δ와 같아 과적합 신호는 없었다.
- 원본 pool은 작업공간 체크아웃에서 119,584개 줄이 CRLF로 변환돼 raw SHA-256이 달랐지만, 파일을 수정하지 않고 LF 정규화로 검증한 내용 해시는 메타데이터와 같은 `6ddf149f...db54`였다. 실행 전후 raw/canonical 해시와 보호 대상 250개 파일의 불일치는 0건이다. T10c 집중 테스트는 69 passed였고, 기존 작업공간 해시 drift에 이미 걸리던 artifact-integrity 검사 2개는 `test-environment.json`에 원인과 실제/기대 해시를 남기고 분리했다.
- **판정: 보류.** 채택 대상 최선인 정책 2가 유의성과 모든 guardrail/CV를 통과했지만 +1.5pp 효과 게이트를 넘지 못했다. 최종 T10 조합은 T10a/T10b/T10c 모두 기존 선택을 유지하므로 원래 T8 단일 prompt × 2,048 × 무필터 majority@32와 동일하며, 합집합 69.31%, Δ=0.00pp(p=1.0)다.

[산출물] `configs/t10c_weighted_voting.json`, `src/weighted_vote.py`, `artifacts/t10c_weighted_voting/`

---

## T10d — 동일 베이스 3-view flat filtered majority@96

```text
[왜 지금 하는가 — 2026-08-27 추가]
T8-3(base prompt + 동결 필터)는 합집합 70.78%, T10a C-1(cot-boxed + 같은 필터)은 70.54%,
T8-4(RFT LoRA + 같은 필터)는 70.30%다. 단일 후보의 순위만 보면 T8-3이 가장 높지만 세 후보는
생성 분포가 다르다. base와 cot-boxed는 프롬프트가 다르고, RFT는 같은 지정 베이스에 허용된 LoRA만
적용한다. 따라서 각 arm의 32개 표를 arm별 승자 3개로 축약하지 않고, 동결 필터 뒤 남은 표 전체를
한 번에 합치면 서로 다른 오류를 상쇄할 수 있다.

[정직성 계약 — 중요]
- 이 조합은 T8/T8-3/T8-4/T10a 결과를 이미 본 뒤 제안된 사후 탐색 후보다. 2026-08-27에 최종
  바이트 규칙을 config로 고정한 뒤 정확한 재집계를 했지만, 개념 자체가 독립 사전등록된 것은 아니다.
- 따라서 +1.5pp·McNemar 수치 게이트 통과 여부는 기술적 진단으로 기록하며, 독립 검증을 통과한
  confirmatory adoption이라고 부르지 않는다.
- 각 surface의 세 예측은 정답을 불러오기 전에 생성·동결한다. 정답은 holdout 평가에서만 사후 사용하고,
  리더보드에는 라벨이 없으며 정확도를 계산하지 않는다.
- holdout 또는 리더보드 결과를 보고 arm, arm 순서, 필터, 폴백, 투표·동점 규칙을 다시 바꾸지 않는다.

[규정 판단과 남은 확인]
- 기록된 운영진 답변은 Majority Voting·Self-Consistency 등 다중 샘플링을 자유롭게 허용하고,
  동일 Qwen2.5-3B-Instruct 베이스 위의 복수 LoRA 앙상블도 합리적 범위에서 허용한다.
- 이 작업은 외부 모델/API/인터넷/retrieval, 계산기, Python·SymPy solver, 계산 verifier를 사용하지 않는다.
  테스트 답은 모델이 이미 출력한 정수 문자열의 빈도만으로 결정한다.
- 문항당 샘플 수의 명시적 상한은 확인된 규정에 없다. 다만 정확한 96-sample 구성과
  extraction path·hit-max 기반 표 제외까지 운영진이 개별 승인한 원문은 아직 없다.
- 그러므로 결과물은 제출 가능한 후보로 만들되, 최종 제출 전 다음 두 항목을 서면 확인한다:
  (1) base prompt 32 + cot-boxed 32 + 같은 base의 RFT LoRA 32를 합친 96표,
  (2) last_integer/standalone_last_line/hit-max/상충 explicit 표의 비수학적 제외.

[모델·생성 계약]
- 공통 base/revision/tokenizer:
  Qwen/Qwen2.5-3B-Instruct @ aa8e72537993ba99e69dfaafa59ed015b17504d1
- arm 1 `base`: adapter=null, T8 base prompt, k=32
- arm 2 `cot_boxed`: adapter=null, T10a cot-boxed prompt, k=32,
  prompt SHA-256 5d78ed32f7344f78cec9144e5944159832de9afb084f0aac7abe5085bb500a91
- arm 3 `rft_r1`: T8 base prompt, k=32, adapter artifacts/t6_sft_v1/adapters/rft_r1,
  adapter SHA-256 c5351995b9874fa27778d564e0748b6e694a26936b1372711535bc28b7c38bd1
- 새 생성 0건, 새 학습 0건. 기존 holdout 3 × 119,584건과 leaderboard 3 × 32,000건만 읽는다.
- 여섯 generation pool은 raw SHA-256과 CRLF→LF 정규화 내용 해시를 실행 전후로 모두 확인한다.

[투표 계약 — flat majority@96]
1. 각 arm에 T8-3의 동결 `drop-low-quality-votes-v1`을 독립 적용한다.
2. 한 arm의 표가 전부 제거되면 그 arm에 한해서 기존 T8-3 계약대로 무필터 32표를 복원한다.
3. 남은 후보를 `base → cot_boxed → rft_r1`, 각 arm 안에서는 sample_index 0..31 순서로 연결한다.
4. 추출 답이 없는 후보는 유효 표에서 제외하고, 나머지 정수 답 문자열을 동일가중치로 한 번만 센다.
5. 최빈 답이 동률이면 위 연결 순서에서 먼저 등장한 유효 답을 선택한다.

이것은 arm별 winner 3개를 다시 투표하는 계층 다수결이 아니다. 최대 96개 원표를 직접 합치는
단일 flat majority이며, 문제 텍스트·정답·문제 유형·리더보드 점수를 입력으로 받지 않는다.

[평가·판정]
- primary: 원래 T8 base unfiltered majority@32 대비 합집합 3,737문항 exact McNemar.
- secondary: T8-3 base filtered majority@32 대비 paired 비교.
- guardrail: random/template/hard/format, invalid 증가, template-group 5-fold 방향 일관성.
- 기존 수치 게이트(Δ >= +1.5pp, p < 0.05, hard/format -2pp 이내, invalid +1pp 이내)를
  그대로 계산하되, 사후 후보이므로 `exploratory_passes_numerical_gate`와 정식 채택을 구분한다.
- 리더보드 831행은 행·ID·정수·바이트 무결성과 기존 세 제출본 대비 변경 수만 기록한다.

[실행]
/Users/kunho/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m analysis.t10d_flat_vote

[완료 조건]
- config의 세 arm·pool canonical LF SHA-256·필터·flat 투표·동점 규칙이 검증된다.
- 단위 테스트가 통과하고 holdout/leaderboard가 같은 집계 함수로 처리된다.
- holdout 예측 파일이 라벨 로드 전에 기록되고 manifest에 freeze SHA-256이 남는다.
- 루트 submission.csv와 artifact 사본이 831행·ID 고유·전량 canonical integer·바이트 동일이다.
- 여섯 원본 generation pool의 실행 전후 identity가 모두 동일하다.

[산출물]
configs/t10d_flat_filtered_majority_k96.json
analysis/t10d_flat_vote.py
tests/test_t10d_flat_vote.py
artifacts/submissions/t10d_flat_filtered_majority_k96/
  submission.csv
  leaderboard-predictions.jsonl
  leaderboard-audit.json
  holdout-predictions.jsonl
  holdout-comparison.json
  comparison.md
  manifest.json
  tests.xml
submission.csv
```

### 실행 결과 — 2026-08-27

- config를 SHA-256 `5953fa2d...f6de`로 고정하고 단위 테스트 5개를 통과했다. 여섯 원본 pool은
  실행 전후 raw/canonical LF identity가 모두 같았고 새 생성·학습·파일 수정은 0건이다.
- 합집합 정확도는 **71.18%(2,660/3,737)**다. 원래 T8 69.31%(2,590) 대비
  **+1.87pp**, 회수 120 / 파손 50, exact McNemar `p=8.02e-8`, paired bootstrap 95% CI
  `[+1.20,+2.54]pp`다. 효과·유의성·hard/format·invalid의 기존 수치 게이트를 모두 통과했다.
- split 정확도와 T8 대비 변화는 random **76.18%(+1.89pp)**, template **75.38%(+1.41pp)**,
  hard **41.82%(+2.18pp)**, format **55.08%(+4.69pp)**로 네 split 모두 양수다.
  template-group 5-fold도 `+1.50/+1.20/+3.31/+2.04/+1.32pp`로 전부 양수였다.
- 현재 holdout 최고였던 T8-3 70.78%보다는 +0.40pp 높지만 회수 68 / 파손 53,
  `p=0.203`, 95% CI `[-0.19,+0.99]pp`로 **T8-3 대비 우위는 유의하지 않다**.
- holdout에서 필터·arm별 폴백 뒤 358,752개 최대 후보 중 328,000개가 flat pool에 남았고,
  유효 표는 327,657개였다. 최종 동점은 68/3,737(1.82%), invalid 예측은 0건이다.
- 리더보드 831행은 79,776개 최대 후보 중 75,038개를 집계했고 유효 표 74,976개,
  동점 10건, invalid 0건이다. 최종 답은 T8-3/C-1/RFT-filtered 제출본 대비 각각
  61/66/70건 바뀌었다. 리더보드 라벨은 없으므로 정확도는 계산하지 않았다.
- 루트 `submission.csv`와 artifact 사본은 831행·ID 831개·전량 canonical integer이고 바이트 동일하다.
  SHA-256은 `4caa701cc1cb00ed39360f367fe92d122221972daa8ff53064aa2dff3421276e`다.
- **판정: 규정 확인 대기 후보.** 수치상 `exploratory_passes_numerical_gate`지만 사후 발견 조합이므로
  독립 사전등록 채택으로 승격하지 않는다. 운영진이 96표 구성과 동결 품질 필터를 서면 허용하면
  이 제출본을 최종 후보로 사용하고, 허용이 불명확하면 기존 C-1 후보로 돌아간다.

---

## T10e — 3-view arm-normalized filtered voting@96

```text
[왜 T10d 뒤에 추가하는가 — 2026-08-27]
T10d flat majority는 필터 뒤 유효 표가 많이 남은 arm에 더 큰 영향력을 준다. 이 차이는 답의 품질이
아니라 hit-max·추출 경로 등으로 제거된 표의 수에서도 생길 수 있다. T10e는 T10d와 완전히 같은
base/cot_boxed/rft_r1 generation pool과 같은 동결 필터를 사용하되, 각 arm의 유효 답 분포가 총질량
1이 되도록 정규화한 뒤 세 분포를 더한다. 새 생성·학습·모델 선택은 0건이다.

[고정 입력]
- 원천 규칙은 `configs/t10d_flat_filtered_majority_k96.json`이며, 기대 SHA-256은
  `5953fa2df41dbfdc994568558ed03a258c18adb3f643200ce4070083eb8df6de`다.
- arm과 순서는 `base → cot_boxed → rft_r1`, 각 k=32로 T10d와 동일하다.
- 각 arm에 T8-3 `drop-low-quality-votes-v1`을 독립 적용하고, 전부 제거된 arm만 기존 무필터
  32표로 복원한다. 복원 뒤에도 추출 가능한 정수 답이 0개면 그 arm의 기여는 0이다.
- 리더보드 기준 파일은 반드시 `data/deep_chal_math_leaderboard_filtered.csv`다. 이 파일을 CSV로
  파싱한 논리 행 831개의 ID와 순서를 그대로 유지하며, 1,000행 원본 리더보드로 확장하지 않는다.

[arm-normalized voting 계약]
각 답 a의 최종 점수는 다음과 같다.

score(a) = Σ_arm count_arm(a) / valid_votes_arm

- 유효 표가 있는 각 arm의 전체 점수 질량은 정확히 1이다. 따라서 세 arm이 모두 유효하면 총질량은 3이다.
- 부동소수점 오차가 동률을 깨지 않도록 점수는 `fractions.Fraction`의 정확한 유리수로 합산한다.
- 최고 점수가 동률이면 동결 순서 `base → cot_boxed → rft_r1`, arm 내부 sample_index 0..31에서
  가장 먼저 나타난 유효 답을 고른다.
- 문제 텍스트·유형·정답·리더보드 점수는 집계 입력으로 사용하지 않는다. holdout 예측 JSONL을
  정답 로드 전에 먼저 기록하고, 이후에만 평가한다.

[실행]
/Users/kunho/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m analysis.t10e_arm_normalized_vote

[산출물]
configs/t10e_arm_normalized_voting.json
analysis/t10e_arm_normalized_vote.py
tests/test_t10e_arm_normalized_vote.py
artifacts/submissions/t10e_arm_normalized_k96/
  submission.csv
  leaderboard-predictions.jsonl
  leaderboard-audit.json
  holdout-predictions.jsonl
  holdout-comparison.json
  comparison.md
  manifest.json
  tests.xml
submission.csv
```

### 실행 결과 — 2026-08-27

- T10e config SHA-256은 `c4953d644c194cadb11f4cbc24cd38da3482d4fdf0ded0106701c6646e9964c2`이며,
  arm 질량·flat과의 차이·정확 동률·zero-valid arm·전부 invalid·비정규 정수 거부를 포함한 집중
  단위 테스트 7개를 통과했다. T10d의 동결 예측 map도 holdout/leaderboard 양쪽에서 바이트 일치했다.
- 합집합 정확도는 **71.29%(2,664/3,737)**다. 원래 T8 69.31%(2,590) 대비
  **+1.98pp**, 회수 128 / 파손 54, exact McNemar `p=4.12e-8`, paired bootstrap 95% CI
  `[+1.28,+2.68]pp`로 기존 T8 기준 수치 게이트를 통과했다.
- T10d flat 71.18% 대비 증분은 **+0.11pp**뿐이다. 회수 15 / 파손 11,
  `p=0.557`, 95% CI `[-0.16,+0.37]pp`이므로 arm 정규화 자체의 개선은 유의하지 않다.
- split 정확도와 T10d 대비 변화는 random **76.18%(+0.00pp)**, template
  **75.63%(+0.24pp)**, hard **42.18%(+0.36pp)**, format **54.69%(-0.39pp)**다.
  template-group 5-fold의 T8 대비 변화는 모두 양수였지만, T10d 대비 변화는
  `+0.27/-0.13/+0.00/+0.00/+0.40pp`로 한 fold가 음수여서 증분 방향 일관성은 없다.
- `data/deep_chal_math_leaderboard_filtered.csv`의 논리 행 **831개**만 같은 순서로 집계했다.
  830문항은 세 arm, 1문항은 두 arm이 유효했고, 정규화 동점 3건·invalid 0건이다. T10d flat과
  최종 답 9개가 달라졌으며 라벨이 없으므로 리더보드 정확도는 계산하지 않았다.
- 루트 `submission.csv`와 artifact 사본은 831행·ID 831개·전량 canonical integer·바이트 동일하고,
  SHA-256은 `19920b909702da2e1b7037d60fed8616b912a6728c6e12135bc53150c62a26d8`이다.
- **판정: 사용자 요청 제출 후보, T10d 보존.** arm 정규화가 T10d보다 4문항 순증해 루트 제출본에는
  T10e를 반영했다. 다만 +0.11pp는 유의하지 않고 5-fold 증분도 전부 비음수가 아니므로 T10d artifact를
  폴백으로 보존한다. 96표 구성과 동결 품질 필터에 대한 운영진 서면 확인 필요성은 T10d와 동일하다.

---

## T11 — AIMO식 hard-CoT SFT → correct/wrong DPO (생성 pass@1 개선)

근거 원문은 [AIMO-2 우승 보고서](https://arxiv.org/abs/2504.16891)와
[AIMO-2 2위 팀 공개 솔루션](https://github.com/imagination-research/aimo2)이다. 우승 팀은 문제당 최대
32개 CoT를 만들고 student pass-rate가 낮은 문제에 생성 예산을 더 배분한 뒤 기대 정답과 맞지 않는
풀이를 제거했다. 2위 팀은 high-difficulty CoT SFT 뒤 correct/shorter chosen을 쓰는 DPO를 수행했고,
로컬 판단의 1차 지표로 aggregated accuracy뿐 아니라 average sample accuracy를 함께 사용했다.

```text
[왜 지금 하는가 — 2026-08-27 추가]
사용자가 확인한 public leaderboard 최고 제출은 artifacts/submissions/t10a_c1_filtered_k32/ 이다.
그 기반인 C(cot_boxed) 생성의 합집합 3,737문항 sample accuracy는 60.5415%, 무필터 majority@32는
69.3604%, 동결 T8-3 필터를 적용한 C-1 majority@32는 70.5379%다. C-1은 같은 생성물을 재집계해
C보다 +1.1774pp를 회수했지만, 필터가 고칠 수 있는 것은 저품질 표이지 생성 자체의 수학 정답률은 아니다.

T8의 k 곡선은 k=4/8/16/32에서 64.68/67.22/68.45/69.31%로 이미 포화 방향이고,
T10b prompt diversity도 기각됐다. 다음 레버를 k=64나 새 prompt로 두지 않고, AIMO에서 실제로 사용한
"저 pass-rate 난문에 정답 검증 CoT를 집중 → SFT → correct/shorter preference 학습"으로 둔다.

[AIMO 근거와 이 대회로 옮길 범위]
1. AIMO-2 우승 팀:
   - 540K 문제와 3.2M long-reasoning CoT를 구축했다.
   - 문제당 최대 32개 풀이를 만들고 Qwen2.5-72B-Math-Instruct의 평균 pass-rate로 난도를 추정해
     어려운 문제에 더 많은 풀이를 생성했다.
   - 기대 정답과 맞지 않는 풀이를 최종 제거한 뒤 CoT SFT를 수행했다.
2. AIMO-2 2위 팀:
   - high-difficulty reasoning trajectory로 SFT한 뒤 2K preference pair로 DPO했다.
   - chosen은 정답이어야 하고, rejected는 정답 또는 오답일 수 있으며, chosen이 더 짧은 pair를 골랐다.
   - 32회 표본의 평균 정답 수를 14.63에서 18.90으로 보고했고 aggregated 정답은 20/30에서 21/30이었다.
     이는 "각 생성의 정답률을 먼저 본다"는 근거지만 SFT와 DPO의 개별 인과효과로 해석하지 않는다.
3. 가져오지 않는 것:
   - 두 팀의 TIR/Python 실행은 본 대회에서 금지다. teacher·학습 target·최종 추론 모두 text-only CoT다.
   - AIMO의 14B/32B, 수백만 행, 8×A800 규모 효과를 3B/단일 GPU에 그대로 외삽하지 않는다.
   - 아래 k=8, hard threshold, LoRA/DPO hyperparameter는 AIMO 수치가 아니라 이 저장소용 사전 등록값이다.

[핵심 가설]
C-style prompt에서 낮은 pass@1을 보이는 canonical train 난문에 정답·완결성이 검증된 text CoT를
집중해 SFT하고, 같은 문항의 correct/wrong preference로 DPO하면 C prompt의 sample accuracy가 오른다.
sample accuracy가 오르면 k를 32보다 늘리지 않아도 C-1 majority@32가 함께 오를 수 있다.

[절대 고정 기준선]
- base: Qwen/Qwen2.5-3B-Instruct @ aa8e72537993ba99e69dfaafa59ed015b17504d1
- prompt: T10a C(cot_boxed), SHA-256 5d78ed32f7344f78cec9144e5944159832de9afb084f0aac7abe5085bb500a91
- generation: temperature=0.8, top_p=0.95, max_new_tokens=2048, bf16
- final inference budget: 문제당 k=32. k=64 이상을 실험하거나 채택하지 않는다.
- vote filter: T8-3 drop-low-quality-votes-v1을 바이트 그대로 재사용한다. 새 필터를 탐색하지 않는다.
- raw 기준선: C sample accuracy 60.5415440193%, C majority@32 69.3604495585%
- deployment 기준선: C-1 filtered majority@32 70.5378645973%(2,636/3,737)
- 현재 제출 폴백: artifacts/submissions/t10a_c1_filtered_k32/ 및 T10e artifact를 모두 보존한다.

[데이터 경계 — 먼저 검증하고 한 건이라도 겹치면 중단]
1. 문제 원장은 data/canonical/train.csv만 쓴다.
2. 다음 ID는 probe·teacher 호출·SFT·DPO에서 전부 제외한다.
   - artifacts/t8_self_consistency/holdout_union_ids.txt의 3,737개
   - data/splits/rft_validation_500_ids.txt의 500개
   - data/suspect_set/ids.txt의 0/48 의심 문항
3. teacher에는 train 문제만 보낸다. holdout 4종, 리더보드 831/1,000문항, 최종 test는 보내지 않는다.
4. 학습 후보 질문을 data/deep_chal_math_leaderboard.csv 원본 1,000행 전체와 exact question,
   숫자 정규화 template, 근접 중복으로 대조한다. 하나라도 걸리면 해당 train 문항을 제외하고 audit에 남긴다.
   831행 filtered 파일만으로 오염 검사를 축소하지 않는다.
5. 라벨은 probe/teacher 출력이 디스크에 동결된 뒤 정확도 판정과 학습 데이터 필터에만 쓴다.
   정답을 student/teacher prompt에 넣지 않는다.

[0단계 — C-style student difficulty probe]
1. 위 경계를 통과한 train 문제 전부에 frozen base+C prompt로 k=8을 생성한다.
   sample seed는 42000..42007로 고정한다. 생성 원문·finish_reason·output_tokens를 전부 보존한다.
2. 각 문항의 correct_count c를 0..8로 계산한다. invalid와 hit-max는 오답으로 센다.
   pass1_proxy = c / 8 이며 정답 문자열 exact match만 사용한다.
3. hard 후보는 c<=2(pass1_proxy<=0.25)다. c 오름차순, 그다음
   sha256('t11-hard-v1:'+id) 오름차순으로 정렬해 최대 2,000문항을 고른다.
4. anchor 후보는 c>=6이고 아래 trace 품질 필터를 통과한 자기생성 정답 풀이가 있는 문항이다.
   c=3..5는 이번 학습에서 쓰지 않는다. threshold를 결과를 보고 바꾸지 않는다.
5. c 분포, hard 문항 수, 문제 길이·유형 분포와 기존 T5 c 분포의 차이를 difficulty-audit.json에 남긴다.

[1단계 — 학습용 teacher 계약과 64문항 preflight]
teacher는 data.md가 허용한 학습 데이터 구축 전용 teacher여야 한다. 정확한 provider/model/revision,
system/user prompt UTF-8 바이트, sampling 값, tool_use=false, 비용·시간 상한을 config에 먼저 고정한다.
teacher 선택은 실험 arm이 아니며 실행 중 교체하지 않는다. 설정이 비어 있으면 생성하지 말고 중단한다.

과거 상용 teacher 시도 3회가 약 $4.6를 쓰고 채택 0행으로 끝났으므로 곧바로 2,000문항을 호출하지 않는다.
hard 목록 첫 64문항에 문항당 4개 text-only CoT를 먼저 생성한다. 다음을 모두 만족해야 full run으로 간다.
- 64문항 중 32문항 이상에서 아래 필터를 통과한 정답 trace가 최소 1개 있다.
- 256개 출력 중 accepted correct trace가 64개 이상이다.
- tool/code 의존 trace 0개, holdout/leaderboard 전송 0건이다.
- 실측 단가와 처리량으로 full run이 config의 동결 비용·시간 상한 안에 든다.

하나라도 실패하면 status=teacher_gate_failed로 종료하고 SFT/DPO를 돌리지 않는다. self-teacher나 다른 API로
조용히 폴백하지 않는다. 통과하면 hard 문항당 4개를 만들고, accepted correct가 0개인 문항만 두 번째
4개를 추가한다. 문항당 최대 8개이며 정답은 prompt에 넣지 않는다.

[2단계 — trace 품질 필터와 학습 데이터 구축]
accepted correct trace는 다음 조건을 전부 만족해야 한다.
- src.extract가 읽은 canonical integer가 gold answer와 exact match
- finish_reason가 length/max_tokens가 아니고 output_tokens < 2048
- 마지막 non-empty line이 정확히 ^FINAL_ANSWER: -?(?:0|[1-9][0-9]*)$
- boxed와 FINAL_ANSWER를 포함한 explicit candidate의 서로 다른 값이 정확히 1개
- 빈 풀이가 아니며 assistant token 128 이상
- backtick 3개의 code fence, <tool_call>, Python/SymPy/import/def/exec 형태의 코드·도구 호출이 없음

SFT 데이터:
- hard 문항당 최대 2개. accepted trace를 token 길이와 content SHA로 정렬해 p50과 p75에 가장 가까운
  서로 다른 trace를 고른다. 이전 RFT의 shortest-only 편향을 반복하지 않는다.
- anchor는 0단계 c>=6의 valid correct C-style 자기생성 trace에서 문항당 1개를 뽑는다.
- anchor 행 수는 hard 행 수의 1/4 이하, 즉 전체의 최대 20%다. 나머지는 hard여야 한다.
- 같은 question/id가 validation 500·holdout·리더보드 보호 집합에 없는지 최종 assert한다.

DPO 데이터:
- 문항당 최대 1 pair, 총 2,000 pair 이하. chosen은 위 조건을 통과한 correct trace다.
- 우선 pair(전체의 최소 75%): rejected는 같은 문항·같은 C prompt의 완결된 오답 trace다.
  malformed/hit-max가 아니라 논리적으로 끝났지만 final answer가 틀린 trace를 쓰며, chosen과 token 길이가
  가장 가까운 것을 골라 단순 길이 신호와 correctness 신호를 분리한다.
- 보조 pair(전체의 최대 25%): rejected도 정답일 수 있으나 chosen_tokens <= 0.8*rejected_tokens이고
  둘 다 완결·무상충·text-only여야 한다. 불필요하게 긴 풀이를 줄이는 신호다.
- correct/wrong pair가 75% 미만이면 DPO data gate 실패로 처리하고 SFT까지만 평가한다.
- 외부 embedding similarity 필터는 쓰지 않는다. AIMO 2위 팀도 similarity 추가 arm이 제출 모델과
  비슷했다고 보고했으므로 새 의존성과 탐색 축을 만들지 않는다.

[3단계 — hard-CoT SFT]
T6-1에서 확인한 로컬 구현 조건을 그대로 승계한다.
- 시작점은 base revision이다. 기존 RFT/GenSelect adapter 위에 이어 학습하지 않는다.
- bf16 LoRA r=64, alpha=128, attention+MLP projection, packing=False
- max_seq_length=4096(input 최대 2048 + response 최대 2048), assistant response token에만 loss
- effective batch=32, seed=42, warmup_ratio=0.03, cosine, 최대 1 epoch
- LR {1e-5, 3e-5, 1e-4}; 각 0.25/0.5/0.75/1.0 epoch checkpoint를 보존한다.
- data/splits/rft_validation_500.csv에 frozen C prompt k=8(seed 52000..52007)로 모든 checkpoint를 평가한다.
  holdout 3,737문항은 checkpoint·LR 선택에 쓰지 않는다.

[4단계 — correct/wrong DPO]
DPO data gate를 통과한 경우에만 3단계의 validation-best SFT checkpoint에서 시작한다.
- reference policy는 그 SFT checkpoint를 동결한 복사본이다.
- bf16 LoRA, packing=False, max_seq_length=4096, beta=0.1, LR=1e-6,
  effective batch=16, seed=43, 최대 1 epoch
- 0.25/0.5/0.75/1.0 epoch checkpoint를 같은 validation C prompt k=8로 평가한다.
- KTO로 자동 대체하지 않는다. 이번 T11의 preference 방법은 AIMO-2 2위 팀이 실제 사용한 DPO로 고정한다.
  KTO는 별도 사전 등록 실험이 있을 때만 실행한다.

[5단계 — holdout을 열기 전 단일 후보 선택]
sample accuracy(pass@1)는 다음처럼 정의한다.

  sample_accuracy = 모든 validation 문항·8개 sample 중 exact-correct sample 수 / 전체 sample 수

invalid·hit-max는 오답이다. 문항별 correct_count/8 차이를 단위로 paired bootstrap 20,000회(seed=42)를
계산한다. base, 모든 SFT checkpoint, 모든 DPO checkpoint 중 validation sample accuracy가 가장 높은 하나만
고른다. 정확히 동률이면 invalid rate → hit-max rate → mean output tokens → 더 이른 checkpoint 순으로 고른다.

선택 후보가 frozen base 대비 validation sample accuracy +1.0pp 이상이고 paired bootstrap 95% CI 하한이
0보다 클 때만 holdout으로 간다. 아니면 T11을 validation_reject로 끝내고 새 holdout 생성은 0건으로 둔다.
선택된 stage/LR/checkpoint/adapter SHA를 final_config.json에 먼저 쓰고 그 뒤에는 바꾸지 않는다.

[6단계 — 단 한 번의 최종 holdout 평가]
동결된 후보 하나만 합집합 3,737문항에서 C prompt k=32로 생성한다. generation seed·sample index·질문 순서는
기존 C pool과 맞추고, 정답을 읽기 전에 raw generations와 predictions를 동결한다.

평가를 두 층으로 분리한다.
1. 생성 품질 primary:
   raw 119,584 sample의 sample accuracy, 문항별 correct-share, invalid, hit-max, 평균/중앙/p95 tokens,
   pass@32, agreement@32를 frozen C raw와 비교한다. 통계 단위는 sample이 아니라 문항별 correct-share이며
   paired bootstrap 20,000회로 CI를 계산한다.
2. 제출 품질 secondary:
   후보 raw pool에 동결 T8-3 필터를 그대로 적용해 candidate C-1 majority@32를 만든다.
   기존 T10a C-1과 합집합 exact McNemar, paired bootstrap, random/template/hard/format split을 비교한다.
   필터 조건·fallback·tie-break는 절대 다시 맞추지 않는다.

[사전 등록 판정 — 실행 후 바꾸지 않는다]
- 채택:
  (a) raw sample accuracy가 C보다 +1.5pp 이상이고 문항 paired-bootstrap 95% CI 하한 > 0,
  (b) candidate C-1 majority@32가 기존 C-1보다 +1.5pp 이상이고 exact McNemar p < 0.05,
  (c) hard/format 어느 split도 기존 C-1 대비 2pp 초과 하락하지 않고,
  (d) raw invalid와 hit-max가 각각 C보다 1pp 초과 증가하지 않으며,
  (e) 1,000문항 k32 예상 추론이 18시간 안에 든다.
- 생성 품질만 성공:
  (a)는 통과했지만 (b)~(e) 중 하나를 못 넘는다. 연구 결과와 adapter는 보존하되 제출은 교체하지 않는다.
- 기각:
  (a)를 못 넘거나 hard/format/invalid/hit-max guardrail을 위반한다. 기존 C-1/T10e를 유지한다.
- SFT와 DPO를 holdout에서 둘 다 돌려 더 좋은 것을 사후 선택하지 않는다. 5단계에서 고른 하나만 평가한다.
- public leaderboard 점수로 teacher, checkpoint, threshold, DPO pair, filter를 재조정하지 않는다.

[채택 뒤 leaderboard 처리]
채택일 때만 동결 후보+C prompt+C-1 filter+k32로
data/deep_chal_math_leaderboard_filtered.csv의 논리 행 831개를 같은 순서로 추론해 별도 artifact에 쓴다.
라벨은 없으므로 accuracy를 계산하지 않는다. 기존 t10a_c1_filtered_k32와 답이 바뀐 ID만 audit에 남긴다.
root submission.csv 교체는 채택 판정과 manifest가 모두 기록된 뒤 한 번만 한다.
원본 1,000행 전체 최종 테스트 리허설과 백업은 T13에서 수행한다.

[실행 순서]
1. configs/t11_aimo_generation_quality.json의 teacher·예산·prompt/config hash를 채우고 freeze한다.
2. scripts/run_t11.sh preflight-data: 경계·오염·ID 교집합 검사.
3. scripts/run_t11.sh probe: C-style k8 difficulty probe와 hard/anchor 목록 생성.
4. scripts/run_t11.sh teacher-preflight: 64문항 gate. 실패하면 즉시 종료.
5. scripts/run_t11.sh build-data: full teacher generation, SFT/DPO 데이터와 audit 생성.
6. scripts/run_t11.sh train-sft: LR/checkpoint sweep와 validation sample accuracy 산출.
7. scripts/run_t11.sh train-dpo: pair gate 통과 시에만 DPO와 validation 평가.
8. scripts/run_t11.sh freeze-candidate: holdout 전 단일 checkpoint 동결.
9. scripts/run_t11.sh evaluate-holdout: k32 한 후보 평가와 사전 등록 판정.
10. 채택일 때만 scripts/run_t11.sh build-leaderboard-submission.

[범위 밖]
- k64 이상, 새 prompt 다양화, 새 vote-weight/filter 탐색, T10e에 네 번째 arm을 사후 추가하는 일
- test-time Python/SymPy/TIR/solver/verifier, 외부 API, retrieval
- teacher에게 정답을 주는 hint-conditioned 풀이, holdout/leaderboard/test 문제 전송
- 외부 CoT를 무검증으로 섞기, answer-only SFT, packing=True, NF4 학습
- DPO 결과를 본 뒤 KTO/GRPO/PPO로 바꾸거나 hyperparameter 격자를 늘리는 일

[완료 조건]
- 보호 ID·full leaderboard 1,000행과의 오염 교집합이 0이고 audit이 남았다.
- teacher preflight 통과/실패가 수치·비용·model revision과 함께 기록됐다.
- 모든 SFT row가 정답·완결·2048 미만·무상충·text-only 조건을 만족하고 hard 비중이 80% 이상이다.
- DPO를 했다면 pair의 75% 이상이 correct-vs-wrong이고 length-only pair는 25% 이하다.
- validation sample accuracy로 후보 하나를 동결한 뒤에만 holdout을 열었다.
- raw pass@1 primary와 filtered majority@32 secondary, 4 split guardrail, 길이·종료·runtime 지표가 남았다.
- 사전 등록 규칙으로 채택/생성 품질만 성공/기각을 결정했고 T13이 그 결과를 참조한다.
- 실행 뒤 발표 자료용 누적 기록표에 T11 행과 AIMO 대비 축소·금지 요소를 추가했다.

[산출물]
configs/t11_aimo_generation_quality.json
scripts/run_t11.sh
src/build_t11_hard_cot.py
src/build_t11_dpo.py
src/train_dpo.py
data/t11_aimo_generation_quality/
  eligible_ids.txt
  student_probe.jsonl
  difficulty-audit.json
  hard_ids.txt
  anchor_ids.txt
  teacher_generations.jsonl
  trace-audit.jsonl
  sft_train.jsonl
  dpo_train.jsonl
  contamination-audit.csv
  manifest.json
artifacts/t11_aimo_generation_quality/
  teacher-preflight.json
  adapters/sft/
  adapters/dpo/
  sft-hp-sweep.json
  dpo-checkpoint-curve.json
  validation-comparison.json
  frozen-candidate.json
  holdout/generations.jsonl
  holdout/raw-sample-quality.json
  holdout/filtered-comparison.json
  final_config.json
  comparison.md
  manifest.json
  tests.xml
artifacts/submissions/t11_c1_filtered_k32/  (채택일 때만)
  submission.csv
  submission-audit.json
  diff-vs-t10a-c1.json
  manifest.json
```

---

## T11b — DeepSeek-14B 4-bit teacher + 결정적 최종 줄 정규화 preflight

```text
[왜 지금 하는가 — T11 실패 뒤 추가]
T11은 teacher가 수학을 전혀 못 풀어서 실패한 것이 아니다. 고정된 첫 64문항×4=256개 출력에서
품질 필터 전 extracted-correct는 76개였지만, 마지막 non-empty line의 FINAL_ANSWER 계약을
256/256개가 위반했다. 라벨을 보지 않고 추출 답을 canonical final line으로 붙이는 반사실 재집계에서는
나머지 품질 조건을 통과한 정답 trace가 71개, 정답 trace가 하나 이상인 문항이 30/64개였다.
따라서 trace 수 기준 64개는 넘지만 문항 수 기준 32개에 2문항 모자랐다.

이번 작업은 이 두 원인만 고친다.
1. teacher를 `DeepSeek-R1-Distill-Qwen-14B` 4-bit로 한 단계 강화한다.
2. gold answer를 읽지 않는 결정적 정규화기로 마지막 줄 형식을 수리한다.

이 작업의 범위는 **동일 64문항 teacher preflight 재검증까지**다. gate를 통과해도 full teacher 생성,
SFT, DPO, validation, holdout, leaderboard 생성은 실행하지 않는다. 통과 뒤 전체 T11을 재개하려면
별도 사전 등록 작업을 먼저 작성한다.

[핵심 판정 질문]
같은 hard 64문항과 같은 문항당 4회 예산에서, DeepSeek-14B의 raw 풀이를 label-blind final-line
normalizer로 정규화했을 때 T11의 원래 teacher gate를 모두 통과하는가?

[의도적으로 바꾸는 것과 고정하는 것]
바꾸는 것:
- teacher model/revision
- 24GB GPU 적재를 위한 고정 4-bit in-flight quantization
- raw 생성 뒤, 라벨을 열기 전에 적용하는 결정적 final-line normalizer

고정하는 것:
- preflight ID 64개와 순서
- 문항당 4개, sample_index 0..3, seed=62000
- T11 teacher system/user prompt의 정확한 UTF-8 바이트와 hash
- temperature=0.7, top_p=0.95, max_input_tokens=2048, max_new_tokens=2048
- 기존 trace 품질 필터, 최소 assistant token 128, code/tool 금지 규칙
- gate: 정답 trace가 있는 문항 >=32/64, accepted correct trace >=64/256
- API 비용 $0, preflight 2시간 및 projected full run 12시간 상한

프롬프트, sampling, ID, gate, trace 길이, 필터를 추가로 바꾸거나 여러 arm을 비교하지 않는다.
4-bit는 14B를 단일 24GB GPU에 적재하기 위한 실행 조건이지 탐색 arm이 아니다.

[재사용 입력 — 실행 전에 hash 확인]
기존 T11 산출물은 읽기 전용으로 재사용하고 절대 수정하거나 덮어쓰지 않는다.
- data/t11_aimo_generation_quality/teacher_preflight_ids.txt
  SHA-256 2856ece38200221d3abd29525ebb8dfa0516a1fc03cb7ae00d17375bf3c50dd9
- data/t11_aimo_generation_quality/hard_ids.txt
  SHA-256 77a10c579e870e3cd912c03fbb19d4635d6b1e826c7bdbae7b1c30b631441abb
- data/t11_aimo_generation_quality/student_probe.jsonl
  SHA-256 31b2af19392bbe86089ddfc27450d75af2aac27fba565dd633c9254ef3c63a4c
- data/t11_aimo_generation_quality/teacher_generations.jsonl
- artifacts/t11_aimo_generation_quality/teacher-preflight.json
  SHA-256 59484f387c30336242592f310b146404c2268171bb539f98d9a3eb4135616606
- configs/t11_aimo_generation_quality.json
  SHA-256 2584181f0a24c4cc4a57c4f0760fca0ba5ee55405250500c4526214b9ab765ba

해시가 하나라도 다르면 실행하지 말고 actual hash와 차이를 보고한다. student difficulty probe와
오염 검사는 재실행하지 않는다. 이미 보호 집합과 full leaderboard 1,000행 대조를 통과해 고정된
동일 64개 ID만 사용한다.

[teacher 신원과 적재 계약 — 변경 금지]
- provider: local_vllm
- model/tokenizer: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
- model/tokenizer revision: 1df8507178afcc1bef68cd8c393f61a886323761
- license: MIT
- trust_remote_code=false
- vLLM==0.27.1, bitsandbytes==0.50.1, dtype=bfloat16
- quantization=bitsandbytes, load_format=bitsandbytes (in-flight 4-bit)
- gpu_memory_utilization=0.90, max_model_len=4096
- max_num_seqs=16, request_chunk_size=8, enable_prefix_caching=true
- tool_use=false, tensor_parallel_size=1

모델 snapshot을 위 revision으로 받은 뒤 generation은 offline mode로 수행한다. 실제 resolved commit,
tokenizer commit, quantization/load arguments, GPU peak allocated/reserved VRAM, load wall time을 metadata에
기록한다. load-only smoke가 OOM이거나 이 revision이 로드되지 않으면 `teacher_load_failed`로 끝낸다.
다른 quant checkpoint, CPU offload, 다른 teacher, 외부 API로 폴백하지 않는다.

[teacher prompt·생성 계약 — T11과 바이트 동일]
- system_prompt SHA-256:
  c81d7fce66ab95f3a8ce549668332e0d226db93e85265e8a99027042eba83593
- user_prompt SHA-256:
  1cc2e308e223d03d222c45448e9a2f53c3aa43a67b0832a44ab89a3c866028f1
- combined prompt SHA-256:
  1b1f80807f63b2d0e6f748a3c54e7423237da07ee354ebccd2773fb4f1a16521
- do_sample=true, temperature=0.7, top_p=0.95
- max_input_tokens=2048, max_new_tokens=2048
- seed=62000, sample_index=0..3, 총 256개

DeepSeek 전용 prompt 변형, `<think>` 강제 prefix, few-shot 예시, 정답 힌트, 길이 지시 추가를 하지 않는다.
이 작업은 model+normalizer 조합만 검증하며 prompt 최적화 실험으로 확장하지 않는다.

[결정적 final-line normalizer 계약]
normalizer는 함수 수준에서 `raw_generation: str`만 입력받는다. question, id, gold answer, label CSV,
student probe 점수, 다른 sample의 답을 입력받지 않는다. 변환은 다음 순서 하나로 고정한다.

1. raw text에서 src.extract와 동일한 표기 정규화를 사용해 명시적 정수 후보를 읽는다.
   허용 source는 final_answer_marker, boxed, standalone_last_line뿐이다.
2. FINAL_ANSWER와 boxed를 포함한 명시적 후보가 둘 이상이면, 정규화 뒤 서로 다른 값이 정확히 1개일
   때만 계속한다. 서로 다른 값이 2개 이상이면 `conflicting_explicit_answers`로 변환하지 않는다.
3. 허용 source에서 canonical integer 하나를 얻지 못하면 `no_safe_integer_candidate`로 변환하지 않는다.
   generic last_integer 폴백으로 본문 숫자를 답으로 승격하지 않는다.
4. raw의 마지막 non-empty line이 이미 정확히 `FINAL_ANSWER: <canonical integer>`이면 본문을 바꾸지 않는다.
5. 아니면 raw의 trailing whitespace만 제거하고 `\nFINAL_ANSWER: <canonical integer>\n`을 덧붙인다.
   기존 풀이, boxed 표현, 숫자, 문장, 순서는 삭제·수정하지 않는다.
6. 산술, 수식 평가, 반올림, 분수/소수 변환, 여러 숫자 조합은 금지한다.

각 행에 raw_generation, normalized_generation, raw/normalized SHA-256, normalization_status,
candidate_source, canonical_candidate를 함께 보존한다. 정규화가 실패한 행도 삭제하지 않는다.
normalizer 출력 256개를 디스크에 원자적으로 쓴 뒤 파일 SHA-256을 freeze하고, 그 뒤에만 gold label을
로드해 correctness와 gate를 계산한다.

[정규화기 필수 테스트]
- boxed 정수, 음수, 0, -0, 천단위 콤마, 유니코드 마이너스, 전각 숫자
- 이미 올바른 FINAL_ANSWER 마지막 줄은 바이트 동일
- 같은 값의 boxed+FINAL_ANSWER는 허용
- 서로 다른 boxed+FINAL_ANSWER는 거부
- 소수, 분수, 수식, 본문 last_integer-only는 거부
- code fence/tool 표현은 normalizer가 지우거나 고치지 않음
- 빈 문자열, 잘린 출력, trailing whitespace
- label 파일을 주지 않아도 실행되며 label 경로/answer 필드를 전달하면 CLI가 거부
- 같은 입력을 두 번 처리했을 때 normalized JSONL SHA-256이 동일

[품질 필터 — normalized text에 적용]
정답 여부 외 조건은 기존 T11과 같다.
- raw finish_reason가 length/max_tokens가 아니고 hit_max_new_tokens=false
- teacher tokenizer로 다시 센 normalized assistant token이 128 이상 2048 미만
- normalized 마지막 non-empty line이 정확히 ^FINAL_ANSWER: -?(?:0|[1-9][0-9]*)$
- normalized text의 explicit candidate가 정규화 뒤 서로 다른 값 정확히 1개
- code fence, tool call, Python/SymPy/import/def/exec 표현이 없음

위 품질 판정을 먼저 동결하고 마지막에 gold answer exact match를 결합해 accepted correct를 계산한다.
normalizer는 finish_reason, hit-max, code/tool, explicit conflict를 수리하지 않는다.

[실행 순서]
1. T11 재사용 파일의 경로·행 수·SHA-256을 검증한다.
2. T11b 전용 config, script, normalizer와 단위 테스트를 작성한다. 원래 T11 teacher 상수를 임의 model을
   허용하도록 전역 완화하지 말고 T11b 경로에서만 새 고정 신원을 허용한다.
3. frozen T11 raw 256개에 normalizer-only replay를 수행한다. 기존 반사실 기준인 accepted correct
   71개, 정답 trace 보유 30문항과 재현 여부를 기록한다. 불일치하면 trace별 diff를 남기고 구현 오류를
   먼저 해결하며, 수치를 맞추기 위해 규칙을 사후 변경하지 않는다.
4. DeepSeek-14B 4-bit load-only smoke와 고정 4문항×1 generation smoke를 수행한다. smoke 출력은 gate에
   포함하지 않고 prompt/model/quantization provenance, OOM, token count, raw 보존만 확인한다.
5. 고정 64문항×4를 한 번 생성해 raw JSONL 256개를 먼저 freeze한다. 기존 T11 JSONL에 append하지 않는다.
6. label-free normalizer로 별도 normalized JSONL을 만들고 SHA-256을 freeze한다.
7. 그 뒤에만 canonical train의 answer를 열어 raw 지표, normalized 품질 지표, accepted-correct 지표와
   문항별 4개 결과를 계산한다.
8. 아래 gate로 `teacher_gate_passed` 또는 `teacher_gate_failed`를 기록하고 무조건 종료한다.

[원래 gate — 임계값 변경 금지]
모두 만족할 때만 `teacher_gate_passed`다.
- 64문항 중 accepted correct trace가 최소 1개인 문항 >=32
- 256개 중 accepted correct trace >=64
- code/tool 의존 trace 0개
- 보호 ID/leaderboard/test 전송 0건
- API 비용 $0
- preflight wall time <=2시간
- 실측 처리량으로 계산한 hard 1,883문항×최대 8회 worst-case full generation <=12시간

raw와 normalized 기준을 모두 보고하되 판정에는 사전 등록된 normalized 기준을 쓴다. teacher가 더 강해도
gate를 하나라도 못 넘으면 실패다. 30/64처럼 근소하게 미달해도 sample을 더 뽑거나 threshold를 낮추지 않는다.

[판정 뒤 행동]
- passed: T11b는 성공으로 기록하되 다음 행동은 `stop_before_full_teacher_generation`이다.
  full run용 새 사전 등록 프롬프트를 작성하기 전에는 T11 build-data/train-sft/train-dpo를 호출하지 않는다.
- failed: T11b를 종료하고 기존 T10a C-1/T10e를 유지한다. Qwen3/DeepSeek-32B/Gemini/API나 추가 k로
  자동 폴백하지 않는다.
- 어느 경우에도 holdout 또는 leaderboard를 열거나 submission.csv를 바꾸지 않는다.

[비파괴 경로]
기존 configs/t11_aimo_generation_quality.json, scripts/run_t11.sh,
data/t11_aimo_generation_quality/**, artifacts/t11_aimo_generation_quality/**를 수정하지 않는다.

새 경로:
- configs/t11b_deepseek14b_teacher_preflight.json
- scripts/run_t11b.sh
- src/normalize_teacher_trace.py
- tests/test_normalize_teacher_trace.py
- data/t11b_deepseek14b_teacher_preflight/
- artifacts/t11b_deepseek14b_teacher_preflight/

[완료 조건]
- 기존 T11 입력·산출물 hash가 실행 전후 동일하다.
- teacher model/tokenizer가 지정 revision으로 로드됐고 4-bit engine arguments와 VRAM이 기록됐다.
- raw 256개와 normalized 256개가 별도 보존되고 각각 SHA-256이 있다.
- normalizer가 label-blind·무산술·결정적임을 단위 테스트와 2회 hash 일치로 확인했다.
- raw/normalized final-line 준수율, extracted-correct, accepted-quality, accepted-correct,
  정답 trace 보유 문항 수, hit-max, code/tool, explicit conflict, token 분포가 모두 기록됐다.
- 원래 gate로 판정한 뒤 full teacher/SFT/DPO/validation/holdout/leaderboard 생성 0건에서 종료했다.
- 실행했다면 발표 자료용 누적 기록표에 T11b 행을 추가했다. 문서 작성만 한 현재 단계에는 추가하지 않는다.

[산출물]
configs/t11b_deepseek14b_teacher_preflight.json
scripts/run_t11b.sh
src/normalize_teacher_trace.py
tests/test_normalize_teacher_trace.py
data/t11b_deepseek14b_teacher_preflight/
  smoke_ids.txt
  raw_teacher_generations.jsonl
  normalized_teacher_generations.jsonl
  normalization-audit.jsonl
  historical-normalizer-replay.json
artifacts/t11b_deepseek14b_teacher_preflight/
  load-smoke.json
  teacher-run-metadata.json
  teacher-preflight.json
  comparison-vs-t11.json
  manifest.json
  tests.xml
```

---

## T11c — Qwen2.5-Math-7B 복구형 teacher preflight

근거 원문은 [Qwen2.5-Math 공식 저장소](https://github.com/QwenLM/Qwen2.5-MATH),
[Qwen2.5-Math-7B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct),
[AIMO-2 우승 보고서](https://arxiv.org/abs/2504.16891)와
[대회 데이터 사용 조건](../information/data.md#데이터-사용-조건)이다. Qwen 공식 저장소는 CoT system
prompt를 `Please reason step by step, and put your final answer within \boxed{}.`로 제시하고,
Qwen2.5-Math-Instruct의 maj@8/RM@8 sampling에는 temperature=0.7, top_p=0.8을 사용한다.

```text
[왜 T11b 뒤에 하는가]
T11의 local Qwen2.5-Math-7B teacher는 수학을 못 풀어서 실패한 것이 아니다. 고정 64문항×4에서
extracted-correct는 76/256(29.6875%)이었고 정답이 한 번 이상 나온 문항은 31/64였다. 원래 gate는
256/256개의 FINAL_ANSWER 마지막 줄 위반과 code/tool 7개 때문에 accepted correct를 0개로 처리했다.
그러나 T11b에서 추가한 label-blind normalizer를 T11 raw에 결정적으로 replay하면 accepted correct는
71/256, 정답 trace 보유 문항은 30/64가 된다. trace 수 기준 64개는 통과하고 문항 기준 32개에 단 2개
부족하다. 같은 조건의 DeepSeek-14B는 normalized accepted correct 23개·12문항, projected full 28.51시간으로
더 멀리 실패했으므로 더 큰 teacher를 찾기보다 Qwen7B의 실행 계약을 한 번만 수리한다.

Qwen raw의 sample별 accepted-correct 누적 문항은 19→27→29→30으로 포화 방향이었다. 따라서 무조건
k를 늘리는 것을 첫 레버로 쓰지 않는다. 먼저 공식 CoT prompt/top_p, 3,072-token completion cap,
질문·sample별 독립 seed, label-blind normalizer를 적용하고, 첫 4개에서 accepted correct가 전혀 없는
문항에만 sample 4..7을 추가한다.

[관측된 실행 결함]
- T11 raw 42/256이 2,048 tokens에서 length 종료됐고, 22/64문항에 length 출력이 최소 1개 있었다.
- 첫 4개에서 정답이 0개인 33문항 중 14문항에 length 출력이 있었다. 새 정답 문항은 2개만 필요하다.
- code/tool 7개는 실제 도구 풀이가 아니라 모두 sample_index=0의 length 출력 끝에 거의 동일한 다국어
  깨진 suffix와 PYTHON token이 붙은 상관 퇴화였다. 이 때문에 전역 seed 하나로 n=4를 생성하지 않는다.
- 원래 projected full 2.49시간은 12시간 상한보다 충분히 작다. 3,072-token 상한과 조건부 k=8을 실제
  preflight 처리량으로 다시 투영하되, 결과를 보고 cap이나 k를 더 늘리지 않는다.

[핵심 가설]
Qwen2.5-Math-7B-Instruct를 BF16 그대로 유지하면서 공식 CoT prompt와 temperature=0.7/top_p=0.8을
사용하고, 2,048에서 잘린 풀이에 3,072 tokens까지 허용하며, 요청마다 독립 seed를 주면 깨진 tail과
미완성 풀이가 줄어든다. label-blind normalizer로 유일한 explicit integer를 FINAL_ANSWER 마지막 줄로만
정규화하고 첫 라운드 미해결 문항에만 4개를 추가하면, 12시간 안에 원래 품질 수치 gate를 넘을 수 있다.

[범위와 강제 정지]
범위는 기존 결과를 보지 않은 새로운 hard 64문항의 local teacher preflight 1회뿐이다.
- 1차: 64문항×sample_index 0..3 = 256개
- 2차: 1차 accepted-correct가 0개인 문항만 sample_index 4..7 = 문항당 추가 4개
- 전체 raw 범위: 최소 256개, 최대 512개

gate를 통과해도 full hard teacher generation, SFT, DPO, validation, holdout, leaderboard, submission은
실행하지 않는다. `teacher_gate_passed`와 `stop_before_full_teacher_generation`을 기록하고 끝낸다.

[T11c에서 바꾸는 것]
- preflight ID: 이미 라벨과 결과를 본 첫 64문항 대신 hard_ids의 다음 64문항
- prompt: Qwen 공식 CoT system prompt + 질문 원문만 있는 user message
- top_p: 0.95 → 0.8; temperature=0.7은 유지
- max_new_tokens: 2,048 → 3,072; max_model_len: 4,096 → 5,120
- 생성: 문항당 n=4 전역 seed 대신 각 (id, sample_index)의 n=1 독립 결정 seed
- T11b에서 검증한 label-blind-final-line-v1 의미론을 처음부터 적용
- 1차 미해결 문항만 고정 sample_index 4..7을 생성하는 조건부 2차
- accepted trace는 normalized assistant 3,072 tokens 미만이면서 student SFT 전체 sequence가 4,096
  tokens 이하여야 한다. SFT에서 잘라 쓰는 trace는 만들지 않는다.

[계속 고정하는 것]
- teacher model/tokenizer revision, BF16, local vLLM, text-only CoT, tool_use=false
- hard 판정과 순서, 보호 ID, 오염 검사 결과, student/base/prompt C와 최종 k=32 경로
- gold를 teacher prompt에 넣지 않고 raw와 label-blind audit를 먼저 freeze하는 순서
- canonical integer exact match만 정답으로 인정; LLM judge·수식 동치·tolerance·pseudo-label 없음
- final 수치 gate: 정답 trace 보유 문항 >=32/64, accepted correct trace >=64
- preflight <=2시간, projected worst-case full <=12시간, API 비용 $0
- holdout·validation·full leaderboard·test·submission 접근 0건

모델, prompt bytes, sampling, ID, seed 공식, normalizer, token 기준, 두 라운드 규칙과 gate를 config hash가
동결된 뒤 바꾸지 않는다. 기존 64문항 결과를 새 설정 선택이나 통과 판정에 사용하지 않는다.

[재사용 입력과 새 64문항 — 생성 전에 hash 확인]
다음 파일은 읽기 전용으로 재사용한다.
- data/t11_aimo_generation_quality/hard_ids.txt
  SHA-256 77a10c579e870e3cd912c03fbb19d4635d6b1e826c7bdbae7b1c30b631441abb, 1,883행
- data/t11_aimo_generation_quality/teacher_preflight_ids.txt
  SHA-256 2856ece38200221d3abd29525ebb8dfa0516a1fc03cb7ae00d17375bf3c50dd9, 첫 64행
- data/t11_aimo_generation_quality/student_probe.jsonl
  SHA-256 31b2af19392bbe86089ddfc27450d75af2aac27fba565dd633c9254ef3c63a4c
- data/t11_aimo_generation_quality/teacher_generations.jsonl
  SHA-256 764f5e8fcc31bc61303ee8ed76e23b653a6b220effeb525f375aa78df9099e3c
- configs/t11_aimo_generation_quality.json
  SHA-256 2584181f0a24c4cc4a57c4f0760fca0ba5ee55405250500c4526214b9ab765ba
- data/t11b_deepseek14b_teacher_preflight/historical-normalizer-replay.json
  SHA-256 1fc1e332dde093fb73909b078eed2f845c2378d48c8aec97791b92c7bbfb9d3d

새 preflight ID는 hard_ids의 zero-based slice [64:128], 즉 원본 순서의 65~128번째 줄을 답 column을
읽지 않고 그대로 쓴다. 예상 계약은 다음과 같다.
- 64행, unique 64개
- 첫 T11 preflight 64개와 교집합 0
- holdout_union, rft_validation_500, suspect_set과 교집합 0
- 첫 ID train-000045, 마지막 ID train-001696
- trailing newline 포함 파일 SHA-256:
  a3f26bbe1fd1f692f1fb695ca73d161f938a112008fc4265014a4c1847114655

`data/t11c_qwen7b_repaired_teacher_preflight/preflight_ids.txt`로 원자적으로 쓴 뒤 위 조건과 hash가 하나라도
다르면 `input_identity_failed`로 종료한다. difficulty probe, contamination audit, hard 정렬을 재실행하거나
다른 64문항을 고르지 않는다.

[teacher 신원 — 변경 금지]
- provider: local_vllm
- model: Qwen/Qwen2.5-Math-7B-Instruct
- model revision: ef9926d75ab1d54532f6a30dd5e760355eb9aa4d
- tokenizer revision: ef9926d75ab1d54532f6a30dd5e760355eb9aa4d
- license: Apache-2.0
- dtype: bfloat16
- quantization/load_format: 없음/auto
- tool_use: false
- cache: /workspace/.hf_home/hub

load-only smoke에서 resolved model/tokenizer commit이 정확히 일치하지 않거나 BF16 load가 OOM이면
`teacher_load_failed`로 끝낸다. 4-bit, CPU offload, 다른 revision/model, 외부 API로 폴백하지 않는다.

[teacher prompt — UTF-8 바이트 고정]
system message는 아래 한 줄이며 trailing newline이 없다.

Please reason step by step, and put your final answer within \boxed{}.

- system prompt bytes: 70
- system prompt SHA-256:
  1ac42a4db949361d680bccf674bfe78603b409d1c10f77fce293f14b9e14cc1e
- user message는 질문 원문과 바이트가 같은 `{question}`뿐이다.
- user template SHA-256:
  bf085a6e12c9d0e23a9dd157df084f933b2ef021caba82def1494bfb84a723c9

`Problem:` prefix, FINAL_ANSWER 지시, 별도 no-tool 문장, few-shot, 답 힌트, 유형 tag, 이전 sample,
student 풀이, self-critique를 추가하지 않는다. Qwen 공식 CoT prompt 자체가 tool-integrated reasoning prompt와
구분되며, code/tool output은 아래 필터에서 전부 거부한다.

[generation 계약]
  engine: vllm
  dtype: bfloat16
  temperature: 0.7
  top_p: 0.8
  max_input_tokens: 2048
  max_new_tokens: 3072
  max_model_len: 5120
  gpu_memory_utilization: 0.92
  max_num_seqs: 64
  request_chunk_size: 16
  enable_prefix_caching: true
  samples_first_round: 4
  samples_second_round: 4
  samples_max: 8
  n_per_logical_request: 1

top_k, min_p, repetition/frequency/presence penalty, stop string, beam search, guided decoding을 추가하지 않는다.
system+user chat template을 적용한 입력이 2,048 tokens를 넘으면 자르지 않고 `input_too_long`으로 전체
preflight를 중단한다.

[독립 seed 계약]
Python built-in hash나 실행 순서에 의존하지 않는다. 각 logical request의 seed는 다음 바이트 공식으로 만든다.

  namespace = b"t11c-qwen7b-repair-v1"
  material = namespace + b"\0" + id.encode("ascii") + b"\0" + str(sample_index).encode("ascii")
  child_seed = int.from_bytes(SHA256(material).digest()[:4], "big") & 0x7fffffff

- namespace SHA-256:
  ccc82aa8a72fb259b6629dc5e0a4410bf3033c6a3cf346a82cfc9be676f74474
- 새 64문항×sample_index 0..7의 512개 예정 seed는 unique 512개여야 한다.
- 각 prompt는 n=1 SamplingParams와 자기 child_seed를 가진다. 한 SamplingParams(seed=62000, n=4)를
  여러 질문에 공유하지 않는다.
- label 없는 `planned-seed-manifest.jsonl`에 64문항×sample_index 0..7의 id, sample_index,
  question/prompt hash, child_seed, sampling과 model revision을 먼저 기록하고 freeze한다. 실제 1차·2차
  request manifest는 이 512행 계획의 순서 보존 부분집합이어야 한다.
- standalone 1개와 batch 안의 같은 logical request가 batch-invariant mode에서 byte-identical인지 synthetic
  smoke로 확인한다. 다르면 `seed_or_batch_invariance_failed`로 끝내고 실제 64문항을 생성하지 않는다.

[1차 raw와 label-blind normalization]
64×4를 생성한 뒤 raw JSONL을 먼저 원자적으로 쓰고 SHA-256을 freeze한다. 각 row는 id, sample_index,
child_seed, raw_generation, token ids/count, finish_reason, input truncation, model/tokenizer revision과 prompt hash를
가진다. raw를 쓴 뒤 수정하거나 덮어쓰지 않는다.

그 다음 T11b에서 검증한 label-blind-final-line-v1 의미론을 적용한다.
1. raw의 FINAL_ANSWER marker, \boxed{}, canonical standalone last line에서 canonical integer 후보를 찾는다.
2. 서로 다른 explicit canonical integer가 정확히 1개일 때만 그 값을 candidate로 삼는다.
3. 마지막 non-empty line이 이미 정확한 FINAL_ANSWER면 byte 그대로 유지한다.
4. 그렇지 않으면 raw.rstrip() 뒤에 정확히 `\n\nFINAL_ANSWER: <candidate>\n`을 붙인다.
5. 후보 없음, non-integer boxed answer, 서로 다른 explicit answer 충돌이면 내용을 고치지 않고 reject한다.
6. last-integer fallback, 산술 계산, gold answer, LLM judge는 normalizer에 제공하지 않는다.

raw hash와 label-blind audit/normalized hash가 freeze되기 전에는 canonical answer column을 로드하지 않는다.
같은 입력을 두 번 normalize한 결과가 byte-identical이어야 한다.

[label-blind 품질 필터]
normalized trace는 다음을 모두 만족해야 accepted_quality다.
- finish_reason가 stop/eos이고 length/max_tokens가 아님
- input truncation 0, raw output_tokens <3072, normalized assistant tokens가 128 이상 3072 미만
- frozen Qwen2.5-3B student tokenizer 기준 prompt+assistant+special token 전체가 4096 이하
- 마지막 non-empty line이 정확히 ^FINAL_ANSWER: -?(?:0|[1-9][0-9]*)$
- explicit canonical integer의 서로 다른 값이 정확히 1개
- 마지막 줄 앞에 비어 있지 않은 self-contained 수학 풀이가 있음
- 기존 T11 CODE_OR_TOOL_RE 기준 code fence, Python, SymPy, import/from/def/exec, calculator/solver,
  tool-call/TIR 흔적이 없음

hit-max, code/tool, explicit conflict, no-safe-candidate, normalized token overflow와 깨진/반복 tail을 raw 전체와
accepted subset 기준으로 각각 보고한다. code/tool trace는 accepted dataset에 0개여야 하며, raw code/tool이
하나라도 있으면 원래 T11 gate와 동일하게 전체 teacher gate를 실패시킨다.

[1차 gold 결합과 조건부 2차]
1차 raw와 label-blind 산출물 hash를 freeze한 뒤에만 새 64문항의 canonical integer gold를 로드한다.

  correct = extracted_answer == canonical_gold
  accepted_correct = accepted_quality AND correct

문항별 sample_index 0..3에서 accepted_correct가 0개인 ID만 원본 64문항 순서로
`second_round_ids.txt`에 쓴다. 이 파일과 selection audit를 freeze한 뒤, generation process에는 ID와 질문만
전달하고 gold는 전달하지 않는다. 해당 ID에 sample_index 4..7을 같은 prompt/sampling/seed 공식으로 정확히
4개씩 생성한다. content·길이·정답을 이유로 sample을 교체하거나 8개를 넘기지 않는다.

2차 raw도 먼저 freeze하고 같은 label-blind normalizer/품질 필터를 독립 실행한 뒤에만 gold를 결합한다.
최종 판정은 1·2차 합집합을 사용하되 first-round-only 지표와 second-round marginal gain을 별도로 보고한다.

[비용·처리량 계산]
- API 호출과 외부 문제 전송은 0이므로 비용은 $0이다.
- model load, 1차 generation, 1차 normalization/gold join, 2차 generation, 최종 audit 시간을 분리한다.
- first-round zero-accepted-question 비율 z와 실측 1·2차 처리량으로 full hard 1,883문항의 expected schedule
  `1883×4 + round(1883×z)×4`를 투영한다.
- 동시에 worst-case 1,883×8=15,064 generations의 wall time을 보수적으로 계산한다.
- 모델 load와 모든 CPU 후처리를 포함한 preflight <=2시간, projected worst-case full <=12시간이어야 한다.
  cap을 4,096으로 올리거나 k>8로 늘려 시간 gate를 다시 맞추지 않는다.

[실행 순서]
1. 재사용 파일의 path/row/hash, 새 slice hash, 보호 ID·기존 preflight 교집합 0을 검증한다.
2. T11c config, local runner, normalizer wrapper와 테스트를 작성하고 source/config hash를 freeze한다.
3. exact revision BF16 load-only smoke와 per-request seed·standalone/batch invariance smoke를 통과한다.
4. label 없는 planned seed manifest 64×8과 first-round request manifest 64×4를 freeze한다.
5. 1차 raw 256개를 생성·freeze하고 label-blind normalization을 두 번 재현해 freeze한다.
6. 그 뒤에만 gold를 결합해 second_round_ids를 동결한다.
7. 미해결 ID에만 sample 4..7을 생성하고 raw→label-blind freeze→gold join 순서를 반복한다.
8. T11 raw, T11 normalizer replay, T11b와 first/final T11c 지표·runtime을 비교한다.
9. 아래 gate로 판정하고 성공·실패와 무관하게 downstream 0건에서 종료한다.

[teacher gate — 모두 만족해야 pass]
- 새 preflight ID 정확히 64개, 기존 T11 preflight/protected ID와 교집합 0
- first round는 모든 문항 sample_index 0..3 정확히 256개
- second round는 first-round accepted-correct 0개 문항에만 sample_index 4..7 정확히 4개씩
- duplicate/missing logical request 0, planned seed collision 0, input truncation 0
- 최종 64문항 중 accepted correct trace가 최소 1개인 문항 >=32
- 1·2차 전체 accepted correct trace >=64
- raw code/tool/TIR trace = 0, accepted code/tool/TIR trace = 0
- label-blind normalizer 두 번 실행 결과 byte-identical
- preflight wall <=2시간, projected worst-case full wall <=12시간, API cost=$0
- teacher prompt에 gold answer 포함 0, protected/validation/leaderboard/test 전송·생성 0

문항 31개, trace 63개, raw code/tool 한 개처럼 근소하게 실패해도 threshold, prompt, top_p, cap, seed, k를
수정하거나 같은 64문항을 다시 생성하지 않는다.

[판정 뒤 행동]
- passed: `teacher_gate_passed`와 `stop_before_full_teacher_generation`을 기록한다. full teacher/SFT/DPO를
  진행하려면 T11c 결과와 실제 z/runtime을 입력으로 쓰는 별도 사전 등록 작업을 먼저 작성한다.
- failed: `teacher_gate_failed`로 T11을 종료하고 public leaderboard 최고인 T10a C-1 filtered k32를 유지한다.
- 어느 경우에도 이 T11c 안에서 DeepSeek, Qwen-72B, reward model, TIR, Python, API teacher로 전환하지 않는다.
- 기존 T11/T11b data/artifacts와 root submission.csv는 읽기 전용이며 수정하지 않는다.

[비파괴 새 경로]
- configs/t11c_qwen7b_repaired_teacher_preflight.json
- configs/supervisor_t11c.conf
- scripts/run_t11c.sh
- scripts/supervisor_t11c.sh
- src/run_t11c_qwen_teacher.py
- tests/test_t11c_qwen_teacher.py
- data/t11c_qwen7b_repaired_teacher_preflight/
- artifacts/t11c_qwen7b_repaired_teacher_preflight/

[완료 조건]
- 새 64 IDs, 공식 prompt, model/tokenizer revision, sampling, seed 512개와 gate가 생성 전에 hash로 동결됐다.
- Qwen7B BF16 load와 독립-seed batch-invariance smoke가 통과했다.
- 1·2차 각각 raw→label-blind freeze→gold join 순서를 지켰고 normalizer가 결정적으로 재현됐다.
- first/final accepted-quality/correct, 정답 문항, 추출 경로, token/finish/code/conflict, marginal gain과 runtime이
  모두 기록됐다.
- expected/worst-case full projection이 12시간 상한과 함께 기록됐다.
- full teacher/SFT/DPO/validation/holdout/leaderboard/submission 생성은 모두 0건이다.
- 실행했다면 발표 자료용 누적 기록표에 T11c 행을 추가한다. 문서만 작성한 단계에는 추가하지 않는다.

[산출물]
configs/t11c_qwen7b_repaired_teacher_preflight.json
configs/supervisor_t11c.conf
scripts/run_t11c.sh
scripts/supervisor_t11c.sh
src/run_t11c_qwen_teacher.py
tests/test_t11c_qwen_teacher.py
data/t11c_qwen7b_repaired_teacher_preflight/
  preflight_ids.txt
  planned-seed-manifest.jsonl
  first-round-request-manifest.jsonl
  first-round-raw.jsonl
  first-round-normalized.jsonl
  first-round-label-blind-audit.jsonl
  second_round_ids.txt
  second-round-request-manifest.jsonl
  second-round-raw.jsonl
  second-round-normalized.jsonl
  second-round-label-blind-audit.jsonl
artifacts/t11c_qwen7b_repaired_teacher_preflight/
  input-verification.json
  load-and-seed-smoke.json
  first-round-labeled-audit.jsonl
  final-labeled-audit.jsonl
  teacher-run-metadata.json
  teacher-preflight.json
  comparison-vs-t11-t11b.json
  manifest.json
  tests.xml
```

---

## T11d — explicit 정수 추출·출력 계약 보강 + frozen replay

```text
[왜 T11c 뒤에 추가하는가 — 2026-08-27]
T11/T11b/T11c는 teacher의 수학 정답뿐 아니라 마지막 FINAL_ANSWER 계약 실패 때문에 usable trace가 크게
줄었다. 별도 T8 raw 진단에서도 합집합 3,737문항의 pass@32는 84.40%, plurality@32는 69.31%로
15.09pp 차이가 났다. 이 564개 selection failure 중 단순 동률은 29개뿐이라 semantic verifier 없이
전부 회수할 수는 없지만, 추출기 자체가 이미 맞게 계산한 답을 다른 본문 숫자로 바꾸는 사례는 먼저
제거해야 한다.

대표 회귀인 train-012155에서는 32개 출력 모두 계산상 400에 도달했다. `FINAL_ANSWER: 400`인 16개는
400으로 추출됐지만, `FINAL_ANSWER: $400.00`인 16개는 explicit integer parser가 거부한 뒤 본문의
마지막 정수 50으로 fallback했다. 그 결과 50=16, 400=16 동률이 되었고 first-generated tie-break가
50을 선택했다. 상세 근거는
`report/t8-pass-majority-diagnostic-2026-08-27/{report.html,summary.json,example_cases.json}`에 있다.

이 작업은 semantic verifier나 SFT를 대신하지 않는다. 목표는
(1) 정수와 표기상 동치인 explicit `.0...` 답을 계산 없이 안전하게 canonicalize하고,
(2) 손상된 explicit answer에서 무관한 본문 숫자로 후퇴하는 경로를 차단하며,
(3) 새 생성이 처음부터 canonical FINAL_ANSWER 마지막 줄을 쓰도록 prompt 계약을 강화하는 것이다.

[비파괴·누수 방지 계약]
- 기존 `artifacts/t8_self_consistency/`, T8-3, T10, T11/T11b/T11c artifact와 root submission.csv는
  읽기 전용이며 수정하지 않는다.
- extractor와 prompt 규칙은 question, ID, gold label, 다른 sample의 답, split, 문제 유형을 입력으로
  받지 않는다.
- 구현·단위 테스트·label-blind old/new prediction freeze가 끝나기 전에는 canonical answer를 로드하지 않는다.
- reused T8 3,737 결과를 본 뒤 허용 문법, fallback, threshold, tie-break를 다시 바꾸지 않는다.
- T8 replay는 이미 반복 사용한 holdout의 paired 진단이다. 이 결과만으로 최종 채택을 주장하지 않고,
  T13에 반영하려면 사전 고정된 fresh validation이 추가로 필요하다고 명시한다.
- 새 모델·adapter·학습·외부 API·인터넷·solver·Python/SymPy 계산 verifier는 사용하지 않는다.

[1. explicit integer-equivalent 추출 계약]
기준 구현은 `src/extract.py`다. 기존 `normalize_integer()`의 canonical label/submit 의미를 무심코 넓히지
말고, 필요하면 explicit occurrence 전용 문자열 normalizer를 별도로 둔다. FINAL_ANSWER와 boxed 안에서만
다음 표기를 notation-only normalization으로 허용한다.

  허용 입력                 canonical result
  FINAL_ANSWER: 400          400
  FINAL_ANSWER: +400         400
  FINAL_ANSWER: $400         400
  FINAL_ANSWER: $400.00      400
  FINAL_ANSWER: 1,234.000    1234
  FINAL_ANSWER: -0.0         0
  FINAL_ANSWER: −１２.００   -12
  FINAL_ANSWER: 400.00 dollars  400
  \boxed{400.00}             400

소수점 이하가 1개 이상 존재하고 모든 자릿수가 0일 때만 문자열에서 제거한다. 기존 notation wrapper,
통화기호, 천단위 콤마, 유니코드 마이너스·전각 숫자 정규화는 유지한다. float/Decimal 변환, 산술,
반올림, 수식 평가 없이 정규식과 문자열 조작만 사용한다. `-0.00`은 기존 -0 규칙처럼 `0`으로 만든다.

다음은 정수와 수학적으로 같아 보이더라도 계속 거부한다.

  FINAL_ANSWER: 400.01
  FINAL_ANSWER: 3/1
  FINAL_ANSWER: 12 + 5
  FINAL_ANSWER: 1e3
  FINAL_ANSWER: \frac{8}{2}
  FINAL_ANSWER: about 400 or 401
  FINAL_ANSWER: 00400.00

`400`과 `400.00`처럼 정규화 뒤 같은 explicit 후보는 같은 값으로 본다. `400`과 `401.00`처럼 서로
다른 canonical 후보는 기존 `conflicting_explicit_answers`로 실패한다.

[2. explicit answer가 fallback보다 우선하는 계약]
FINAL_ANSWER 또는 boxed가 한 번이라도 등장하면 generic body `last_integer`로 내려가지 않는다.

1. 모든 explicit occurrence를 source order로 수집한다.
2. 안전하게 canonicalize된 explicit 후보가 둘 이상이면 정규화 뒤 서로 같은지 확인한다.
3. 모두 같으면 기존 final_answer_marker > boxed 우선순위와 마지막 동일 marker 규칙으로 선택한다.
4. 서로 다른 canonical 값이 있으면 `conflicting_explicit_answers`로 실패한다.
5. 숫자를 포함한 explicit occurrence가 non-zero decimal, 분수, 수식 등이라 안전하게 변환되지 않으면
   다른 valid explicit 후보가 있더라도 전체 출력을 `non_integer_only`로 실패시킨다. 손상된 최종 선언과
   앞선 숫자 중 하나를 임의로 신뢰하지 않는다.
6. explicit 문법은 있지만 안전한 후보가 하나도 없으면 내용에 따라 `non_integer_only` 또는
   `no_supported_answer_marker`로 실패한다.
7. `standalone_last_line`과 `last_integer`는 FINAL_ANSWER/boxed occurrence가 전혀 없는 출력에만 적용한다.
8. failure reason enum을 추가해야 한다면 모든 downstream schema·테스트를 함께 갱신한다. 가능하면 기존
   세 reason의 의미를 보존해 새 enum 확산을 피한다.

반드시 지켜야 하는 회귀 예시는 다음과 같다.

  입력:
  There were 50 trees and the total was $400.00.
  FINAL_ANSWER: $400.00
  결과: answer=400, path=final_answer_marker

  입력:
  There were 50 trees.
  FINAL_ANSWER: 400.25
  결과: non_integer_only. 50으로 fallback하면 실패다.

  입력:
  \boxed{400}
  FINAL_ANSWER: $400.00
  결과: answer=400

  입력:
  \boxed{400}
  FINAL_ANSWER: 401.00
  결과: conflicting_explicit_answers

[3. 생성 prompt 출력 계약]
`configs/t8_self_consistency.json`과 기존 prompt를 덮어쓰지 않는다. 모델/revision/tokenizer,
temperature, top_p, max input/output tokens, seed와 engine 설정을 그대로 복제한
`configs/t11d_extractor_contract.json`을 만들고 prompt만 아래처럼 바꾼다. 실제 직렬화된 prompt와
SHA-256을 config·run metadata·manifest에 기록한다.

  Solve the following problem carefully.

  The required answer is a single base-10 integer. You may use units, currency
  symbols, decimals, and mathematical notation in your reasoning, but the final
  line must contain only the canonical integer answer.

  Output contract:
  - Write exactly one FINAL_ANSWER marker.
  - The final non-empty line must match:
    FINAL_ANSWER: -?(0|[1-9][0-9]*)
  - Do not use a currency symbol, comma, decimal point, unit, LaTeX, markdown,
    or \boxed{} on that line.
  - If the computed result is a whole monetary amount such as $400.00, write 400.
  - Do not write anything after the FINAL_ANSWER line.
  - Before finishing, silently verify that the final line satisfies this contract.

  Problem:
  {question}

reasoning 본문에는 단위·통화·소수·수식 사용을 허용한다. 제한은 마지막 canonical answer line에만 적용한다.
모델에게 gold 힌트, 답의 크기·부호, 문제 유형 tag, few-shot 정답 예시를 추가하지 않는다.

[4. 필수 단위·회귀 테스트]
`tests/test_extract.py`의 기존 테스트를 의도 없이 삭제하지 않는다. 기존 `FINAL_ANSWER: 2.0` 실패 기대는
새 explicit-only 계약에 맞게 성공 기대 2로 바꾸되, markerless `normalize_integer("2.0")`을 계속 거부할지
여부는 위의 explicit-only 범위와 일관되게 고정한다. 최소 다음을 테스트한다.

성공:
- `FINAL_ANSWER: $400.00` -> 400 / final_answer_marker
- `FINAL_ANSWER: 2.0` -> 2 / final_answer_marker
- `FINAL_ANSWER: +1,234.000` -> 1234
- `FINAL_ANSWER: -0.00` -> 0
- `\boxed{400.00}` -> 400 / boxed
- `\boxed{400}\nFINAL_ANSWER: $400.00` -> 400
- 단위 suffix, Unicode minus, full-width digit와 `.0...`의 조합

실패:
- `FINAL_ANSWER: 2.5`
- `FINAL_ANSWER: 12 / 4`
- `FINAL_ANSWER: 1e3`
- `FINAL_ANSWER: 12 + 5`
- `work 50\nFINAL_ANSWER: 400.25`가 50으로 fallback하는 경우
- `work 23\nFINAL_ANSWER:`가 23으로 fallback하는 경우
- `\boxed{400}\nFINAL_ANSWER: 401.00`
- valid explicit integer와 invalid numeric explicit occurrence가 함께 있는 경우

기존 forbidden-import, no-eval/no-arithmetic AST 테스트를 유지한다. `eval`, `exec`, `compile`, float,
Decimal, SymPy, NumPy, SciPy, z3 또는 계산 helper를 쓰지 않는다. extractor 함수에 label/question/ID/
다른 sample 인자를 추가하지 않는다.

최소 집중 테스트:

  pytest -q \
    tests/test_extract.py \
    tests/test_evaluate.py \
    tests/test_self_consistency.py \
    tests/test_submit.py \
    tests/test_weighted_vote.py

가능하면 전체 test suite도 실행하고 JUnit XML을 새 artifact에 보존한다. 기존 작업공간이 dirty하므로 이
작업과 무관한 실패·변경을 별도로 기록하고 사용자 파일을 되돌리지 않는다.

[5. frozen T8 extractor-only replay]
새 생성 없이 다음 frozen source를 읽기 전용으로 재사용한다.

- artifacts/t8_self_consistency/generations.jsonl — 119,584 rows = 3,737×32
- artifacts/t8_self_consistency/holdout_union_ids.txt — 3,737 IDs
- artifacts/t8_self_consistency/sweep.json
- artifacts/t8_self_consistency/manifest.json

현재 checkout의 generation/sweep/ID는 CRLF일 수 있다. raw hash가 다르면 파일을 수정하지 말고
CRLF→LF canonical content hash가 manifest와 같은지 확인한다. canonical label과 네 split도 manifest
hash를 검증한다.

실행 순서:
1. old extractor source/config hash와 새 extractor source/config hash를 기록한다.
2. old/new extractor를 같은 raw generation에 적용하되 label을 로드하지 않는다.
3. 각 candidate의 answer/path/failure reason과 question별 plurality prediction을 별도 JSONL에 쓴다.
4. old/new prediction 및 label-blind changed-case audit를 원자적으로 기록하고 SHA-256을 freeze한다.
5. old path가 frozen T8 majority 2,590/3,737=69.3069%, pass 3,154/3,737=84.3993%를 재현하는지
   assert한다. 이 재현 검증 뒤에만 canonical answer와 split membership을 결합한다.
6. new path의 paired rescue/break/net, pass, plurality, invalid와 split 변화를 계산한다.
7. 결과를 본 뒤 grammar/fallback/tie-break를 다시 바꾸거나 여러 변형 중 최고를 고르지 않는다.

반드시 보고할 항목:
- old/new plurality accuracy와 pass@32
- old/new invalid output rate와 failure reason 분포
- final_answer_marker / boxed / standalone_last_line / last_integer / none 분포
- answer 또는 path가 바뀐 candidate 수와 question 수
- rescued / broken / net gain, exact McNemar, paired bootstrap 95% CI
- random/template/hard/format별 paired 변화. 네 split은 겹치므로 합산하지 않는다
- explicit zero-decimal normalization으로 회수된 표·문항 수
- explicit barrier 때문에 제거된 기존 last_integer 표·문항 수
- 새롭게 invalid가 된 출력과 그 원래 fallback answer
- train-012155의 32개 old/new extraction과 plurality 결과

T8-3 +1.47pp는 참고 비교로만 병기한다. 새 extractor와 T8-3를 조합해 threshold나 조건을 재탐색하지 않는다.

[6. 새 prompt label-blind A/B canary]
새 prompt가 실제로 contract 위반을 줄이는지는 holdout이 아닌 별도 train pool에서 확인한다.

- T8 frozen union, T11 protected/validation, full leaderboard 1,000 IDs를 모두 제외한다.
- 남은 canonical train pool에서 ID hash로 128문항을 결정론적으로 선택하고 ID 목록·순서·SHA-256을
  generation 전에 freeze한다.
- 같은 base model/revision과 T8 sampling을 사용해 old prompt A와 new prompt B를 각각 128×4 생성한다.
- 두 arm은 같은 logical question/sample child seed를 사용하며 arm 차이는 prompt뿐이다.
- raw A/B를 먼저 원자적으로 기록·freeze한 다음 label 없이 형식 지표를 계산한다.
- full 3,737×32 재생성, prompt 문구 재탐색, seed 교체, 실패 sample 재생성은 하지 않는다.

label-blind canary 지표:
- 마지막 non-empty line이 정확히 `^FINAL_ANSWER: -?(?:0|[1-9][0-9]*)$`인 비율
- FINAL_ANSWER marker 수가 정확히 1개인 비율
- currency/comma/zero-decimal/non-zero-decimal explicit answer 비율
- final_answer_marker / boxed / standalone_last_line / last_integer / none 분포
- conflicting explicit answer, invalid output, hit-max, input truncation
- output token mean/median/p95와 wall time
- A/B candidate-level paired contract compliance 변화

prompt B는 strict final-line 비율이 A보다 높고, last_integer·invalid가 증가하지 않으며, hit-max가
+1pp보다 악화하지 않을 때만 `format_canary_passed`로 기록한다. 하나라도 실패하면 prompt B는 미채택하고
원래 T8 prompt를 유지한다. 작은 canary의 gold accuracy를 prompt 선택에 사용하지 않는다.

[판정과 T13 연결]
- extractor functional tests가 실패하면 `extractor_contract_failed`로 종료한다.
- tests가 통과하면 frozen T8 paired 결과는 `reused_holdout_diagnostic`로 기록한다. 순증이어도 이 값만으로
  최종 extractor를 자동 채택하지 않는다.
- prompt canary gate를 통과하면 `prompt_format_canary_passed`로만 기록하며 full generation을 시작하지 않는다.
- extractor와 prompt를 T13 최종 경로에 넣으려면 이 문서 밖의 fresh validation 또는 최종 테스트 전에
  별도로 동결된 확인 절차가 필요하다. 확인 전 T13 기본 경로는 기존 extractor와 prompt를 유지한다.
- 성공·실패와 무관하게 기존 artifact, leaderboard prediction, submission은 변경하지 않는다.
- 실행했다면 발표 자료용 누적 기록에 T11d 행을 추가한다. 문서만 추가한 단계에는 표 행을 만들지 않는다.

[비파괴 새 경로]
- configs/t11d_extractor_contract.json
- analysis/t11d_extractor_contract.py
- tests/test_t11d_extractor_contract.py
- data/t11d_extractor_contract/canary_ids.txt
- artifacts/t11d_extractor_contract/

[완료 조건]
- `$400.00` 등 zero-only decimal explicit 답이 canonical integer로 추출된다.
- invalid/incomplete explicit answer가 본문 last_integer로 fallback하지 않는다.
- explicit-only 범위, conflict, failure reason 의미가 단위 테스트로 고정된다.
- old T8 prediction이 frozen 69.3069%/84.3993%를 재현한 뒤에만 label이 결합된다.
- old/new prediction과 changed-case audit가 label-blind로 먼저 freeze되고 hash가 기록된다.
- reused T8 rescue/break/net과 split guardrail을 규칙 재탐색 없이 한 번 계산한다.
- old/new prompt canary가 같은 128 IDs·logical seeds에서 실행되고 label-blind 형식 지표가 기록된다.
- full holdout regeneration, 학습, API, leaderboard 생성·평가, submission 변경은 모두 0건이다.

[산출물]
src/extract.py
tests/test_extract.py
tests/test_t11d_extractor_contract.py
configs/t11d_extractor_contract.json
analysis/t11d_extractor_contract.py
data/t11d_extractor_contract/
  canary_ids.txt
  old-prompt-request-manifest.jsonl
  new-prompt-request-manifest.jsonl
artifacts/t11d_extractor_contract/
  input-verification.json
  old-extractions.jsonl
  new-extractions.jsonl
  changed-cases-label-blind.jsonl
  old-predictions.jsonl
  new-predictions.jsonl
  frozen-replay-comparison.json
  frozen-replay-comparison.md
  canary/old-prompt-generations.jsonl
  canary/new-prompt-generations.jsonl
  canary/format-comparison.json
  canary/format-comparison.md
  manifest.json
  tests.xml
```

---

## T12 — CMU-MATH ORM 재현: pointwise scoring + geometric weighted majority@32

```text
[왜 지금 하는가]
T8의 base majority@32는 합집합 69.31%, pass@32는 84.40%다. 즉 15.09pp의 oracle gap이 남아 있고,
564문항은 32개 후보 안에 정답이 있었지만 다수결이 오답을 골랐다. 반면 T9 GenSelect는 32개 후보를
한 프롬프트에 넣고 후보 번호를 생성하게 했으며, full32 55.90%, 동일 예산 28풀이+4선택 65.80%로
majority@32보다 낮았다. 후보를 한꺼번에 비교·생성하는 방식이 실패했다고 해서, 각 풀이를 독립적으로
채점하는 outcome reward model(ORM)까지 실패한 것은 아니다.

이번 과제는 AIMO Progress Prize 1에서 실제 2위를 기록한 CMU-MATH 파이프라인의 선택부를 재현한다.
새로운 선택 규칙을 여러 개 탐색하는 과제가 아니라, 아래에 고정한 pointwise ORM과 geometric weighted
majority@32 한 가지를 fresh validation에서 검증하는 과제다.

[고정 실행 환경 — 단일 호스트 2× RTX 4090]
T12의 학습, fresh generation, 추가 ORM-train candidate generation, fresh/reused candidate scoring은
VRAM 24GB인 NVIDIA GeForce RTX 4090 정확히 2개가 장착된 단일 호스트에서 실행한다. 두 GPU의 VRAM
48GB를 하나의 공유 메모리처럼 취급하지 않고, 각 GPU에 Qwen 3B base와 필요한 adapter의 완전한 복제본을
각각 올린다. GPU 하나나 다른 기종으로 조용히 폴백하지 않는다.

공통 분산 계약:
- generation은 tensor parallel을 쓰지 않는다. `CUDA_VISIBLE_DEVICES=0`과 `1`인 독립 vLLM worker 두 개를
  띄우고, 각 question의 k=32 전체를 한 worker가 생성한다. question을 두 worker 사이에서 쪼개 n=16씩
  생성하지 않아 기존 k=32 request·logical seed·sample_index 의미론을 보존한다.
- generation shard는 frozen question IDs를
  `(sha256("t12-generation-shard-v1:" + question_id), question_id)`로 정렬한 뒤 위치 `mod 2`로 정한다.
  fresh 1,000문항은 GPU별 정확히 500문항·16,000 generations가 된다. 새 ORM-train candidate를 만들 때도
  같은 규칙을 쓰며 두 shard의 문항 수 차이는 최대 1이다.
- ORM scoring도 tensor parallel이나 한 GPU 폴백을 쓰지 않는다. frozen candidate key
  `(question_id, sample_index)`를 `(sha256("t12-score-shard-v1:" + key), key)`로 정렬한 뒤 위치 `mod 2`로
  나누고, 동일 base+ORM adapter를 올린 독립 worker 두 개가 각각 점수를 낸다. fresh 32,000 candidates는
  GPU별 정확히 16,000개다.
- training만 `torchrun --nproc_per_node=2`의 NCCL DDP를 사용한다. 각 rank가 full model replica를 가지며
  tensor parallel, FSDP, ZeRO, CPU/NVMe offload는 사용하지 않는다. checkpoint와 최종 adapter는 rank 0만
  원자적으로 쓰고 두 rank가 모두 같은 global step을 마친 뒤에만 complete로 표시한다.
- logical rank 0/1의 shard 대응을 full run 전에 동결하고, 물리 GPU UUID 대응과 worker별 시작·종료 시각은
  실행 attempt마다 기록한다. worker는 서로의 JSONL에 쓰지 않고 전용 shard 파일에만 쓴다. 두 shard가
  모두 complete이고 hash 검증, 누락 0, 중복 0을 통과한 뒤에만 canonical key 순서로 병합한다.
- 한 GPU, worker 또는 NCCL rank가 실패하면 전체 phase를 실패로 기록한다. 성공한 shard는 보존하되 두
  RTX 4090이 다시 준비된 뒤 동일 manifest로 실패 shard만 재개한다. 단일 GPU 직렬 재실행은 허용하지 않는다.

[대회 검증 근거 — 보고서에 그대로 남길 것]
- CMU-MATH는 AIMO Progress Prize 1에서 private score 22/50으로 2위를 기록했다.
- policy model과 reward model은 모두 같은 DeepSeekMath-7B-RL 체크포인트에서 출발했다.
- reward model은 문제와 후보 풀이 하나를 입력받아 그 풀이가 맞을 확률을 0~1 점수로 출력했다.
- 같은 정수 답을 낸 후보를 묶고, 그 답의 표 수에 후보 점수의 기하평균을 곱해 최종 답을 정했다.
- 공개 train 10문항의 소규모 ablation에서 policy majority@32는 2/10, ORM weighted vote는 4/10이었다.
  표본이 10문항뿐이므로 이 수치를 일반화 근거로 과장하지 않는다. 다만 private 22/50은 이 파이프라인이
  실제 대회 제출에 사용됐다는 근거다.
- CMU는 약 7,000개 고유 문제, 37,880개 problem-solution pair를 사용했고, 문제별 correct/incorrect를
  1:1로 맞췄다. 서로 다른 보간 모델과 중간 체크포인트 출력으로 오답 다양성을 확보했으며, 정수로
  파싱되는 오답만 reward data에 포함했다. reward model 학습은 2 epochs, learning rate 2e-5였다.
- 근거:
  https://blog.ml.cmu.edu/2024/07/29/cmu-math-teams-innovative-approach-secures-2nd-place-at-the-aimo-prize/
  https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/writeups/cmu-math-2nd-place-solution-all-code-and-datasets-
  https://github.com/AIMO-CMU-MATH/CMU_MATH-AIMO
- AIMO-2의 GenSelect 논문은 시간 제약 때문에 해당 Kaggle 우승 제출에는 넣지 못했다고 명시한다.
  따라서 T9의 근거와 이번 ORM의 대회 검증 근거를 섞지 않는다.

[T9 GenSelect와의 차이 — 구현 전에 테스트로 고정]
- T9: 한 입력에 여러 후보를 넣고 선택 번호/답을 생성한다. 후보 순서와 컨텍스트 길이에 민감하다.
- T12: 한 번에 문제+후보 풀이 하나만 넣고 scalar logit 하나를 낸다. 후보 번호를 생성하지 않는다.
- T12는 요약문이 아니라 후보의 전체 풀이 trace를 채점한다.
- 최종 선택은 argmax 한 개가 아니라 답별 지지 표 수와 점수를 함께 쓰는 고정 수식이다.
- T9 adapter, T9 prompt, T9 candidate-summary 데이터는 재사용하지 않는다.

[규정 계약]
운영진은 동일 베이스에서 학습한 verifier adapter를 채점·선별 용도로 쓰는 것이 외부 모델 사용에
해당하지 않으며 Best-of-N에 사용할 수 있다고 서면 확인했다. 따라서 solver와 ORM 모두
Qwen/Qwen2.5-3B-Instruct의 고정 revision
aa8e72537993ba99e69dfaafa59ed015b17504d1에서 출발한 별도 LoRA adapter만 허용한다.

금지:
- 다른 베이스 모델, 외부 API, 외부 verifier, teacher의 추론 시 사용
- Python/SymPy/solver/코드 실행 결과를 ORM 입력으로 주거나 점수에 반영
- gold answer, 정답 여부, split 이름, 문제 ID를 ORM 추론 입력에 포함
- 리더보드 1,000문항 또는 보호 validation 문항을 ORM 학습에 포함
- 수식의 지수, score threshold, answer=0 penalty, 후보 필터를 validation 결과를 보고 조정

허용:
- solver adapter와 ORM adapter를 같은 고정 베이스 위에서 순차적으로 로드
- 모델이 이미 출력한 문제+풀이 문자열을 ORM이 읽어 scalar score를 출력
- 고정 답 추출기로 후보의 정수 답 문자열을 읽고 같은 답끼리 그룹화
- 아래에 사전 등록한 기하평균 가중 투표와 결정적 tie-break

[목표]
1. 오염 없는 pointwise ORM train/validation data를 만든다.
2. 동일 Qwen 베이스의 별도 LoRA ORM을 CMU 설정에 가깝게 2 epochs 학습한다.
3. label-blind로 동결한 base k=32 후보 풀에서 각 풀이를 독립 채점한다.
4. geometric weighted majority@32가 raw majority@32와 동결 T8-3 filter@32를 fresh validation에서
   유의하게 이기는지 판단한다.
5. 통과할 때만 T13의 최종 제출 후보로 넘긴다. reused T8 holdout은 진단 전용이다.
6. 모든 GPU-heavy phase가 동결된 2× RTX 4090 data-parallel/DDP 경로를 실제로 사용했음을 worker별
   처리량, peak VRAM, utilization, OOM과 merged coverage로 증명한다.

[Phase 0 — 입력·보호 집합 감사]
학습이나 새 생성을 하기 전에 다음을 실행하고 artifacts/t12_cmu_orm/input-verification.json에 기록한다.
- base/tokenizer revision, solver prompt/config, 기존 candidate artifact의 SHA-256
- GPU 수·정확한 model name·UUID·VRAM, driver/CUDA/PyTorch/vLLM/NCCL 버전, power limit,
  `nvidia-smi topo -m`, rank-to-UUID mapping, 각 GPU의 시작 전 free VRAM
- 두 GPU에서 model-load smoke, NCCL all-reduce smoke, 독립 vLLM worker smoke가 모두 성공했는지
- T11/T11b/T11c에서 확정한 보호 5,475 IDs
- 리더보드 원본 1,000 IDs 전체
- T3~T11d에서 한 번이라도 모델/규칙 선택에 사용한 validation·holdout IDs
- exact normalized text, near-duplicate, template-group 기준의 train/validation 교집합 수

교집합이 1건이라도 있으면 해당 question/template group 전체를 ORM train에서 제거한다. 보호 집합을
외부 API나 다른 모델에 보내지 않는다. GPU가 정확히 2× RTX 4090이 아니거나 smoke가 하나라도 실패하면
`hardware_gate_failed`로 종료한다. audit와 hardware gate가 끝나기 전에는 학습·fresh generation을
시작하지 않는다.

[Phase 1 — fresh validation을 먼저 동결]
1. 기존 선택에 한 번도 쓰지 않은 canonical 문제 중 1,000문항을 고른다.
2. ORM train과 question-disjoint이면서 normalized-text/near-duplicate/template-group까지 disjoint하게 만든다.
3. 유형·난도 proxy·문제 길이·답 부호/자리수·format strata가 한쪽으로 쏠리지 않게 층화한다.
4. IDs와 split 생성 seed를 data/cmu_orm/validation.csv 및 validation-manifest.json에 쓰고 해시한다.
5. 이후 이 1,000문항은 ORM train data 생성, checkpoint 선택, prompt/filter 탐색에 절대 사용하지 않는다.

validation IDs를 동결한 뒤 T8의 고정 base prompt와 생성 설정으로 각 문항 k=32를 정확히 한 번 생성한다.
생성 단계는 gold column을 읽을 수 없는 독립 vLLM worker 두 개로 실행한다. 위의 generation shard를
generation-shard-manifest.json에 먼저 쓰고 hash한 뒤, GPU 0/1이 각각 전용 JSONL과 run metadata를 쓴다.
두 worker는 동일 model/prompt/sampling config를 쓰고 question별 k=32 request를 유지한다. worker별 raw
outputs, logical seed, request hash를 freeze하고, 각 GPU 500문항·16,000 rows, 전체 question별 정확히
32 rows, 누락·중복 0을 검증한 뒤 canonical `(question_id, sample_index)` 순서로
artifacts/t12_cmu_orm/fresh-validation/generations.jsonl을 만든다. 병합본까지 hash한 뒤 후보 풀을 freeze한다.

[Phase 2 — CMU식 ORM 학습 데이터]
기존 비보호 RFT candidate pool과 base/RFT 중간 checkpoint의 저장된 출력만 사용한다. 새 candidate가
필요하면 오직 ORM-train IDs에 대해서만 고정 manifest를 먼저 만들고, 위의 2-GPU generation shard와
독립 vLLM worker 계약으로 생성한다. 모델 보간은 하지 않는다.

각 row는 다음 필드만 갖는다.
- question_id, normalized_question, full_candidate_trace, extracted_integer
- label: frozen exact-match extractor로 extracted_integer == gold이면 1, 아니면 0
- generator_source, generator_checkpoint_hash, prompt_hash, sampling_seed

데이터 규칙:
- 정수 답이 파싱되는 후보만 사용한다. malformed/empty 후보를 쉬운 negative로 채우지 않는다.
- 각 문제는 correct와 incorrect가 모두 있어야 한다.
- 문제별 positive:negative를 정확히 1:1로 맞춘다. 한쪽이 많으면 source/checkpoint/seed 다양성을 우선해
  deterministic hash sampling하며, 문제 하나가 전체 loss를 지배하지 않게 각 class 최대 4개로 제한한다.
- 같은 raw trace 중복, gold가 trace에 주입된 row, 보호/near-duplicate/template-overlap row는 제거한다.
- split은 row가 아니라 question/template group 단위다.
- 목표는 CMU 규모에 근접한 7,000 unique questions / 약 38,000 pairs다. 최소 5,000 questions와
  25,000 balanced pairs를 확보하지 못하면 임의로 완화하지 말고 data_gate_failed로 종료한다.
- train manifest에 class 수, question 수, source/checkpoint별 분포, 길이 분위수, 제거 사유별 수량과
  모든 입력/출력 SHA-256을 기록한다.

[Phase 3 — pointwise ORM 학습]
동일 베이스의 AutoModelForSequenceClassification(num_labels=1) + LoRA를 사용한다. classifier score head는
modules_to_save에 포함해 adapter와 함께 저장한다. 입력은 question과 full_candidate_trace 하나뿐이며
gold answer나 extracted correctness는 넣지 않는다. score는 sigmoid(scalar logit)로 정의한다.

고정 설정:
- base revision: aa8e72537993ba99e69dfaafa59ed015b17504d1
- bf16, gradient checkpointing, packing=false, max_length=4096
- LoRA rank=64, alpha=128, dropout=0.05, 기존 프로젝트와 같은 target modules
- BCEWithLogitsLoss, 별도 class weight 없음(문제별 1:1 balance)
- epochs=2, learning_rate=2e-5, warmup_ratio=0.03, cosine schedule
- DDP world_size=2, NCCL, GPU별 per-device batch size=1, gradient accumulation=16
- global effective batch size = 2 GPUs × 1 sample × 16 accumulation = 32, seed=42
- 두 rank의 deterministic distributed sampler는 같은 frozen row order에서 출발하고 epoch별 `set_epoch`를
  호출한다. world_size나 accumulation을 자동 변경하지 않는다.
- epoch 2를 사전 등록 final로 사용한다. validation을 보고 epoch 1과 2 중 고르지 않는다.
- learning rate, epoch, rank, prompt에 대한 sweep을 하지 않는다.

CMU의 full reward-model fine-tuning을 대회 규정에 맞는 same-base LoRA로 치환한 것이 유일한 의도적
차이임을 보고서에 명시한다. 학습 중 candidate-level ROC-AUC, PR-AUC, Brier score, ECE, positive/negative
score 분포와 loss를 기록하되, 이 값으로 집계 수식을 바꾸지 않는다. rank별 samples/tokens, step time,
active GPU utilization, peak VRAM, OOM, NCCL 오류와 전체 DDP throughput도 함께 기록한다.

[Phase 4 — geometric weighted majority@32]
frozen candidate i의 추출 답을 y_i, ORM sigmoid score를 s_i라고 한다. 수치 안정성을 위해
s_i = clip(s_i, 1e-6, 1-1e-6)로 고정한다. 답 a를 낸 후보 수를 n_a라 할 때 점수는 정확히 다음이다.

  W(a) = n_a * (product_{i: y_i=a} s_i)^(1/n_a)
       = n_a * exp(mean_{i: y_i=a}(log(s_i)))

W(a)가 가장 큰 정수 답을 최종 답으로 낸다. 동률이면 해당 답의 최초 생성 index가 더 작은 쪽,
그마저 같으면 정수값 오름차순으로 고정한다. invalid 후보는 그룹에서 제외한다. score가 NaN이거나
유효 후보가 0개면 raw majority의 기존 fallback을 사용하고 발생 수를 기록한다.

다음은 primary가 아니며 진단으로만 계산한다.
- 단일 최고점 후보 argmax ORM
- raw majority@32
- 동결 T8-3 vote-quality filter@32
- oracle pass@32

arithmetic mean, max score, score 합, score exponent, threshold, answer=0 감점, 풀이 길이 감점,
extraction-path weight는 이번 과제에서 탐색하지 않는다. ORM과 T8-3 filter를 결합한 새 규칙도 만들지 않는다.

[Phase 5 — label-blind freeze 후 단 한 번 평가]
ORM score는 gold를 읽지 않는 두 독립 GPU worker가 위의 score-shard-manifest.json에 따라 계산한다.
두 worker는 같은 frozen scoring batch size와 tokenizer/model/adapter hash를 사용하고 각자 전용 score
JSONL과 metadata를 쓴다. 각 16,000 candidates, 전체 coverage 32,000, key 누락·중복 0, finite raw logit을
검증한 뒤에만 canonical candidate 순서로 candidate-scores.jsonl을 병합한다. reused T8 replay도 같은
2-GPU scoring 계약을 사용한다.

fresh validation의 다음 파일을 gold를 읽기 전에 생성·해시한다.
- raw majority predictions
- frozen T8-3 filter predictions
- candidate별 ORM raw logit/sigmoid score
- answer group별 n, geometric mean, W(a), tie/fallback reason
- ORM weighted predictions
- changed-case label-blind audit

모든 prediction과 config hash를 manifest에 쓴 뒤에만 gold를 결합한다. 평가는 다음을 모두 낸다.
- 전체 accuracy와 raw majority/T8-3 대비 delta
- paired rescue/break/net, exact McNemar p-value, paired bootstrap 95% CI
- 사전 동결한 5 folds 각각의 delta
- hard/format/길이/답 부호·자리수 strata와 invalid/tie/fallback 비율
- candidate-level ROC-AUC/PR-AUC/Brier/ECE 및 answer-cluster 크기별 정확도
- 1,000문항 solve 시간, 32,000 candidate score 시간, GPU별/합산 처리량, GPU별 peak VRAM·utilization·OOM,
  두 worker를 동시에 시작한 시점부터 merge 완료까지의 2-GPU makespan과 총 wall-clock

기존 T8 합집합 3,737문항 replay는 fresh 판정이 끝난 뒤 재현성·오류 분석용으로 한 번만 실행한다.
그 결과로 formula나 모델을 바꾸거나 채택 판정을 뒤집지 않는다.

[사전 등록 판정]
채택(PASS)은 다음을 모두 만족할 때만 한다.
1. fresh 1,000문항에서 ORM weighted@32가 raw majority@32와 frozen T8-3 filter@32를 각각 이긴다.
2. 두 baseline 중 정확도가 높은 쪽보다 절대 +1.5pp 이상이다.
3. 그 강한 baseline과의 paired McNemar p<0.05이고 bootstrap 95% CI 하한이 0보다 크다.
4. 사전 동결 5 folds의 delta가 모두 양수다.
5. hard와 format strata 어느 쪽도 강한 baseline보다 2.0pp 초과 하락하지 않는다.
6. candidate-level ROC-AUC >= 0.65이고 NaN score는 0건이다.
7. 동결된 2× RTX 4090 경로의 solve+ORM score+집계 1,000문항 makespan이 18시간 이하이며 두 GPU 모두
   OOM=0이다. 두 worker의 GPU 시간을 더한 값이 아니라 실제 동시 실행 wall-clock으로 판정한다.

+0.01~+1.49pp이거나 통계/guardrail 하나라도 실패하면 HOLD다. 0 이하이면 REJECT다. HOLD/REJECT이면
ORM adapter를 최종 환경에 로드하지 않고 기존 T8/T10 경로를 유지한다. 결과를 본 뒤 재학습·재보정·
formula 변경을 원하면 새 과제와 새 fresh validation을 먼저 사전 등록해야 한다.

[비파괴 구현]
기존 src/train.py, src/generate.py, src/vote.py, src/extract.py의 현재 동작을 바꾸지 않는다.
새 파일로만 구현한다.
- configs/t12_cmu_orm.json
- src/t12_sharding.py
- src/build_orm_data.py
- src/train_orm.py
- src/orm_score.py
- src/orm_vote.py
- scripts/run_t12_cmu_orm.sh
- tests/test_t12_sharding.py
- tests/test_orm_data.py
- tests/test_orm_vote.py

data/cmu_orm/와 artifacts/t12_cmu_orm/만 새로 쓴다. 기존 artifact, leaderboard prediction,
submission.csv는 수정하지 않는다. 중단·재시작 시 raw generation이나 score 파일을 덮어쓰지 말고
logical shard/config manifest hash가 같을 때만 이어서 실행한다. 물리 UUID가 바뀌면 정확히 같은
2× RTX 4090·software stack으로 hardware preflight와 rank-to-UUID rebind audit를 새로 통과한 뒤 재개한다.

[필수 단위 테스트]
- 입력 ID/candidate 순서를 섞어도 2-way shard assignment와 manifest hash가 같고 두 shard 크기 차이가
  최대 1이다.
- generation에서는 한 question의 sample_index 0..31이 한 GPU에만 있고, score에서는 각 candidate key가
  정확히 한 GPU에만 있다.
- worker 완료 순서를 뒤집어도 canonical merge가 byte-identical이며 missing/duplicate/cross-shard write를
  모두 거부한다.
- world_size=2, per-device batch=1, accumulation=16일 때만 global effective batch가 32다.
- 문제별 positive:negative가 정확히 1:1이고 question/template leakage가 0이다.
- ORM scoring batch size나 candidate 순서를 바꿔도 각 raw logit이 허용 오차 안에서 같다.
- [0.9, 0.9] 두 표와 [0.99] 한 표에서 전자의 W가 더 크다.
- [0.9, 0.1]의 기하평균이 arithmetic mean으로 잘못 계산되지 않는다.
- score clip, invalid 제외, NaN fallback, tie-break가 golden fixture와 byte 동일하다.
- gold/split/question_id가 ORM inference prompt에 들어가지 않는다.
- 동일 manifest 재실행 결과 prediction과 group audit SHA-256이 같다.

[필수 2-GPU integration smoke]
- 두 device name이 모두 정확히 `NVIDIA GeForce RTX 4090`이고 각 worker가 서로 다른 UUID만 점유한다.
- DDP 한 optimizer step 뒤 두 rank의 trainable parameter checksum이 허용 오차 안에서 같다.
- frozen generation fixture의 두 shard를 다시 실행했을 때 각각의 raw generation과 merged output이
  byte-identical하다.
- frozen 32-candidate fixture의 2-GPU merged logits가 단일 4090 reference와 허용 오차 안에서 같고,
  group weight와 최종 prediction은 byte-identical이다.
- 한 worker 강제 실패 시 merged complete 파일을 만들지 않고, 성공 shard를 보존한 재개가 원래 실행과
  byte-identical하다.

[완료 조건]
- 공식 CMU 근거, 로컬 LoRA 치환점, 데이터 규모 차이가 report에 명시되어 있다.
- protection audit와 fresh validation manifest가 학습 전에 freeze되어 있다.
- hardware preflight에 2× RTX 4090 UUID/topology/software와 모든 smoke 성공이 남아 있다.
- data gate를 통과한 balanced pointwise ORM corpus와 전 행 provenance가 있다.
- world_size=2·global batch=32의 고정 설정으로 만든 2-epoch ORM adapter와 rank별 학습 로그가 있다.
- fresh k=32 후보·ORM score·group weight·prediction이 label-blind로 먼저 freeze되어 있고, generation과
  scoring 각각 두 shard의 coverage·hash·병합 감사가 있다.
- 사전 등록한 PASS/HOLD/REJECT가 모든 지표와 함께 report에 적혀 있다.
- PASS가 아니면 T13 최종 경로와 submission에 ORM 흔적이 없다.
- 실행 전에는 발표 자료용 누적 표에 성능 행을 추가하지 않는다. 실행 후 확정값만
  "+ CMU-MATH pointwise ORM geometric weighted majority@32 (T12)" 행에 기록한다.

[산출물]
configs/t12_cmu_orm.json
src/t12_sharding.py
src/build_orm_data.py
src/train_orm.py
src/orm_score.py
src/orm_vote.py
scripts/run_t12_cmu_orm.sh
tests/test_t12_sharding.py
tests/test_orm_data.py
tests/test_orm_vote.py
data/cmu_orm/
  validation.csv
  validation-manifest.json
  train.jsonl
  train-manifest.json
artifacts/t12_cmu_orm/
  input-verification.json
  hardware-preflight.json
  distributed-run-manifest.json
  adapter/
  train-metrics.json
  train-rank-metrics/
  fresh-validation/generation-shard-manifest.json
  fresh-validation/generation-shards/
  fresh-validation/generations.jsonl
  fresh-validation/score-shard-manifest.json
  fresh-validation/score-shards/
  fresh-validation/candidate-scores.jsonl
  fresh-validation/group-weights.jsonl
  fresh-validation/predictions.jsonl
  fresh-validation/changed-cases-label-blind.jsonl
  fresh-validation/evaluation.json
  fresh-validation/evaluation.md
  reused-t8-diagnostic.json
  manifest.json
  tests.xml
```

---

## T12b — T12b-dev: question-local ORM 내부 개발

```text
[왜 T12를 그대로 키우지 않는가]
T12는 fresh 1,000문항에서 raw majority@32 87.40%, frozen T8-3 filter@32 87.40%, ORM weighted@32
87.60%로 순증 2문항(+0.20pp)에 그쳐 HOLD다. McNemar p=0.790527, paired bootstrap 95% CI는
[-0.50,+0.90]pp이고, 5 folds의 delta도 +0.5/-0.5/+0.5/-0.5/+1.0pp라 채택 근거가 없다.

문제는 후보 생성량이 아니다. oracle pass@32가 96.90%이므로 raw majority가 틀린 126문항 중 95문항에는
정답 후보가 있었다. 그러나 T12가 회수한 것은 8문항뿐이고 6문항을 파손했으며, 10문항은 오답에서 다른
오답으로 바꿨다. 남은 선택 가능 오류 87문항을 분해하면 42문항은 정답 그룹의 ORM 평균 점수가 더 높아도
현재 `n * geometric_mean(score)`의 표 수 항에 졌고, 45문항은 ORM 자체가 오답 그룹을 정답 그룹 이상으로
채점했다. ORM argmax 정확도도 82.80%로 raw majority보다 4.60pp 낮다.

후보 전체를 섞은 global ROC-AUC 0.8123만 보면 채점기가 좋아 보이지만, 실제 선택과 가까운 question-local
macro AUC는 0.7405다. fresh valid candidate의 실제 정답률은 81.09%인데 train prior는 50%, 평균 예측
score는 64.23%, ECE는 16.86%다. hard 222문항에서는 +1.80pp였지만 non-hard 778문항에서는 -0.26pp다.
따라서 T12b의 우선순위는 k 증가나 단순 7B 확대가 아니라 다음 순서로 고정한다.

  question-local ranking 학습
  -> source-balanced hard-negative mining
  -> answer-group aggregation 학습
  -> selective override
  -> 조건부 7B 검토

[평가 지위 — 이전 실험 관행과 T12b-dev의 범위]
T3~T10은 같은 고정 holdout 합집합 3,737문항을 반복 사용했다. T12에서 `fresh`라고 부른 1,000문항도
T8~T10의 선택에는 쓰이지 않았지만, 실제로는 1,000문항 전부가 T5 RFT pool에 속하고 T5의 k=16 생성·정답
일치 집계에 사용됐다. 따라서 프로젝트에서 지금까지 사용한 `fresh`는 "직전 모델·규칙 선택에 쓰이지 않은
문항"이라는 상대적 의미였지, T3~T12의 어떤 처리에도 한 번도 등장하지 않은 절대적 untouched 집합이라는
뜻은 아니었다.

T12b에만 모든 과거 train/validation/holdout과 T5 pool까지 배제한 fresh-2 1,000문항을 요구하면 canonical
16,373문항 중 eligible 문항이 0개라 실행 자체가 불가능하다. 이 과제에서는 그 조건을 제거하고, 기존 T12
ORM corpus의 6,034문항을 template-group nested CV로 재사용하는 `T12b-dev`를 수행한다.

다음 경계는 유지한다.
- 기존 T12 fresh 1,000문항과 reused T8 3,737문항은 T12b 가설의 진단 근거일 뿐 loss, coefficient,
  calibration, threshold 또는 arm 선택에 다시 사용하지 않는다.
- T12b-dev의 모든 정확도·AUC·통계는 내부 개발 성능이다. `fresh PASS`, `PASS/HOLD/REJECT`, 독립 재현,
  T13 승격 근거로 표현하지 않는다.
- 내부 개발이 끝나면 후보 설정 하나를 결정적으로 동결할 수 있다. 그 뒤 원본 leaderboard 1,000문항에
  대한 label-blind 예측 파일을 만들 수 있지만, 실제 제출과 점수 확인은 별도 명시적 `T12b-LB` 단계다.
- leaderboard 점수를 확인한 뒤 loss, coefficient, calibration, threshold, fallback을 고치거나 같은 가설로
  재제출하지 않는다. 그 점수는 `one-shot leaderboard evidence`이지 fresh-2 검증이 아니다.

[목표]
1. 독립 후보의 절대 정답 확률만 맞추는 pointwise BCE를 question-local 상대 순위 학습으로 바꾼다.
2. generator source와 prompt/style만 보고 정오를 맞히는 shortcut을 억제한다.
3. 고정 `n * geometric mean` 대신 answer group 단위 점수를 nested out-of-fold 예측으로 학습한다.
4. ORM이 충분한 증거를 보일 때만 raw majority를 뒤집고, 쉬운 다수결 정답은 보존한다.
5. 내부 nested CV에서 ranking, group selector, selective override의 후보 하나를 고정하되, 결과를 T13이나
   최종 채택 판정에 자동 반영하지 않는다.

[고정 실행 경계]
- base/tokenizer revision, solver adapter, candidate generation prompt/sampling, 정답 추출기는 T12와 같다.
- 내부 OOF 평가는 이미 존재하는 T5 base k=16 풀을 쓰고, 동결 후보의 leaderboard 추론만 기존 k=32 계약을 쓴다.
- T12의 단일 호스트 2× RTX 4090 generation/scoring shard와 DDP 계약을 그대로 따른다.
- 먼저 3B ORM에서 ranking과 aggregation 가설을 검증한다. 이번 과제에서 7B와 k>32를 함께 탐색하지 않는다.
- Python/SymPy/solver/코드 실행 결과, gold, split 이름, question ID를 ORM 입력이나 group feature에 넣지 않는다.
- 기존 T12 artifact, submission, adapter를 덮어쓰지 않고 전부 새 namespace에 쓴다.
- 기존 T12 full adapter는 6,034문항 전체를 학습했으므로 같은 문항의 OOF 대조군으로 쓰지 않는다. 공정한
  Arm A는 outer fold마다 pointwise BCE를 다시 학습하고 held-out question에만 nGM을 적용한다.

[Phase 0 — T12 진단 동결과 provenance]
다음 입력을 artifacts/t12b_question_local_orm/input-verification.json에 경로와 SHA-256으로 기록한다.
- artifacts/t12_cmu_orm/fresh-validation/evaluation.json
- report/t12-orm-diagnostic-2026-08-28/diagnostic-summary.json
- data/cmu_orm/train.jsonl 및 train-manifest.json
- T12 adapter와 candidate generator source 전체
- T10a C-1, T8-3, T12 leaderboard submission.csv와 audit

진단 기준값은 다음과 같이 manifest에 복사해 이후 보고서의 비교 기준으로만 쓴다.
- raw/ORM/oracle/argmax = 87.40/87.60/96.90/82.80%
- rescue/break/wrong-to-wrong = 8/6/10, selectable recovery = 8/95
- global/within-question macro AUC = 0.8123/0.7405
- mean score/fresh correctness/ECE = 64.23/81.09/16.86%
- score-lost/support-lost mechanism = 45/42 questions
- hard/non-hard delta = +1.80/-0.26pp

T12 leaderboard 831행은 raw majority와 50건, T8-3과 49건이 달랐고, valid prediction이 없어 정수 0을
강제로 쓴 행이 3건이었다. 이 CSV에는 label이 없으므로 성능 근거로 해석하지 않고, T12b의 fallback 계약에서
`forced zero = 0건`을 요구하는 label-blind 안전 근거로만 사용한다.

[Phase 1 — 기존 6,034문항과 nested split을 학습 전에 동결]
1. ranking 학습 원본은 `data/cmu_orm/train.jsonl`의 30,912행·6,034문항으로 고정한다. T12 manifest와 각
   candidate source의 경로·행 수·SHA-256을 기록하고, 행을 새로 만들거나 삭제하면 별도 corpus version으로
   남긴다.
2. group selector와 override의 공통 개발 추론 풀은 `artifacts/t5_rft_r1/generations.jsonl`에서 위 6,034문항에
   해당하는 base 후보를 사용한다. 현재 coverage는 모든 문항에 정확히 k=16이다. 균형 학습용으로 sampling된
   `train.jsonl`의 표 수를 raw majority로 사용하지 않는다.
3. k=16 개발 풀은 leaderboard k=32의 대용 지표일 뿐이다. 보고서에 이 분포 차이를 명시하고, 절대 k=32
   성능처럼 표기하지 않는다. vote margin과 support threshold는 유효 후보 수로 나눈 비율로 정의해 k 변화에
   덜 민감하게 만든다.
4. 6,034문항을 `template_group_id` 단위 outer 5-fold로 고정한다. 각 outer-train을 다시 template group 단위
   inner 4-fold로 나눈다. 같은 question/template group의 모든 candidate source와 trace는 항상 같은 fold다.
   row split은 금지한다.
5. 각 outer fold에서는 inner fold만으로 loss/checkpoint, calibration, group coefficient, override threshold를
   고른 뒤 outer-test를 정확히 한 번 예측한다. outer-test label은 그 fold의 어떤 fit이나 선택에도 들어가지
   않는다. 최종 OOF는 6,034문항 각각이 정확히 한 번 outer-test가 된 prediction만 합친다.
6. 최종 배포 설정은 outer 5회에서 inner CV가 고른 조합의 최빈값으로 정한다. 동률이면 override coverage가
   낮은 조합, 그다음 사전 등록 lexicographic 순서를 쓴다. leaderboard 점수나 기존 T12/T8 진단값으로
   동률을 깨지 않는다.
7. 기존 T12 fresh 1,000, reused T8 3,737, leaderboard 1,000 ID가 6,034문항 개발 split이나 OOF fit 입력으로
   추가 유입되지 않았음을 감사한다. T5 pool과의 중복은 이 개발 과제의 의도된 재사용이므로 실패 조건이 아니다.

split hash가 달라지거나 outer-test label이 fit/selection에 들어가면 해당 CV 실행은 무효화한다. 다만 이
무효화는 새 untouched 문항을 요구하지 않고, 같은 동결 split에서 오염된 fold를 처음부터 다시 실행한다.

[Phase 2 — source-balanced question batches와 hard negatives]
현재 T12 최종 train 30,912행은 문제별 전체 1:1이지만 generator source별 label prior가 크게 다르다.
- t11 cot-boxed: 13,228행, positive 57.36%
- t5 base: 11,515행, positive 46.07%
- t5 targeted: 3,284행, positive 55.76%
- t12 high-temperature: 2,211행, positive 10.45%
- t7 hard-tail: 674행, positive 74.48%

이 분포에서는 ORM이 풀이의 수학적 타당성 대신 prompt/source/길이/style을 shortcut으로 쓸 수 있다.
T12b corpus는 다음 계약으로 다시 만든다.
- batch의 기본 단위는 row가 아니라 question이다. 한 question의 positive와 negative를 같은 batch item에 둔다.
- question마다 서로 다른 trace의 positive 2~4개와 hard negative 2~4개를 우선 확보한다. 한쪽이 2개 미만이면
  pairwise/listwise 학습에서 제외하고 pointwise 보조 데이터에도 별도 표기한다.
- source별 positive:negative를 정확히 1:1로 맞춘다. source 안에서도 prompt format, trace length quartile,
  problem type, hard/normal, extraction path, answer support bucket의 양·음성 분포를 deterministic matching한다.
- 모든 교차 cell을 억지로 정확히 맞춰 데이터가 소실되지 않게 source 1:1을 hard constraint로 두고,
  나머지 속성은 standardized mean difference 절대값 <=0.10을 data gate로 둔다.
- 동일 normalized trace와 사실상 같은 풀이의 반복 sample은 question 안에서 한 번만 사용한다. 같은 답을 낸
  복제 trace 수로 loss가 커지지 않게 unique trace hash를 저장한다.
- hard negative는 정수로 정상 추출되지만 오답인 후보 중 frozen cross-fitted T12 score가 높은 후보,
  raw vote 상위 오답 그룹 후보, 정답과 근접한 그럴듯한 중간 계산 오류를 우선한다. gold answer 문자열을
  입력 feature로 쓰지 않고 label 생성과 offline mining에만 사용한다.
- source, prompt, 길이를 가리는 shortcut audit를 별도로 두고, source-only/length-only probe가 높은 AUC를
  내면 학습을 시작하지 않는다. 목표는 두 probe 모두 ROC-AUC <=0.60이다.
- 위 balance·dedup 뒤에도 최소 5,000 unique questions와 25,000 rows, question별 양·음성 동시 보유를
  만족해야 한다. 부족하면 sampling 규칙을 사후 완화하지 않고 `data_gate_failed`로 종료한다.

[Phase 3 — question-local ranking ORM]
입력 형식과 sequence-classification head는 T12와 같고, candidate logit을 z_i라 한다. positive i와 같은
question의 negative j에 대해 pairwise loss를 다음처럼 정의한다.

  L_pair(i,j) = softplus(-(z_i - z_j))

한 question에 여러 positive가 있으면 candidate 전체의 listwise 보조 loss를 다음처럼 정의한다.

  L_list(q) = -log(
      sum_{i in positive(q)} exp(z_i / tau)
      / sum_{j in valid(q)} exp(z_j / tau)
  )

internal metric을 계산하기 전에 `tau in {0.5, 1.0}`, `lambda_pair in {0.5, 1.0}`,
`lambda_list in {0, 0.25}`의 고정 grid를 manifest에 적는다. primary ranking arm은
`L = L_BCE + lambda_pair * L_pair`이고, `+ lambda_list * L_list`는 internal nested CV에서만 비교하는
사전 등록 보조 arm이다. grid를 본 뒤 확장하거나 outer OOF·leaderboard 결과로 값을 다시 고르지 않는다.

학습 규칙:
- question sampler가 같은 question의 positive/negative를 동일 rank에 보낸다. DDP rank 사이에 pair를
  쪼개지 않는다.
- 한 question의 모든 가능한 pair를 쓰지 않고 deterministic hard-negative 우선 최대 16 pairs로 제한한다.
- source와 question마다 loss 총합을 정규화해 후보 수가 많은 source/question이 gradient를 지배하지 않는다.
- 3B base revision, LoRA 구조, bf16, max_length=4096, 2×4090 DDP는 T12를 유지한다.
- epoch/checkpoint와 loss arm은 within-question macro AUC를 primary, answer-group top-1 accuracy를 secondary,
  Brier/ECE를 guardrail로 한 nested group CV에서만 고른다.
- global candidate AUC만 높고 within-question macro AUC가 오르지 않는 checkpoint는 선택하지 않는다.

[Phase 4 — answer-group aggregation]
T12의 `W(a) = n_a * GM(p_i)`는 표 수와 후보 품질의 상대 크기를 고정해 42개의 support-lost error를 남겼다.
T12b는 후보 확률 p_i의 logit z_i를 answer group a별로 모아 다음 점수를 학습한다.

  G(a) = alpha * log(n_a)
       + beta  * mean(z_i : y_i = a)
       - gamma * variance(z_i : y_i = a)

alpha, beta, gamma는 0 이상으로 제한하고 L2 regularization을 둔다. 각 internal fold의 train portion에서
정답 answer group의 softmax cross-entropy를 최소화해 fit하고, 그 fold의 held-out questions에만 적용한다.
최종 coefficient 설정은 각 outer fold의 inner CV 선택 결과를 Phase 1의 최빈값 규칙으로 결정한다. 그 뒤
6,034문항 전체로 최종 ORM을 학습하고 같은 설정의 coefficient를 한 번 fit해 leaderboard 추론 전에 freeze한다.

각 group에 support n, unique trace count, mean/median/min/std logit, raw vote margin, invalid/hit-max rate를
모두 로그로 남기되 이번 primary G(a)는 위 3개 항만 사용한다. 확장 feature meta-selector는 진단용이며
`DEV_CANDIDATE` 판정을 뒤집지 않는다. 확장하려면 별도 개발 과제로 등록한다.

[Phase 5 — calibration은 보조, selective override가 최종 정책]
train 50% prior와 fresh candidate 81.09% correctness 차이를 줄이기 위해 internal calibration folds에서만
다음을 비교한다.
- temperature scaling
- class-prior logit correction
- question-wise logit centering 뒤 temperature scaling

held-out Brier/ECE가 가장 낮고 within-question macro AUC와 group top-1을 낮추지 않는 방법 하나만 freeze한다.
calibration은 score 해석을 고치는 단계이지, 현재 45개의 scorer-misrank를 해결한 것으로 과장하지 않는다.

최종 정책은 raw majority를 default로 둔다. ORM top group이 raw top group과 다를 때 아래 label-blind 조건을
모두 만족하는 경우에만 ORM으로 override한다.
- raw top-2 normalized vote margin <= frozen m_max
- ORM alternative normalized support >= frozen n_min
- ORM top과 raw top의 G score gap >= frozen g_min
- raw top vote share <= frozen r_max

m_max는 {0,0.0625,0.125,0.25}, n_min은 {0.125,0.1875,0.25},
g_min은 {0.25,0.5,1.0} log-score gap, r_max는 {0.40,0.50,0.60}의 고정 grid만 쓴다.
normalized margin/support의 분모는 해당 question의 valid candidate 수다. internal label/metric을 열기 전에 grid와 tie-break를
manifest에 적고, nested group CV에서 override coverage 1~15% 범위 안에서 net gain을 최대화하되 break와
wrong-to-wrong guardrail을 만족하는 조합 하나만 고른다.
동률이면 override 수가 적은 보수적 조합, 그다음 사전 등록된 lexicographic 순서를 쓴다. outer OOF나
leaderboard 결과를 보고 threshold를 움직이거나 hard/format 여부를 이용한 예외 규칙을 추가하지 않는다.

[사전 등록 arms]
- Arm A: outer-train에서 다시 학습한 T12 pointwise BCE + 기존 `n * geometric mean(score)`
- Arm B1: BCE + pairwise ranking + 기존 nGM
- Arm B2: BCE + pairwise + listwise 보조 loss + 기존 nGM
- Arm C: internal CV에서 B1/B2 중 고른 ranking ORM + learned G(a), 모든 문항에 적용
- Arm D: Arm C + 위의 selective override; raw majority가 default인 최종 후보

기존 T12 full adapter+nGM replay는 `Arm A0` 구현 진단으로만 남기며 OOF 성능표나 arm 선택에 넣지 않는다.
Arm A~D는 모두 같은 outer-test question과 같은 frozen T5 base k=16 후보 풀에서 비교한다. B1/B2와 C/D의
선택은 각 outer fold의 inner CV에서 끝내고, outer-test에는 그 fold에서 이미 선택한 정책 하나만 적용한다.

[Phase 6 — zero 없는 fallback 계약]
T12 leaderboard CSV의 3개 forced-zero를 반복하지 않는다. 모든 후보가 invalid여서 raw/filtered/ORM group이
모두 없으면 다음을 순서대로 실행한다.
1. 기존 T4c greedy prompt, temperature=0, max_new_tokens=2048로 결정적 1회 생성 후 frozen extractor 적용
2. 실패 시 동결 explicit-integer contract prompt로 temperature=0 결정적 1회 repair 생성
3. 그래도 정수 답이 없으면 submission.csv를 만들지 않고 `fallback_gate_failed`로 종료

fallback prompt와 두 request seed/config는 leaderboard prediction을 만들기 전에 freeze한다. question ID별
기존 leaderboard answer lookup, gold lookup, 임의 0/최빈 답/이전 CSV 답 복사는 금지한다. 내부 OOF와
leaderboard에서 forced-zero, null, NaN, 범위 밖 문자열은 모두 0건이어야 한다.

[Phase 7 — nested OOF 평가와 최종 개발 후보 동결]
각 outer-test fold에서 gold를 결합하기 전에 candidate logit, calibrated score, answer-group feature/G score,
raw prediction, override 여부와 이유, fallback 결과, Arm A~D prediction을 저장·해시한다. 다섯 outer fold를
합친 6,034문항 OOF에서 다음을 계산한다.
- Arm A~D accuracy와 raw majority@16 대비 paired delta
- rescue/break/wrong-to-wrong/net, exact McNemar p, paired bootstrap 95% CI
- outer 5 folds 각각의 delta와 source/template/problem type/hard/format strata
- global 및 within-question macro/median AUC, group top-1, Brier, ECE
- override coverage, 네 조건별 통과 수, invalid/tie/fallback/forced-zero
- fold별 학습/scoring 처리량, peak VRAM, OOM, 2-GPU makespan

OOF 보고가 끝나면 Phase 1의 최빈값·동률 규칙으로 최종 loss/checkpoint/calibration/coefficient/threshold 하나를
정한다. 6,034문항 전체로 최종 ORM을 한 번 학습하고 모든 구성요소와 config hash를
`frozen-dev-candidate.json`에 기록한다. OOF 결과를 본 뒤 grid를 추가하거나 규칙을 손으로 고르는 것은 금지한다.

[내부 개발 판정 — PASS 금지]
다음은 독립 채택 gate가 아니라 leaderboard 실행 비용을 쓸 만한 후보인지 정하는 개발 gate다.
1. nested OOF에서 Arm D가 Arm A와 raw majority@16을 모두 이기고, 더 강한 쪽보다 절대 +1.5pp 이상 높다.
2. outer 5 folds의 paired delta가 모두 양수다.
3. 강한 내부 baseline 대비 paired bootstrap 95% CI 하한 > 0이고 exact McNemar p < 0.05다.
4. within-question macro AUC >= 0.80이다.
5. rescue >= 20, break <= 5이며 changed questions 중 wrong-to-wrong 비율 <=25%다.
6. non-hard accuracy가 강한 내부 baseline보다 낮지 않고 hard/format strata도 2.0pp 초과 하락하지 않는다.
7. NaN/null/forced-zero=0이고 outer-test label leakage=0이다.

모두 만족하면 `DEV_CANDIDATE`, 양의 개선이지만 하나라도 실패하면 `DEV_HOLD`, delta<=0이면 `DEV_REJECT`로
기록한다. 어느 상태도 `PASS/HOLD/REJECT`로 바꿔 쓰거나 T13 승격 근거로 사용하지 않는다.

[Phase 8 — 선택적 T12b-LB one-shot handoff]
`DEV_CANDIDATE`일 때만 별도 명시적 실행 승인을 받아 원본 leaderboard 1,000문항 전체에 다음을 수행한다.
1. 동결 T12 solver 계약으로 문제당 k=32를 생성한다. 831행 필터본은 사용하지 않는다.
2. `frozen-dev-candidate.json`의 ORM, calibration, group coefficient, override, fallback을 그대로 적용한다.
3. leaderboard label·기존 제출 답·점수를 읽지 않은 상태에서 prediction과 submission candidate를 먼저 해시한다.
4. 실제 제출은 정확히 한 번 하고, 사전에 지정한 full-1,000 baseline 점수와 비교한다.
5. 점수를 본 뒤 설정을 바꾸거나 같은 T12b 가설로 재제출하지 않는다.

leaderboard는 aggregate score만 주므로 문항별 rescue/break, McNemar, bootstrap CI, fold/strata 지표를 계산할 수
없다. 결과는 `LB_GAIN`, `LB_FLAT`, `LB_LOSS`와 delta만 기록하며 `fresh-2 PASS`라고 부르지 않는다. T13 반영은
T12b-dev가 자동으로 수행하지 않고, leaderboard 결과를 본 사용자의 별도 결정 과제로 남긴다.

[조건부 7B와 k 확장]
3B T12b-dev가 nested OOF에서 within-question macro AUC와 group top-1을 개선한 뒤에도 capacity 부족이 명확할
때만 별도 T12c-dev로 Qwen 7B math same-base-policy/verifier LoRA를 비교한다. 이번 leaderboard 점수를 7B나
threshold 선택에 재사용하지 않는다. oracle pass@32가 이미 96.90%이므로 k>32 생성은 ranking/aggregation을
먼저 해결한 뒤의 별도 개발 과제이며 T12b-dev와 동시에 바꾸지 않는다.

[비파괴 구현]
새 파일로만 구현한다.
- configs/t12b_question_local_orm.json
- src/build_question_local_orm_data.py
- src/train_question_local_orm.py
- src/orm_group_selector.py
- src/orm_selective_override.py
- scripts/run_t12b_question_local_orm.sh
- tests/test_question_local_orm_data.py
- tests/test_question_local_orm_loss.py
- tests/test_orm_group_selector.py
- tests/test_orm_selective_override.py

data/cmu_orm_v2/와 artifacts/t12b_question_local_orm/만 새로 쓴다. 기존 T12 code/artifact/CSV와 root
submission.csv는 수정하지 않는다. `DEV_CANDIDATE`여도 실제 제출은 별도 명시적 T12b-LB 단계에서만 한다.

[필수 테스트]
- question batch의 positive/negative가 같은 DDP rank에 있고 다른 question pair가 섞이지 않는다.
- z_positive가 z_negative보다 커질수록 pairwise loss가 단조 감소한다.
- listwise numerator에는 positive만, denominator에는 해당 question valid candidates만 들어간다.
- source별 1:1, matched-feature SMD <=0.10, unique trace, question/template split leakage=0을 검증한다.
- candidate/row 순서와 DDP worker 완료 순서를 바꿔도 pair sampling과 out-of-fold prediction이 동일하다.
- G(a)의 alpha/beta/gamma non-negative constraint, variance penalty, tie-break가 golden fixture와 일치한다.
- outer-test label이 loss/checkpoint/coefficient/calibration/threshold fit에 들어가면 즉시 실패한다.
- override 네 조건 중 하나라도 거짓이면 raw majority를 보존한다.
- no-valid fixture에서 두 단계 fallback을 순서대로 실행하고 임의 0을 절대 쓰지 않는다.
- 기존 T12 fresh/reused IDs와 leaderboard label/score가 ORM prompt, feature, fold 선택에 들어가지 않는다.
- 6,034문항이 outer-test에 정확히 한 번씩만 나오고 template group의 fold 교차가 0이다.
- 최종 설정의 최빈값·coverage·lexicographic 동률 처리가 golden fixture와 일치한다.
- Arm A의 nGM 구현이 기존 T12 golden fixture와 byte-identical하되, fold model은 outer-train만 학습한다.
- k=16과 k=32에서 normalized margin/support 계산이 같은 비율 fixture에 대해 동일하다.

[완료 조건]
- 기존 T12 fresh/reused 결과가 diagnosis-only로 봉인되어 있다.
- T12 6,034문항의 outer 5-fold·inner 4-fold template-group split이 학습 전에 hash·freeze되고 leakage가 0이다.
- T5 base k=16 개발 추론 풀이 6,034×16의 coverage가 정확하고 균형 train row와 분리돼 있다.
- source-balanced question-local corpus, hard-negative provenance, shortcut probe가 data gate를 통과한다.
- Arm A~D의 nested OOF within-question 지표와 다섯 outer fold 결과가 함께 남아 있다.
- 최종 loss/checkpoint, group coefficient, calibration, selective override threshold의 선택 과정과 hash가 남아 있다.
- `DEV_CANDIDATE/DEV_HOLD/DEV_REJECT`가 개발 gate와 함께 report에 기록되어 있다.
- 어떤 개발 판정도 PASS나 T13 승격으로 기록되지 않았고 root submission.csv가 바뀌지 않았다.
- T12b-LB를 별도로 실행했다면 full 1,000 label-blind prediction hash, 사전 지정 baseline, 단 한 번의 점수와
  `LB_GAIN/LB_FLAT/LB_LOSS`만 기록되어 있으며 그 점수로 재튜닝하지 않았다.

[산출물]
configs/t12b_question_local_orm.json
data/cmu_orm_v2/
  internal-folds.json
  train.jsonl
  train-manifest.json
  dev-candidate-pool-manifest.json
artifacts/t12b_question_local_orm/
  input-verification.json
  source-balance-audit.json
  shortcut-probes.json
  nested-cv/fold-*/
  out-of-fold-candidate-scores.jsonl
  out-of-fold-group-scores.jsonl
  out-of-fold-predictions.jsonl
  internal-arm-comparison.json
  group-selector.json
  calibration.json
  selective-override-policy.json
  adapter/
  frozen-dev-candidate.json
  leaderboard-label-blind/generations.jsonl          (T12b-LB 승인 시)
  leaderboard-label-blind/candidate-scores.jsonl     (T12b-LB 승인 시)
  leaderboard-label-blind/group-scores.jsonl         (T12b-LB 승인 시)
  leaderboard-label-blind/predictions.jsonl          (T12b-LB 승인 시)
  leaderboard-label-blind/submission-candidate.csv   (T12b-LB 승인 시)
  leaderboard-one-shot-result.json                    (점수 확인 시)
  evaluation.json
  evaluation.md
  manifest.json
  tests.xml
```

---

## T13 — 동결 · 제출 리허설 · runbook

```text
[목표]
최종 테스트 당일에 판단할 일이 하나도 없게 만든다. 당일에는 정해진 명령만 실행한다.

[작업]
1. 전면 동결
   채택 모델(어댑터가 있으면 어댑터, 없으면 base revision), 프롬프트, 생성 설정, 추출기, 투표 규칙을
   모두 고정하고 해시를 기록한다. 2026-08-27 기준 public leaderboard 최고 경로는
   T10a C-1 filtered majority@32이고, 로컬 후보는 T10e arm-normalized filtered voting@96이며
   T10d를 폴백으로 보존한다. T11이 사전 등록 게이트를 통과하면 T11 adapter+C prompt+C-1 k32를
   단일-model 제출 경로로 승격하고, 통과하지 못하면 기존 경로를 유지한다.
   T11d extractor/prompt는 reused T8 replay나 format canary만으로 넣지 않고, 별도로 동결한 fresh
   validation까지 통과했을 때만 source/config hash와 함께 최종 경로에 반영한다.
   T12 pointwise ORM은 fresh validation에서 HOLD로 확정되었으므로 ORM adapter와 geometric weighted
   majority@32를 최종 환경에 로드하지 않는다.
   T12b-dev question-local ORM의 nested CV 결과는 내부 개발 성능이므로 T13에 자동 반영하지 않는다.
   별도 T12b-LB one-shot 결과를 확인한 뒤 사용자가 별도 채택 과제로 명시적으로 결정하기 전까지 기존
   T10a C-1/T10e 경로를 유지하며, ranking adapter·coefficient·threshold 일부만 따로 가져오지 않는다.
   T9 GenSelect와 선택 전용 LoRA는 미채택이므로 최종 추론 경로에 포함하지 않는다.
   이 시점 이후 코드를 바꾸지 않는다.

2. src/submit.py 검증
   - 입력 파일에서 컬럼명을 읽어 그대로 따른다. 하드코딩하지 않는다
     (리더보드 CSV의 세 번째 열은 "answer"가 아니라 앞에 공백이 붙은 " answer"다.
      최종 테스트 파일은 컬럼명이 id일지 ID일지, CSV일지 parquet일지 확정되지 않았다.
      Overview/Rules는 test.parquet, Data 페이지는 CSV 3종으로 표기가 다르다.
      공개된 sample submission을 무조건 우선한다)
   - 모든 행에 정수 하나를 채운다. 추출 실패 시에도 빈 값을 두지 않고 채택 경로의 동결 fallback을 쓴다.
     별도 채택 결정으로 T12b 경로를 쓰는 경우에도 임의 0을 쓰지 않으며 두 단계 결정적 fallback까지
     실패하면 파일을 만들지 않는다.
   - 제출 전 자동 검증: 행 수가 입력과 일치, ID 누락 0, ID 중복 0, 모든 값이 ^-?(?:0|[1-9][0-9]*)$ 만족
   - 검증 실패 시 파일을 쓰지 않고 중단한다

3. 리더보드 1,000문항 전체로 전체 리허설
   831행 필터본이 아니라 원본 1,000행 전체를 쓴다. 필터본을 쓰면 169문항이 누락된다.
   실제 소요 시간을 재고 24시간 예산과 대조한다. T10e를 쓰려면 세 arm 전체를,
   T11을 채택했다면 동결 adapter+C-1 k32 경로를 그대로 리허설한다. 별도 채택 과제에서 T12b-LB 후보를
   명시적으로 선택했다면 base k32 생성부터 ranking ORM score, calibration, group selector, selective override,
   두 단계 fallback까지 `frozen-dev-candidate.json` 묶음 그대로 리허설한다. T12의 단일 호스트 2× RTX 4090
   generation/score shard 계약을 유지하고 한 GPU 직렬 폴백이나 tensor parallel로 경로를 바꾸지 않는다.

4. 백업
   - T11 adapter를 채택했다면 가중치와 base revision을 원격 밖으로 한 벌 더 복사한다.
     별도 채택 과제에서 T12b-LB 후보를 선택했다면 ranking ORM adapter, classifier score head, calibration, group coefficient,
     override policy, fallback prompt/config도 함께 복사한다.
     미채택이면 base revision, 최종 prompt/config, T10d/T10e 집계 규칙과 source pool hash를 백업한다.
   - 환경 스냅샷(requirements.lock, 모델 revision)을 함께 보관한다.
   - 예비 체크포인트를 2종 확보한다: 최종 채택안과, 더 단순하지만 동작이 검증된
     artifacts/submissions/t10a_c1_filtered_k32/ 폴백.

5. runbook.md 작성
   최종일에 실행할 명령을 순서대로 적는다. 각 단계의 예상 소요 시간과 실패 시 대응을 함께 적는다.
   GPU 인스턴스가 사라졌을 때의 복구 절차도 포함한다.

6. 재현 패키지 정리 (수상 후보자 의무)
   학습 코드, 추론 코드, 채택 adapter, 사용 데이터셋·teacher model/revision·접근 방법,
   하드웨어/라이브러리 버전, 방법론 문서(아키텍처, 전처리, 하이퍼파라미터)를 정리한다.
   외부 데이터와 teacher를 사용했다면 라이선스·비용·필터·full leaderboard 오염 감사도 명시한다.

[완료 조건]
- submission.csv가 1,000행, ID 누락·중복 0, 전량 정수로 생성되었다.
- 전체 리허설 소요 시간이 24시간에서 최소 6시간 여유를 남긴다.
- base revision·최종 prompt/config·채택 adapter 또는 집계 source pool이 원격 밖에 백업되어 있다.
- T11 채택 여부와 최종 경로가 runbook·manifest·백업에서 일치한다.
- T11d 채택 여부, extractor source hash와 prompt/config hash가 runbook·manifest·백업에서 일치한다.
- T12 HOLD, T12b-dev 내부 판정, T12b-LB 실행 여부, 별도 T12b 채택 결정 여부가 각각 명시되어 있다.
  별도 채택 결정이 있었다면 ORM adapter/config hash, calibration, group coefficient, override/fallback policy와
  2× RTX 4090 shard/DDP 계약이 runbook·manifest·백업에서 일치한다.
- T9 GenSelect adapter는 미채택으로 명시되어 있고 최종 환경에서 로드되지 않는다.
- runbook.md만 보고 처음 보는 사람이 실행할 수 있다.

[산출물]
src/submit.py, runbook.md, submission.csv (리허설본), artifacts/t13_rehearsal/manifest.json
```

---

## 발표 자료용 누적 기록

각 작업이 끝날 때마다 다음 표의 해당 행을 채운다. 최종 수상은 모델 성능 50% + 발표 평가 50%이므로,
이 표 자체가 리더보드 1~2pp보다 배점상 더 큰 산출물이다.

판정 통계는 **합집합 3,737문항의 짝지은 McNemar**다. split별 정확도는 진단용으로만 읽는다.
random holdout(N=1637) 단독은 최소 검출 가능 효과가 약 2.4pp라서 그보다 작은 변화를 구분하지 못한다.
합집합(N=3737)은 약 1.6pp까지 내려간다. T6-4의 "+0.86pp"가 p=0.363이었던 것이 이 열을 만든 이유다.

| 단계 | random | template | hard | format | invalid | 합집합 Δ vs T4c (p) | 비고 |
|---|---|---|---|---|---|---|---|
| base (T3) | 64.20% | 64.63% | 28.73% | 32.03% | 15.03% (random) | — | greedy·1024 tokens·엄격 B0 추출기; vLLM 13.24 gen/s, GPU 99.3%, seed 42 바이트 동일 |
| + fallback 추출 (T4b) | 67.07% | 66.16% | 30.91% | 38.28% | 0.49% (random) | — | 동일 T3 생성 바이트 재파싱·추가 GPU 생성 0건; random +2.87pp, invalid -14.54pp vs base |
| + 2048 토큰 (T4c) | 67.99% | 66.95% | 32.55% | 46.88% | 0.61% (random) | 기준 (62.70%) | 최종 채택; random +3.79pp vs base(+0.92pp vs T4b), format invalid 1.95%, random hit-max 2.81%, GPU 99.6%, OOM 0 |
| answer-only SFT (T6-2) | 23.21% | 22.48% | 14.36% | 18.75% | 0.06% (random) | **-40.81pp** (p≈0) | CoT 가치 증명용 대조군; 평균 출력 8.2 tokens, 파손·오답 라벨 포함으로 데이터 품질상 불리 |
| 외부 CoT SFT (T6-3) | 60.84% | 61.33% | 21.64% | 37.50% | 0.79% (random) | **-7.06pp** (p=2e-24) | 평균 출력 439.9 tokens; hit-max random 2.81%→8.00%·format 5.08%→22.66%로 종료 실패 |
| RFT SFT (T6-4) | 68.85% | 67.50% | 31.45% | 41.80% | 0.55% (random) | +0.40pp (**p=0.517**) | 유의하지 않음; 467문항이 뒤집히고 순증 15문항. hard -1.09pp·format -5.08pp |
| RFT + 외부 (T6-5) | 67.93% | 67.62% | 29.45% | 40.62% | 0.49% (random) | -0.48pp (**p=0.449**) | 본안 미채택·T4 base 유지; 504문항 뒤집힘. 사후 분석은 T6-1 참조 |
| RFT-v2 SFT (T6-1 A) | 67.93% | 67.81% | 32.55% | 42.97% | 0.49% (random) | +0.27pp (**p=0.679**) | 보류·T4 base 유지; 0.75 epoch(val 48.8%) 선택, 472문항 뒤집힘, 95% CI [-0.87,+1.41]pp. format -3.91pp로 추가 게이트도 실패 |
| RFT-v2 + 외부 (T6-1 B) | 59.19% | 59.99% | 24.73% | 35.16% | 0.79% (random) | **-8.30pp** (p=5.30e-32) | 기각·T4 base 유지; 712문항 뒤집힘, 95% CI [-9.67,-6.92]pp. hard -7.82pp·format -11.72pp |
| ~~SFT-v2 (T7)~~ | — | — | — | — | — | — | **미실행 (2026-08-22 결정)** — T6-1 A 보류·B 기각으로 채택 어댑터 없음. T7을 데이터 품질 감사로 축소하고 GPU 예산을 T8·T9로 이전. 근거는 T7의 「SFT-v2 학습을 잘라낸 이유」 |
| + maj@k (T8) | 74.28% | 73.98% | 39.64% | 50.39% | 0.53% (random) | **+6.61pp** (**p=2.49e-36**) | fixed k=32 채택; 합집합 69.31%, pass@32 84.40%, agreement@32 70.44%, tie 4.66%, 1,000문항 0.840h. adaptive 4→32는 평균 k=18.12에서 동일 예산 고정 대조군보다 +0.75pp였으나 fixed k=32보다 0.027pp 낮아 미채택 |
| + RFT maj@k (T8-1) | 75.08% | 74.22% | 39.45% | 53.52% | 0.52% (random) | **+7.14pp** (**p=1.70e-38**) | 보류·T4c + T8 유지; fixed k=32 합집합 69.84%, 자체 greedy 대비 +6.74pp(p=1.14e-41). 기존 T8 대비 +0.54pp(95% CI [-0.29,+1.36], p=0.228)로 +1.5pp 채택 게이트 미달; hard -0.18pp·format +3.13pp로 guardrail 통과. staged adaptive는 평균 k=17.48·합집합 69.47%로 fixed k=32보다 -0.37pp여서 미채택; 1,000문항 0.635h, GPU 99.65%, OOM 0 |
| + disagreement-routed CoT (T8-2) | 74.71% | 73.67% | 39.82% | 53.12% | 0.43% (random) | **+6.61pp** (**p=5.16e-35**) | 기각·T8 fixed k=32 유지; primary C 합집합 69.31%, T8 대비 +0.00pp(95% CI [-0.70,+0.70], 89개 개선/89개 악화, p=1.0). strong-CoT fixed B는 69.52%, T8 대비 +0.21pp(95% CI [-0.54,+0.96], p=0.626)인 ablation only. hard +0.18pp·format +2.73pp·합집합 invalid -0.13pp로 guardrail은 통과했지만 효과 게이트 실패. staged 1,000문항 1.042h, strong pool 8.58 gen/s·GPU 99.95%·OOM 0 |
| + vote-quality filter (T8-3) | 75.57% | 75.50% | 40.73% | 53.12% | 0.07% (filtered random pool) | **+8.08pp vs T4c**; **+1.47pp vs T8** (**p=6.80e-10**) | 보류·T8 fixed k=32 유지; 합집합 70.78%, 회수 69/파손 14, 95% CI [+1.00,+1.95]pp. 5-fold 모두 양수(+0.80~+2.52pp)였으나 +1.5pp 채택 게이트에 0.028pp 미달. 새 생성·학습 0건, 831행 후보 제출본은 34건 변경·폴백 5건이며 라벨 정확도는 계산하지 않음 |
| + RFT maj@k + vote-quality filter (T8-4) | 75.38% | 74.71% | 40.36% | 54.69% | 0.01% (filtered random pool) | **+7.60pp vs T4c**; **+0.99pp vs T8** (**p=0.0245**) | 보류·T8 fixed k=32 유지; 합집합 70.30%. RFT 무필터 대비 +0.455pp(회수 19/파손 2, p=2.21e-4)였지만 현재 T8 대비 +1.5pp 게이트 실패. T8-3보다 -0.482pp(p=0.262). 새 생성·학습 0건, 831행은 17건 변경·폴백 0건이며 라벨 정확도는 계산하지 않음 |
| + GenSelect (T9) | 71.17% | 70.74% | 35.27% | 44.92% | 0.00% (random) | **+3.10pp** (**p=5.03e-8**) | 미채택·T8 fixed k=32 유지; 엄격 동일 예산 few-shot 28풀이+4선택 합집합 65.80%, T8 maj@32 대비 -3.51pp (p=4.22e-17). 학습 어댑터 full32 55.90%로 few-shot full32 65.40%보다 -9.50pp; 출력 125.9 tokens, 셔플 답 일치 58.98% |
| + prompt improvement (T10a C) | 75.44% | 73.18% | 38.36% | 53.52% | 1.04% (random) | **+0.05pp vs T8** (**p=0.942**) | 보류·T8 base prompt 유지; 핵심 cot-boxed C 합집합 69.36%, 95% CI [-0.67,+0.78]pp. boxed +10.25pp였지만 final marker -11.12pp·last_integer +0.37pp·필터 대상 +1,142건. 재사용 B는 +0.21pp(p=0.626), D는 -0.45pp(p=0.254). C/D 각 119,584건, 9.34/9.17 gen/s, OOM 0; T10b base는 기존 T8 프롬프트 |
| + prompt improvement + vote-quality filter (T10a C-1) | 76.05% | 74.71% | 39.64% | 54.30% | 0.18% (filtered random pool) | **+7.84pp vs T4c**; **+1.23pp vs T8** (**p=0.00256**) | 보류·T8 base majority@32 유지; 합집합 70.54%. C 무필터 대비 +1.18pp(회수 61/파손 17, p=5.66e-7, 95% CI [+0.72,+1.66])였지만 +1.5pp 게이트 실패. 동일 필터의 T8-3보다 -0.24pp(p=0.557). 새 생성·학습·정책 탐색 0건; 동일 holdout 조합 진단 |
| + prompt diversity (T10b C) | 74.22% | 73.30% | 38.36% | 50.78% | 0.28% (random) | **-0.51pp vs T8** (**p=0.195**) | 기각·T8 단일 prompt majority@32 유지; 합집합 68.80%, 87개 회수/106개 파손, 95% CI [-1.23,+0.21]pp. agreement@32는 70.44%→62.97%로 낮아졌지만 정확도 향상으로 이어지지 않음. T8-3 필터 적용 C는 70.19%로 A+필터보다 -0.59pp(p=0.137). 119,584건, 9.56 gen/s, 1,000문항 0.929h, OOM 0; T10c 입력은 기존 T8 pool |
| + extraction-path weighted vote (T10c P2) | 75.26% | 75.26% | 40.73% | 51.95% | 0.53% (random source pool) | **+7.79pp vs T4c**; **+1.18pp vs T8** (**p=6.21e-8**) | 보류·T8 단일 prompt majority@32 유지; 연속 정책 최선 P2 합집합 70.48%, 회수 56/파손 12, 95% CI [+0.75,+1.61]pp로 +1.5pp 게이트 미달. P1은 T8-3 바이트 동일 재현(70.78%); P2는 P1보다 -0.29pp(p=0.0801). 가중 tie 2.44%, 폴백 1건, 5-fold 모두 양수·과적합 신호 없음, 새 생성 0건. 최종 T10은 원래 T8과 동일 |
| + 3-view flat filtered majority@96 (T10d) | 76.18% | 75.38% | 41.82% | 55.08% | 0.00% (final predictions) | **+8.48pp vs T4c**; **+1.87pp vs T8** (**p=8.02e-8**) | 규정 확인 대기 후보; 합집합 71.18%, 회수 120/파손 50, 95% CI [+1.20,+2.54]pp로 기존 수치 게이트 통과. T8-3 대비 +0.40pp(p=0.203)는 유의하지 않음. 5-fold 전부 양수, 새 생성·학습 0건. 831행 제출본 SHA-256 `4caa701c...276e`; 동일 베이스 96표와 동결 품질 필터의 운영진 서면 확인 전에는 독립 채택으로 승격하지 않음 |
| + 3-view arm-normalized filtered voting@96 (T10e) | 76.18% | 75.63% | 42.18% | 54.69% | 0.00% (final predictions) | **+8.59pp vs T4c**; **+1.98pp vs T8** (**p=4.12e-8**) | 사용자 요청 제출 후보·T10d 폴백 보존; 합집합 71.29%, 회수 128/파손 54, 95% CI [+1.28,+2.68]pp. T10d flat 대비 +0.11pp(15/11, p=0.557, 95% CI [-0.16,+0.37])로 증분은 유의하지 않고 5-fold 중 1개가 음수. 필터된 리더보드 831행만 사용, T10d와 답 9건 변경·동점 3건·invalid 0건, 제출 SHA-256 `19920b90...6d8` |
| hard-CoT SFT → correct/wrong DPO (T11) | — | — | — | — | — | — | **teacher_gate_failed·학습 미실행**; 보호 5,475 ID를 제외하고 full leaderboard 1,000행 오염 대조에서 37문항(template 22/near 15)을 추가 제외해 10,861문항을 probe했다. frozen base+C k=8에서 c<=2 hard는 1,883문항, c>=6 strict anchor 후보는 7,382문항이었다. 고정 local `Qwen2.5-Math-7B-Instruct` teacher의 첫 64문항×4에서 품질 필터 전 extracted-correct는 76/256이었으나 `FINAL_ANSWER` 마지막 줄 위반 256/256, hit-max 42, code/tool 표현 7로 accepted correct가 0/256이었다. 사전등록대로 full teacher·SFT·DPO·validation·holdout은 모두 중단(새 holdout 0건), API 비용 $0, projected worst-case teacher 2.49h였으며 기존 T10a C-1/T10e를 보존했다. AIMO 대비 3B·단일 GPU·hard 최대 2,000·teacher k=4→조건부 8·text-only로 축소했고 TIR/Python/SymPy/solver, k>=64, 새 prompt/filter 탐색, 사후 checkpoint 선택은 금지했다. |
| DeepSeek-14B teacher preflight (T11b) | — | — | — | — | — | — | **teacher_gate_failed·full teacher/SFT/DPO 미실행**; 동일 64문항×4·seed 62000·prompt/sampling에서 `DeepSeek-R1-Distill-Qwen-14B` revision `1df8507...23761`을 vLLM 0.27.1+cu129/bitsandbytes 4-bit로 실행했다. raw→label-blind normalized는 final-line 44→81, accepted-quality 44→72, accepted-correct 17→23, 정답 trace 보유 문항 10→12/64였고 extracted-correct는 36/256으로 동일했다. 원래 gate의 문항 12<32, trace 23<64, code/tool 2>0, projected full 28.51h>12h가 실패했다. hit-max 173, preflight 0.494h, peak VRAM 24,082 MiB, 보호 전송 0, API $0; 폴백·holdout·leaderboard·submission 변경 없이 기존 T10a C-1/T10e를 유지했다. |
| Qwen7B repaired teacher preflight (T11c) | — | — | — | — | — | — | **teacher_gate_failed·full teacher/SFT/DPO 미실행**; 새 hard slice 64문항에 공식 Qwen CoT prompt, temperature 0.7/top_p 0.8, 3,072-token cap, 요청별 독립 seed를 고정했다. 1차 accepted-correct는 54 traces·27/64문항, 미해결 37문항에만 sample 4..7을 추가한 최종은 56 traces·29/64문항이었다. raw code/tool 2, hit-max 63, preflight 0.791h, expected/worst full 22.42/28.41h, API $0; label-blind normalizer 2회 byte 동일, 보호 전송·downstream·submission 변경 0건. |
| + explicit integer contract + frozen replay (T11d) | 75.57% | 75.75% | 40.55% | 50.78% | 9.34% (union candidates) | **+8.03pp vs T4c**; **+1.42pp vs T8** (**p=2.88e-7**) | **reused_holdout_diagnostic·T13 미채택**; old T8 2,590/3,737 plurality와 3,154/3,737 pass를 먼저 재현한 뒤 80문항 회수/27문항 파손, 순증 53, 95% CI [+0.91,+1.95]pp를 기록했다. zero-decimal explicit 1,185표·86문항을 정규화했고 train-012155는 50/400 동률→400 만장일치가 됐다. 반면 strict explicit barrier로 invalid 0.77%→9.34%, pass@32 84.40%→82.98%가 되어 fresh validation 전 채택하지 않는다. 별도 train 128문항에서 원격 RTX 4090으로 A/B 각 512개를 동일 paired child seed로 label-blind 생성·freeze했다. prompt B는 strict final-line 64.06%→72.07%(+8.01pp, paired p=0.0035), invalid 4.88%→2.93%, hit-max 0.78%→0.98%였지만 last_integer가 1.17%→4.69%로 증가해 사전등록 gate에 실패했다. 따라서 prompt B는 미채택하고 기존 T8 prompt를 유지한다. 집중 테스트 71건 통과, 기존 artifact·leaderboard·submission 변경 0건. |

### 데이터 품질 감사 기록 (별도 슬라이드)

주최 측이 준 필터 파일을 그대로 쓰지 않고 성격을 검증했다는 것 자체가 발표 항목이다.
아래는 실행 중 자연히 나오는 값들이므로 따로 실험할 필요가 없다.

| 항목 | 값 | 출처 |
|---|---|---|
| 대회 측 제외 627행 중 이미지 표지 없음 | 270행 (43.1%) | 확정 — 2026-08-20 재검증 |
| RFT R1 c 분포 (c=0 / 1 / 2~3 / >=4) | 1,801 / 534 / 677 / 9,624 (pool 12,636) | 확정 — T5 metrics.json |
| RFT R1 학습 가중치가 c>=4에 쏠린 비율 | 40,645행 중 94.7% (c=1~3은 5.3%) | 확정 — T6 사후 분석 |
| pass@16 85.7% vs greedy 68% 갭 | 17.7pp — RFT가 노려야 할 실제 헤드룸 | 확정 — T5 audit + T4c |
| packing attention 격리 부재 (pack당 4.6샘플) | 학습 샘플 약 78%가 무관한 앞 문제를 보며 학습됨 | 확정 — T6 사후 분석 |
| 기존 RFT adapter 정밀도 프로브 | NF4 67.20% / bf16 68.85%, 절대차 1.65pp → bf16 LoRA | 확정 — T6-1 precision-probe.json |
| RFT-v2 재가중 후 c=1~3 학습 비중 | 1,978 / 6,588행 = 30.02% | 확정 — T6-1 manifest.json |
| RFT-v2 assistant 토큰 길이 | median 493 / p95 1,501.95 / max 1,971 | 확정 — T6-1 manifest.json |
| R2 추가 수확 문항 수 / 행 수 | 486문항 / 911행 (1,801문항의 26.99%; T9 기준 100문항 이상) | T7 manifest.json — T9 최우선 층으로 사용 |
| **의심 집합 크기 (0/48)** | 1311문항 (canonical 16,373의 8.01%; R1 c=0 1,801문항의 72.79%) | T7 — canonical 잔존 파손·오답 비율의 하한 대리 지표 |
| 의심 집합 20개 표본 분류 (파손/오답/고난도) | 파손 4 / 오답 9 / 고난도 7; 파손+오답 13/20 → 1,170.65문항 (canonical 7.15%), Wilson 95% 779.57–1,474.67문항 | T7 sample20_review.md — n=20이라 구간이 넓음 |
