# Phase 0 결과 보고서

실행일: 2026-08-03 (KST)  
실제 원격 작업 경로: `/workspace/university-deep-learning-challenge-2026`  
실험 ID: `p0_20260803T102000Z_env-smoke_aa8e7253_s42`

## 결과

로드맵의 Phase 0(규칙·환경 고정) 작업과 종료 조건을 모두 충족했다. Phase 1의 split 생성이나 baseline 전체 평가는 수행하지 않았다.

- 고정 model/tokenizer: `Qwen/Qwen2.5-3B-Instruct`
- 공통 revision: `aa8e72537993ba99e69dfaafa59ed015b17504d1`
- 공통 seed: `[42, 2026, 3407]`
- 성공한 online smoke 2회의 전체 생성 텍스트와 추출 답이 동일했다.
- offline cache-only reload의 생성 텍스트와 추출 답도 online 결과와 동일했다.
- 원본 train/leaderboard CSV의 전후 SHA-256이 동일했다.
- focused tests 8개와 통합 verification 7개가 모두 통과했다.

## 입력 snapshot과 Git 상태

- 로컬 archive: `C:\tmp\university-deep-learning-challenge-2026_20260803_191039_KST.tar.gz`
- 원격 archive: `/workspace/incoming/university-deep-learning-challenge-2026_20260803_191039_KST.tar.gz`
- 로컬·원격 SHA-256: `7a353d48c33a49676e2c52aa5651e065c09fd7720fb479a45f24887a7481cd8a`
- archive 항목 수: 426 (`.git` 포함)
- Git HEAD: `f7ba0809bf18697ee9d3f8f563c7796fab26fc75`
- 원격에는 로컬 전역 설정과 같은 `core.autocrlf=true`를 repository-local로 설정해 snapshot의 Git 상태를 정확히 재현했다.
- 초기 수정·untracked 파일은 모두 보존했다. stage, commit, branch 생성, push는 수행하지 않았다.

## 설치 전 원격 환경

| 항목 | 값 |
|---|---|
| Vast image | PyTorch (`vast-ai/base-image` derivative) |
| workspace volume | `false` — recycle/destroy 시 소실 |
| OS | Ubuntu 24.04.4 LTS |
| kernel | Linux 6.8.0-60-generic |
| Python | 3.12.13 (`/venv/main/bin/python`) |
| GPU | 1 × NVIDIA GeForce RTX 4090 |
| GPU UUID | `GPU-a9b76bc8-5409-fc15-8eba-2e8a0192cf30` |
| VRAM | 24,564 MiB |
| driver / CUDA | 570.133.07 / 12.8 |
| PyTorch | 2.11.0+cu128 |
| CPU visibility | AMD EPYC 7B13, 64 cores / 128 logical CPUs |
| RAM visibility | 270,055,575,552 bytes; Vast live manifest는 128,275,447,808 bytes를 보고 |
| `/workspace` before | 322,122,547,200 total / 321,926,893,568 free bytes |
| `/workspace` after final verification | 322,122,547,200 total / 315,258,732,544 free bytes |

전체 설치 전 manifest는 실험 artifact의 `environment.preinstall.json`에 보존했다. 서로 다른 RAM/CPU 표시는 컨테이너의 OS 가시 범위와 Vast live allocation 지표를 둘 다 보존한 결과다.

## 고정 패키지

| 패키지 | 버전 |
|---|---|
| PyTorch | `2.11.0+cu128` |
| Transformers | `5.14.1` |
| Accelerate | `1.14.0` |
| PEFT | `0.20.0` |
| TRL | `1.9.2` |
| bitsandbytes | `0.50.0` |
| huggingface-hub | `1.18.0` |
| safetensors | `0.8.0` |
| tokenizers | `0.22.2` |

설치 명령:

```bash
uv pip install --python /venv/main/bin/python \
  transformers==5.14.1 accelerate==1.14.0 peft==0.20.0 trl==1.9.2 \
  bitsandbytes==0.50.0 safetensors==0.8.0 tokenizers==0.22.2 \
  huggingface-hub==1.18.0
```

처음에는 PyPI 최신 `tokenizers==0.23.1`을 명시했으나 Transformers의 `<=0.23.0` 요구와 충돌해 실제 설치 전에 resolver가 중단했다. 존재하는 호환 배포본 `0.22.2`를 dry-run으로 확인한 후 위 명령으로 설치했다. 실패와 성공 로그를 모두 보존했다. `uv pip check`와 핵심 package import 검사는 통과했다.

## 모델 고정과 cache

Hugging Face 공식 model API가 반환한 commit SHA를 model과 tokenizer에 동일하게 사용했다. 다운로드와 모든 `from_pretrained` 호출에서 전체 SHA를 명시했다.

```bash
hf download Qwen/Qwen2.5-3B-Instruct \
  --revision aa8e72537993ba99e69dfaafa59ed015b17504d1 \
  --cache-dir /workspace/.cache/huggingface
```

cache snapshot은 `/workspace/.cache/huggingface/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1`이며 약 5.8 GiB다. 다른 모델은 다운로드하거나 로드하지 않았다. 모델 cache와 weights는 저장소와 결과 archive에서 제외한다.

## smoke inference 재현

리더보드나 최종 테스트가 아닌 책 개수에 관한 짧은 synthetic 문제를 사용했다. 세 실행 모두 seed 42, `do_sample=False`, 최대 192 new tokens, BF16, 동일 prompt와 동일 revision을 사용했다.

| 실행 | 모드 | 추출 답 | input/output tokens | generation latency |
|---|---|---:|---:|---:|
| run 1 | online-capable | `7` | 73 / 65 | 3.0615 s |
| run 2 | online-capable | `7` | 73 / 65 | 3.0298 s |
| offline | `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `local_files_only=True` | `7` | 73 / 65 | 3.0265 s |

세 실행의 전체 생성 텍스트는 동일했다. 첫 진단 실행에서는 marker가 설명 마지막 문장에 이어 출력되어 줄 시작 anchor가 실패했다. 실패 raw JSON/log를 보존하고, 계산을 추가하지 않은 채 명시적 `FINAL_ANSWER:` marker의 위치만 읽도록 순수 구문 extractor를 수정했다.

## 데이터 보호와 규칙 준수

| 원본 | 작업 전 SHA-256 | 작업 후 SHA-256 |
|---|---|---|
| `data/deep_chal_math_train.csv` | `94f3302a6240b91b6fb3d093696b898750b8c4ca1d8ae1eb54210358664af9df` | 동일 |
| `data/deep_chal_math_leaderboard.csv` | `f00b83805479140fb4d59fedb01c092e16c6cd35ac588f387b281ffea55eb2d7` | 동일 |

- 리더보드 질문을 외부 API, 웹 검색, 외부 모델 또는 외부 서비스에 전송하지 않았다.
- smoke inference는 synthetic prompt만 사용했다.
- 모델 생성 코드 실행, Python/SymPy 계산 피드백, solver, 계산 verifier와 retrieval을 구현하거나 사용하지 않았다.
- 답 extractor는 모델이 쓴 명시적 marker 뒤의 텍스트만 읽는다.
- `deep_chal_math_leaderboard_filtered.csv`를 학습 또는 평가 범위 축소에 사용하지 않았다.

## 생성 산출물과 재실행

주요 재실행 명령:

```bash
source /venv/main/bin/activate
python scripts/collect_environment.py --repo-root . --output /tmp/environment.json
python -m unittest -v tests/test_phase0_environment.py tests/test_phase0_smoke_inference.py
python scripts/phase0_smoke_inference.py \
  --revision aa8e72537993ba99e69dfaafa59ed015b17504d1 \
  --cache-dir /workspace/.cache/huggingface --seed 42 \
  --output /tmp/smoke.json
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/phase0_smoke_inference.py \
  --revision aa8e72537993ba99e69dfaafa59ed015b17504d1 \
  --cache-dir /workspace/.cache/huggingface --seed 42 --offline \
  --output /tmp/smoke-offline.json
```

상세 manifest, 원시 출력, 로그, package freeze와 SHA-256 목록은 `artifacts/experiments/p0_20260803T102000Z_env-smoke_aa8e7253_s42/`에 있다.

## 로드맵 반영

Phase 0 작업 체크박스 7개는 각각 config, manifest, checklist, 환경 수집 결과와 smoke verification 근거가 있어 완료 처리했다. 종료 조건 세 가지도 충족했다. Phase 1 항목은 변경하지 않았다.

현재 blocker는 없다. 단, `/workspace`가 persistent volume이 아니므로 인스턴스를 recycle/destroy하기 전에 결과 archive를 외부로 내려받아야 한다.
