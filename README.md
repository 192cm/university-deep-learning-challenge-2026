<div align="center">

# 🧠 아주 소중한 딥러닝 챌린지 2026

**수학 추론 실험 파이프라인** — `Qwen/Qwen2.5-3B-Instruct`를 미세조정하고 정수 답 제출까지 검증합니다.

<br>

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![vLLM](https://img.shields.io/badge/vLLM-0.27.1-5E5CE6?style=flat-square)](https://docs.vllm.ai/)
[![Hugging Face](https://img.shields.io/badge/Base-Qwen2.5--3B--Instruct-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

<br>

[기능](#기능) · [빠른 시작](#빠른-시작) · [사용법](#사용법) · [설정](#설정) · [구조](#구조) · [모델 및 평가](#모델-및-평가) · [의존성](#의존성)

</div>

이 저장소는 대학 연합 딥러닝 챌린지에서 처음 보는 수학 문제의 정수 답을 추론하기 위한 실험 코드와 재현 기록을 관리합니다. 데이터 구축, 모델 생성, 미세조정, 후보 선택, 평가, 제출 파일 검증을 단계별 실험으로 분리하고 각 단계의 설정·해시·지표를 `manifest.json`에 남깁니다.

## ✨ 기능

- **재현 가능한 데이터 구축** — 원본 데이터에서 canonical train, 고정 holdout, RFT pool을 SHA-256 기반으로 결정론적으로 생성합니다.
- **검증 가능한 생성 파이프라인** — Hugging Face 또는 vLLM으로 label-blind 생성 결과와 실행 메타데이터를 resume 가능한 JSONL로 저장합니다.
- **문자열 기반 정수 추출** — `FINAL_ANSWER:`와 `\boxed{}` 등을 읽고 표기만 정규화하며, 추론 중 계산이나 수식 동치 판정을 수행하지 않습니다.
- **단계별 학습 실험** — RFT 데이터 구축, QLoRA SFT, DPO, self-consistency, 후보 선택, ORM 기반 점수화를 각각 고정된 설정으로 비교합니다.
- **제출 전 안전장치** — exact-match 평가, 다수결·가중 투표, ID coverage·중복·정수 형식 검증과 실험별 audit을 제공합니다.

---

## 🚀 빠른 시작

전체 학습과 모델 생성은 Linux/CUDA 환경을 기준으로 합니다. CPU에서도 데이터 계약과 대부분의 단위 테스트를 실행할 수 있지만, 생성·학습 단계에는 GPU와 로컬 모델 캐시가 필요합니다.

### 환경 준비

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock pytest
```

### 모델 및 런타임 설정

실험 설정은 모델과 tokenizer를 `Qwen/Qwen2.5-3B-Instruct` revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`로 고정합니다. 모델을 설정 파일의 Hugging Face 캐시에 준비한 뒤, 재현 실행에서는 다음 환경 변수를 사용합니다.

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

외부 API credential은 최종 로컬 추론·평가에 사용하지 않습니다. T12/T12b ORM 경로는 설정상 NVIDIA GeForce RTX 4090 2장이 필요합니다.

### 테스트 실행

```bash
python -m pytest -q
```

---

## 📖 사용법

### 데이터 재생성

T2 설정은 원본 CSV를 수정하지 않고 canonical 데이터와 고정 holdout을 생성합니다. `--verify-reproducibility`를 지정하면 독립적인 두 번의 materialization 결과가 같은지 확인합니다.

```bash
python -m src.build_data \
  --config configs/t2_data.json \
  --verify-reproducibility
```

주요 결과는 `data/canonical/`, `data/splits/`, `data/rft_pool_ids.txt`, `data/answer_only/`에 기록됩니다.

### 베이스라인 생성 및 평가

아래 예시는 고정 random holdout에서 vLLM greedy baseline을 생성하고 exact-match 지표를 계산합니다. `configs/*.json`에 기록된 `/workspace/.hf_home` 캐시와 CUDA 런타임을 사용할 수 있는 환경에서 실행해야 합니다.

```bash
mkdir -p artifacts/local-baseline

python -m src.generate \
  --config configs/t3_baseline.json \
  --input data/canonical/train.csv \
  --ids-file data/splits/random_holdout_ids.txt \
  --output artifacts/local-baseline/generations.jsonl \
  --metadata artifacts/local-baseline/run-metadata.json \
  --engine vllm

python -m src.evaluate \
  --generations artifacts/local-baseline/generations.jsonl \
  --labels data/splits/random_holdout.csv \
  --k 1 \
  --output artifacts/local-baseline/metrics.json
```

### 실험 runner

`scripts/`의 runner는 이전 단계의 산출물과 manifest를 확인하면서 각 실험을 실행합니다. 대부분의 GPU runner는 `/workspace`와 `/venv/main`이 준비된 원격 Linux 환경을 전제로 합니다.

| 단계 | 목적 | 실행 파일 |
|---|---|---|
| T5 | 16-sample rejection sampling 기반 RFT 생성 | `scripts/run_t5_rft.sh` |
| T6 | answer-only·RFT·외부 CoT QLoRA SFT 비교 | `scripts/run_t6.sh` |
| T7 | 두 번째 RFT 라운드와 hard-tail 데이터 구축 | `scripts/run_t7.sh` |
| T8 | self-consistency와 adaptive sampling | `scripts/run_t8.sh` |
| T9 | 생성 후보를 읽고 선택하는 GenSelect | `scripts/run_t9.sh` |
| T11 | hard-CoT 품질 확인과 SFT/DPO 경로 | `scripts/run_t11.sh` |
| T12 | ORM 점수화·가중 투표·2-way sharding | `scripts/run_t12_cmu_orm.sh` |
| T12b | question-local ORM 데이터 gate와 후속 학습 | `scripts/run_t12b_question_local_orm.sh` |

### 제출 payload 검증

`src.submit`은 리더보드 문항에 대한 생성 결과를 label-blind majority-vote payload로 변환합니다. 아래 명령은 현재 보존된 T8 `k=32` 생성 결과를 검증하며, 최종 CSV 자체가 아니라 `submission-prepared.json`을 작성합니다.

```bash
python -m src.submit \
  --input data/deep_chal_math_leaderboard.csv \
  --generations artifacts/submissions/t8_majority_k32/generations.jsonl \
  --config configs/t8_self_consistency.json \
  --metadata artifacts/submissions/t8_majority_k32/run-metadata.json \
  --output artifacts/submissions/t8_majority_k32/submission-prepared.json \
  --k 32
```

---

## ⚙️ 설정

실험 계약은 `configs/*.json`에 둡니다. 모델 revision, prompt, sampling, 학습 하이퍼파라미터, 출력 경로, 의사결정 gate를 코드와 분리해 관리합니다.

| 설정 파일 | 역할 |
|---|---|
| `configs/t2_data.json` | canonical 데이터·holdout·RFT pool 생성 규칙 |
| `configs/t3_baseline.json` | greedy baseline 생성 계약 |
| `configs/t4_output_contract.json` | 답 출력 형식과 generation 재측정 계약 |
| `configs/t5_rft_r1.json` | 16-sample RFT 생성 설정 |
| `configs/t6_sft_v1.json` | QLoRA SFT 학습·calibration 설정 |
| `configs/t8_self_consistency.json` | `k=32` self-consistency와 adaptive sampling |
| `configs/t9_genselect.json` | 후보 선택기 학습·평가 설정 |
| `configs/t12_cmu_orm.json` | pointwise ORM과 2-GPU scoring 설정 |
| `configs/t12b_question_local_orm.json` | question-local ranking ORM 설정 |

---

## 🏗️ 구조

```text
.
├── src/
│   ├── build_data.py       # 데이터셋과 holdout 생성
│   ├── generate.py         # HF/vLLM 생성
│   ├── extract.py          # 표기 기반 정수 추출
│   ├── evaluate.py         # exact-match 평가
│   ├── train_sft.py        # QLoRA SFT
│   ├── train_dpo.py        # DPO 학습
│   ├── genselect.py        # 후보 선택기
│   ├── orm_*.py            # ORM 점수화와 투표
│   └── submit.py           # 제출 payload 검증
├── configs/                # 단계별 실험 계약
├── scripts/                # 실행·원격 runner
├── data/                   # 원본·파생 데이터
├── artifacts/              # 생성물·adapter·manifest·지표
├── tests/                  # 단위·계약 테스트
└── docs/                   # 대회 정보와 전략 문서
```

```text
원본 CSV + T2 config
  │ 데이터 검증·결정론적 분할
  ▼
src/build_data.py ──▶ canonical train / holdouts / RFT pool
  │ 학습 또는 label-blind 생성
  ▼
src/generate.py ──▶ raw generations.jsonl
  │ 표기만 정규화한 후보 답
  ▼
src/extract.py ──▶ src/evaluate.py / self-consistency / ORM / submit.py
  │ 선택·형식·coverage 검증
  ▼
submission.csv
```

> 학습·추론의 계약은 config와 manifest로 고정하고, 정답 라벨은 생성이 아닌 평가 단계에서만 사용합니다.

---

## 🤖 모델 및 평가

### 모델과 추론 규칙

- **학생 모델** — `Qwen/Qwen2.5-3B-Instruct`, revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`
- **생성 엔진** — Hugging Face 또는 vLLM; 실험 설정에서는 vLLM `bfloat16` 경로가 주로 선택되어 있습니다.
- **답 형식** — 모델 출력에서 정수 문자열을 추출한 뒤 `^-?(?:0|[1-9][0-9]*)$` 형식으로 정규화합니다.
- **추론 중 제한** — 외부 API·검색·코드 실행·수학 solver를 사용하지 않으며, majority voting·self-consistency와 같은 다중 샘플링은 별도 계약으로 관리합니다.

### 데이터와 지표

| 항목 | 현재 저장소 기준 |
|---|---:|
| 원본 train | 17,000행 |
| organizer exclusion | 627행 |
| canonical train | 16,373행 |
| leaderboard | 1,000행 |
| random/template holdout | 각 1,637행 |
| hard/format diagnostic | 550행 / 256행 |
| RFT pool | 12,636행 |
| 평가 지표 | Accuracy (Exact Match) |

리더보드 원본 CSV의 세 번째 헤더는 공백이 포함된 ` answer`이므로, 로더는 헤더를 `strip()`한 뒤 사용합니다. 자세한 대회 조건과 데이터 provenance는 [대회 정보 문서](docs/information/README.md), 전략과 실험 판단 근거는 [전략 문서](docs/strategy/winning-strategy.md)에서 확인할 수 있습니다.

---

## 📦 의존성

`requirements.lock`은 기록된 Linux/CUDA 실행 환경의 패키지를 고정합니다.

| 구성 요소 | 기록된 버전 또는 조건 |
|---|---|
| Python | 3.12.13 |
| PyTorch | 2.13.0+cu130 |
| Transformers | 5.15.1 |
| PEFT | 0.20.0 |
| TRL | 1.10.0 |
| vLLM | 0.27.1 |
| GPU | 생성·학습용 CUDA GPU; T12/T12b는 RTX 4090 2장 |

단위 테스트는 `pytest`, runner와 일부 하드웨어 검사는 Bash·`jq`·`nvidia-smi`를 사용합니다. 저장소에 license 파일은 확인되지 않아 README에 라이선스를 지정하지 않습니다.
