<div align="center">

# 🧮 T12 CMU ORM 최종 추론 패키지

**정수형 수학 문제 추론 패키지** — Qwen2.5-3B가 생성한 후보 풀이를 학습된 LoRA ORM으로 채점해 `test_submission.csv`를 만듭니다.

<br>

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/) [![PyTorch CUDA](https://img.shields.io/badge/PyTorch-CUDA%20BF16-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/) [![Qwen2.5 3B](https://img.shields.io/badge/Qwen2.5-3B-FFD21E?style=flat-square)](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) [![vLLM 0.27.1](https://img.shields.io/badge/vLLM-0.27.1-5E5CE6?style=flat-square)](https://docs.vllm.ai/) [![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)

<br>

[주요 기능](#-주요-기능) · [빠른 시작](#-빠른-시작) · [사용 방법](#-사용-방법) · [설정](#️-설정) · [동작 구조](#️-동작-구조) · [재현성 검증](#-재현성-검증) · [의존성](#-의존성)

</div>

---

## ✨ 주요 기능

- **노트북 단일 실행** — `t12_cmu_orm_inference.ipynb`를 위에서 아래로 실행해 입력 검증부터 제출 파일 생성까지 완료합니다.
- **학습된 ORM 포함** — 정답 가능성을 채점하는 rank-64 LoRA 어댑터와 scalar score head를 `model/`에 제공합니다.
- **동결된 T12 추론** — 문항당 후보 32개를 생성하고 `지지 표 수 × ORM 점수의 기하평균`이 가장 큰 정수 답을 선택합니다.
- **중단 후 재개** — 후보 생성과 ORM 채점 결과를 JSONL에 누적해 완료된 문항과 후보를 다시 처리하지 않습니다.
- **제출 전 자동 검증** — 입력 순서, 행 수, ID 고유성, 빈 답과 정수 형식을 검사하고 SHA-256 감사 기록을 남깁니다.

---

## 🚀 빠른 시작

### 1. 환경 준비

NVIDIA CUDA GPU가 있는 Linux 환경에서 저장소 루트를 기준으로 실행합니다. 모델 파일은 약 457 MiB이므로 Git LFS가 필요합니다.

```bash
git lfs install
git lfs pull
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. GPU와 모델 캐시 설정

Qwen 베이스 모델은 공개 모델이므로 별도 자격 증명이 필요하지 않습니다. 사용할 GPU와 Hugging Face 캐시 위치만 지정합니다.

```bash
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="$PWD/.cache/huggingface"
```

### 3. 실행

다음 명령은 노트북을 처음부터 끝까지 실행하고 같은 폴더에 `test_submission.csv`를 생성합니다.

```bash
jupyter nbconvert --execute --to notebook --inplace \
  --ExecutePreprocessor.timeout=-1 \
  t12_cmu_orm_inference.ipynb
```

---

## 📖 사용 방법

### 노트북

대화형 실행이 필요하면 아래 명령으로 노트북을 열고 **Run All**을 선택합니다.

```bash
jupyter lab t12_cmu_orm_inference.ipynb
```

실행 중간 산출물과 최종 결과는 다음 위치에 저장됩니다.

| 경로 | 내용 |
|------|------|
| `output/generations.jsonl` | Qwen이 생성한 문항당 32개 후보 풀이 |
| `output/candidate_scores.jsonl` | LoRA ORM의 후보별 logit과 sigmoid 점수 |
| `output/prediction_diagnostics.jsonl` | 문항별 답 그룹, 가중치, 폴백 정보 |
| `output/submission_audit.json` | 입력·중간 산출물·제출 파일의 행 수와 SHA-256 |
| `test_submission.csv` | 최종 제출 파일 |

생성 또는 채점 셀이 중단되면 같은 셀을 다시 실행하면 됩니다. 완료된 결과는 유지되고 남은 항목만 이어서 처리합니다.

### CLI

노트북 없이 각 단계를 직접 실행할 수도 있습니다.

```bash
COMMON_ARGS=(
  --input test_data.csv
  --adapter model/t12_cmu_orm_adapter
  --work-dir output
  --output test_submission.csv
)

python t12_pipeline.py validate "${COMMON_ARGS[@]}"
python t12_pipeline.py generate "${COMMON_ARGS[@]}"
python t12_pipeline.py score "${COMMON_ARGS[@]}"
python t12_pipeline.py submit "${COMMON_ARGS[@]}"
```

> `--limit 2` 같은 제한 실행은 스모크 테스트용입니다. 최종 제출에서는 제한을 제거해 입력 전체를 처리해야 합니다.

---

## ⚙️ 설정

> 모든 실행 설정은 `t12_cmu_orm_inference.ipynb` 상단 파라미터 셀에서 관리합니다. 경로와 GPU를 바꾸기 위해 추론 코드를 수정할 필요가 없습니다.

| 키 | 기본값 | 설명 |
|----|--------|------|
| `INPUT_PATH` | `test_data.csv` | `id`, `question`을 읽을 테스트 입력 |
| `ADAPTER_PATH` | `model/t12_cmu_orm_adapter` | 학습된 LoRA ORM 폴더 |
| `OUTPUT_PATH` | `test_submission.csv` | 최종 제출 CSV 경로 |
| `BASE_MODEL_ID_OR_PATH` | `Qwen/Qwen2.5-3B-Instruct` | Hugging Face 모델 ID 또는 동일 revision의 로컬 snapshot |
| `LIMIT` | `None` | 처리할 선두 문항 수로, 최종 실행에서는 반드시 `None` |
| `GENERATION_GPU` / `SCORING_GPU` | `0` / `0` | 후보 생성과 ORM 채점에 사용할 GPU |

세부 실행 기본값은 T12 설정과 동일하게 `k=32`, `seed=42`, `temperature=0.8`, `top_p=0.95`, `max_new_tokens=2048`, ORM batch size `4`로 고정되어 있습니다. 베이스 모델 revision은 다음 commit으로 고정됩니다.

```text
aa8e72537993ba99e69dfaafa59ed015b17504d1
```

인터넷을 사용할 수 없는 환경에서는 `BASE_MODEL_ID_OR_PATH`를 위 revision의 로컬 Qwen snapshot 경로로 설정합니다. 프로젝트에서 별도로 학습한 모델은 `model/t12_cmu_orm_adapter/`에 모두 포함되어 있습니다.

---

## 🏗️ 동작 구조

```
.
├── t12_cmu_orm_inference.ipynb     # 전체 실행 노트북
├── t12_pipeline.py                 # standalone 추론 구현
├── test_data.csv                   # 2,000문항 입력
├── model/
│   └── t12_cmu_orm_adapter/
│       ├── adapter_model.safetensors
│       ├── adapter_config.json
│       └── manifest.json
├── PACKAGE_MANIFEST.json           # 패키지 파일 SHA-256
└── requirements.txt                # 실행 의존성
```

```
test_data.csv
      │ 문제 텍스트
      ▼
Qwen2.5-3B causal LM ──▶ 후보 풀이 32개
                              │ 문제 + 전체 풀이
                              ▼
                    LoRA ORM sequence classifier
                              │ 후보별 정답 점수
                              ▼
              정수 추출·답 그룹별 기하 가중 투표
                              │ 문항별 정수 하나
                              ▼
                    test_submission.csv
```

> 풀이 생성 모델과 후보 채점 모델을 분리하고 두 단계를 별도 프로세스로 실행해, 생성 종료 후 GPU 메모리를 반환한 다음 ORM을 로드합니다.

---

## 🔎 재현성 검증

| 검증 항목 | 확인 값 |
|----------|---------|
| 테스트 입력 | 2,000행, 고유 ID 2,000개, 비어 있지 않은 정답 0개 |
| 테스트 입력 SHA-256 | `106358e4d0365b72c387a9d31f233d52e9122b8547c3cd8d076cf25f39295f9b` |
| ORM adapter SHA-256 | `8dcf2404a6889270af846b51dda1ea450c87c161b2fe5d0b6a165adab9684e9c` |
| 저장된 adapter tensor | LoRA tensor 504개와 scalar score head 1개 |
| 답 추출기 대조 | 기존 T12 생성 10,000건과 결과 일치 |
| 과거 제출 재현 | 831행 T12 제출 CSV와 바이트 동일, SHA-256 `e0dffe211c2a8ea97d323530e2ed5e4626f2dd4726f71b21807cf85d55705e11` |

`PACKAGE_MANIFEST.json`에는 노트북, 추론 코드, 입력 데이터와 모델 파일의 SHA-256이 기록되어 있습니다. 파이프라인은 추론 전에 adapter hash와 베이스 모델 identity를 검사합니다.

현재 패키지를 구성한 로컬 작업공간에는 CUDA와 vLLM이 없어 2,000문항 전체 GPU 추론은 실행하지 않았습니다. 노트북 구조, 입력·모델 무결성, 답 추출·투표 동작은 검증했으며, standalone 코드로 기존 T12 제출물을 바이트 단위로 재현했습니다.

---

## 📦 의존성

| 구성 요소 | 조건 또는 버전 | 역할 |
|----------|----------------|------|
| Python | `3.12` | 노트북과 파이프라인 실행 |
| PyTorch | `>=2.8`, CUDA·BF16 지원 빌드 | 생성·ORM 추론 |
| Transformers | `5.15.1` | tokenizer와 sequence classifier 로드 |
| PEFT | `0.20.0` | LoRA ORM 결합 |
| vLLM | `0.27.1` | 문항당 32개 후보 생성 |
| JupyterLab | `>=4.4` | 노트북 실행 환경 |
| Git LFS | 설치 필요 | 457 MiB `adapter_model.safetensors` 내려받기 |

원래 T12 실험은 RTX 4090 두 장을 독립 shard로 사용했습니다. 이 패키지는 저장소 제출과 실행 편의를 위해 한 개의 visible CUDA GPU에서 생성과 채점을 순차 실행하며, 모델·프롬프트·샘플링·채점·정수 추출·투표 규칙은 유지합니다.
