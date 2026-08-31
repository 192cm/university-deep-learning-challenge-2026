<div align="center">

# 🧠 제5회 대학 연합 아주 소중한 딥러닝 챌린지 2026

**프로젝트 아카이브** — 수학 추론 시스템을 설계하고 실험한 과정, 판단, 결과를 보존합니다.

<br>

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.13](https://img.shields.io/badge/PyTorch-2.13%20%7C%20CUDA%2013-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![vLLM 0.27.1](https://img.shields.io/badge/vLLM-0.27.1-5E5CE6?style=flat-square)](https://docs.vllm.ai/)
[![Qwen2.5 3B](https://img.shields.io/badge/Base-Qwen2.5--3B--Instruct-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

<br>

[프로젝트 개요](#-프로젝트-개요) · [실험 여정](#-실험-여정) · [기록 읽는 법](#-기록-읽는-법) · [저장소 구조](#-저장소-구조) · [공개 범위](#-데이터와-공개-범위) · [환경 기록](#-환경-기록)

</div>

> [!NOTE]
> **최종 제출물은 [`final/`](final/) 폴더를 참고해 주세요.** 제출용 코드, 추론 노트북, 모델 어댑터와 실행 안내를 함께 정리했습니다.

> [!IMPORTANT]
> 이 저장소는 **재현용 배포판이 아니라 프로젝트 기록 보관소**입니다. 대회가 제공한 데이터와 그 파생본은 공개할 수 없으므로 포함하지 않습니다. 대신 소스 코드, 설정, 실험 로그, 집계 지표, 분석 보고서와 의사결정 근거를 보존합니다.

## 🎯 프로젝트 개요

아주대학교 AI융합교육원이 주최한 대학 연합 딥러닝 챌린지에서, 수학에 특화되지 않은 고정 베이스 모델 `Qwen/Qwen2.5-3B-Instruct`로 처음 보는 수학 문제의 정수 답을 추론하는 시스템을 개발했습니다. 평가는 정답 문자열의 exact match를 기준으로 했습니다.

이 저장소의 중심은 최종 점수 하나가 아니라, T0부터 T13까지 이어진 실험에서 **무엇을 시도했고, 어떤 근거로 채택·보류·기각했는지**를 남기는 데 있습니다.

- **단계별 실험 기록** — 데이터 계약, 출력 형식, SFT/RFT, self-consistency, 후보 선택, ORM 실험을 순서대로 보존합니다.
- **근거 기반 의사결정** — 각 실험의 gate, 집계 지표, audit과 후속 판단을 함께 기록합니다.
- **평가 오염 방지** — 생성 단계에서는 정답 라벨을 사용하지 않고, 평가는 별도 단계로 분리했습니다.
- **비용을 고려한 탐색** — 사전 gate를 통과하지 못한 경로는 고비용 학습으로 확장하지 않았습니다.
- **기록 가능한 결과 중심** — 대용량 원시 산출물 대신 설정, manifest, 요약 지표, 보고서와 실행 당시 소스 스냅샷을 남깁니다.

---

## 🧭 실험 여정

| 단계 | 탐색 내용 | 내부 검증 결과와 판단 |
|---|---|---|
| T0–T4 | 베이스라인과 출력 계약 | random holdout 정확도 `64.20% → 67.99%`, invalid rate `15.03% → 0.61%` — 출력 계약 채택 |
| T5–T7 | RFT 데이터와 QLoRA SFT | 유의미한 개선이 없거나 성능이 하락해 베이스 모델 유지 — SFT-v2는 실행 전 중단 |
| T8 | fixed majority@32 | random holdout `74.28%`, union `69.31%`, T4c 대비 `+6.61%p` — 채택 |
| T8-3–T10c | vote-quality filter, prompt 다양화, GenSelect, weighted vote | 일부 근접 결과가 있었지만 사전 gate를 충족하지 못해 보류 또는 기각 |
| T10d–T10e | 3-view 추론 | union `71.18%` / `71.29%` — 규칙 확인과 운영 판단이 남은 후보로 보존 |
| T11 | teacher 기반 SFT/DPO 경로 | teacher preflight gate 실패 — 후속 고비용 학습 미실행 |
| T12 | pointwise ORM weighted vote | fresh validation `87.4% → 87.6%` (`+0.2%p`) — 효과가 불충분해 보류 |
| T12b | question-local selective override | override `0건`, baseline과 동일 — 기존 결과 보존 |

> 표의 수치는 서로 다른 내부 holdout 또는 진단 조건에서 얻은 실험 기록이며, 공식 리더보드 점수가 아닙니다. 조건이 다른 행의 수치를 직접 비교해서는 안 됩니다.

T12에서는 ORM이 raw majority의 선택을 24건 바꿨고, 8건을 복구하는 동안 6건을 악화시켜 순증은 2건이었습니다. 후보 수준의 분류력은 확인했지만 fold별 효과가 일관되지 않아, 모델 점수를 최종 선택 규칙으로 채택하지 않았습니다.

---

## 📚 기록 읽는 법

처음 살펴본다면 아래 순서가 프로젝트의 흐름을 가장 잘 보여줍니다.

1. [실험 실행 기록](docs/strategy/execution-prompts.md) — T0–T13의 실행 맥락, 관찰 결과, 채택·보류·기각 판단을 시간순으로 정리한 중심 문서입니다.
2. [초기 전략](docs/strategy/winning-strategy.md) — 실험을 시작할 때 세운 가설과 우선순위를 담고 있습니다.
3. [T8 진단 보고서의 근거 기록](report/t8-pass-majority-diagnostic-2026-08-27/source-notes.md) — majority voting을 채택한 근거와 오류 분석을 기록합니다.
4. [T12 ORM 진단 보고서의 근거 기록](report/t12-orm-diagnostic-2026-08-28/source-notes.md) — ORM의 개선·악화 사례를 집계하고 보류 결정을 설명합니다.
5. [`configs/`](configs/)와 [`artifacts/`](artifacts/) — 실험 계약과 공개 가능한 manifest, metrics, audit, 실행 로그를 확인할 수 있습니다.

코드의 현재 상태는 [`src/`](src/), 실행 흐름은 [`scripts/`](scripts/), 계약 검증은 [`tests/`](tests/)에 남아 있습니다. 이 파일들은 당시 작업을 설명하고 보존하기 위한 자료이며, 데이터와 모델이 빠진 새 clone에서 전체 실험을 그대로 실행할 수 있다는 의미는 아닙니다.

---

## 🏗️ 저장소 구조

```text
.
├── src/          # 생성·평가·학습·선택 로직
├── configs/      # 단계별 실험 계약과 의사결정 gate
├── scripts/      # 로컬·원격 실행 runner
├── analysis/     # 실험 후 분석 코드
├── artifacts/    # 공개 가능한 manifest·지표·audit·로그
├── report/       # 장기 보존용 진단 보고서
├── docs/         # 대회 정보, 전략, 실행 기록
├── tests/        # 단위·계약 테스트
├── submission.csv
└── data/         # 대회 제공 데이터 — 로컬 전용, Git 제외
```

```text
대회 제공 데이터 (로컬 전용)
          │
          ▼
  config + runner ──▶ 학습·생성 (원시 산출물은 로컬 전용)
                              │
                              ▼ 집계·진단
                         artifacts/
                              │
                              ▼
                    reports · docs · submission
```

설정과 실행 코드는 실험의 의도를, manifest와 집계 지표는 실제 수행 여부를, 보고서는 결과 해석과 다음 결정을 설명합니다. 원시 데이터나 모델 파일이 없어도 프로젝트의 판단 과정이 이어지도록 이 세 층을 함께 보존합니다.

---

## 🔒 데이터와 공개 범위

대회 문제와 정답은 주최 측이 제공한 자료이므로 경로와 형식에 관계없이 Git에 올리지 않습니다. `.gitignore`는 확장자 허용 목록이 아니라 민감하거나 기록 가치가 낮은 파일을 차단하는 denylist로 관리합니다.

| Git에 보존 | 로컬에만 보존 |
|---|---|
| 소스 코드, 테스트, 실행 스크립트, 설정 | `data/` 전체와 대회 데이터 복사본 |
| 전략 문서, 실행 기록, 분석 보고서 | 문제문·정답·행 단위 평가 결과가 담긴 CSV |
| manifest, 집계 metrics, comparison, audit | 원시 생성 결과인 JSONL·NDJSON |
| 대회 원문을 포함하지 않는 로그와 소스 스냅샷 | 모델 가중치, adapter, checkpoint, optimizer state |
| 제출 CSV와 결과 이미지 | 캐시, 가상환경, 임시 파일, 압축 파일 |

따라서 이 저장소만 clone해서 실험을 재현할 수는 없습니다. 실행에는 별도로 사용 권한을 가진 대회 데이터, 로컬 모델 자산, 기록된 하드웨어 환경이 필요합니다. 설정과 로그에 남은 `/workspace` 등의 절대 경로는 당시 원격 실행 환경을 나타내는 역사적 기록입니다.

---

## 📦 환경 기록

아래 버전은 설치 지원 범위가 아니라, 주요 실험을 수행했을 당시의 환경 스냅샷입니다. 전체 패키지 목록은 [`requirements.lock`](requirements.lock), 런타임 정보는 [`environment.json`](environment.json)에 보존되어 있습니다.

| 구성 요소 | 기록된 버전 또는 조건 |
|---|---|
| Python | 3.12.13 |
| PyTorch | 2.13.0+cu130 |
| Transformers | 5.15.1 |
| PEFT | 0.20.0 |
| TRL | 1.10.0 |
| vLLM | 0.27.1 |
| datasets | 5.0.1 |
| bitsandbytes | 0.50.1 |
| T12/T12b 하드웨어 | NVIDIA GeForce RTX 4090 2장 |

베이스 모델은 `Qwen/Qwen2.5-3B-Instruct` revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`로 고정해 기록했습니다.
