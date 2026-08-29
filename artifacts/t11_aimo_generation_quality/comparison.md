# T11 hard-CoT SFT → correct/wrong DPO

- 판정: **teacher_gate_failed**
- accepted correct trace: 0/256
- 품질 필터 전 extracted-correct: 76/256
- 정답 trace가 있는 문항: 0/64
- final-line 계약 위반: 256/256
- 사전등록대로 SFT/DPO/validation/holdout 생성은 실행하지 않았다.

AIMO 대비 축소·금지: 3B 단일 GPU, hard 최대 2,000문항, teacher k=4→조건부 8, text-only CoT만 허용했다. TIR/Python/SymPy/solver, k>=64, 새 prompt·vote filter 탐색, holdout 사후 checkpoint 선택은 금지했다.
