# Phase 2 v2 tests and safe resume

## Recorded local results

- `python -m py_compile ...`: Phase 2 v2 scripts and test module passed.
- `python -m unittest tests.test_phase2_v2 -v`: 12/12 passed.
- `python -m unittest discover -s tests -v`: 56 tests ran; 48 passed and 8 Phase 0 environment checks errored because the current Windows `C:\Python313` session does not contain the pinned Linux `/venv/main` packages (`torch`, `transformers`, `accelerate`, `peft`, `trl`, `bitsandbytes`) or CUDA.
- All Phase 1 and Phase 2 functional tests passed; no dependency installation, baseline rerun, or GPU startup was attempted.
- `python scripts/verify_phase2.py --config configs/phase2.json --env-file .env`: existing Phase 2 v1 passed 25/25 checks.
- `python scripts/verify_phase2_v2.py --config configs/phase2_v2.json --env-file .env`: final Phase 2 v2 artifacts passed 38/38 checks.
- `git diff --check`: passed; Git printed only line-ending conversion warnings for pre-existing tracked files.

## Safe resume after the failed comparison gate

No paid quality-audit or main-Batch command is authorized from this state. The unchanged low and medium settings both failed the fixed comparison gate, so rerunning them would only repeat paid work.

The current evidence can be reproduced without paid API calls:

```powershell
python scripts/run_phase2_v2_luna.py --config configs/phase2_v2.json metrics --stage comparison --effort low
python scripts/run_phase2_v2_luna.py --config configs/phase2_v2.json metrics --stage comparison --effort medium
python scripts/run_phase2_v2_luna.py --config configs/phase2_v2.json select-effort
python scripts/finalize_phase2_v2.py --config configs/phase2_v2.json
python scripts/verify_phase2_v2.py --config configs/phase2_v2.json --env-file .env
```

A future paid resume requires an explicitly reviewed prompt/configuration or model strategy change in a new dataset version, followed again by smoke and comparison gates. Do not continue directly to the 100-row audit or main Batch.
