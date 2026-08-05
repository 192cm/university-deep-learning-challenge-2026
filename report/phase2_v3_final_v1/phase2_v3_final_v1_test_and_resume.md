# phase2_v3_final_v1 tests and safe resume

## Recorded local results

- `python -m py_compile ...`: configured phase2_v3_final_v1 scripts passed.
- Phase 2 contract/budget tests: 24/24 passed before paid execution.
- `python -m unittest discover -s tests -v`: 56 tests ran; 48 passed and 8 Phase 0 environment checks errored because this Windows session lacks the pinned Linux CUDA/model packages; no test expectation was relaxed.
- `python scripts/verify_phase2_v2.py --config configs/phase2_v3_final_v1.json --env-file .env`: final configured artifacts passed 41/41 checks.
- Smoke passed; both fixed 40-row comparison efforts failed the unchanged quality gate, so 100-row quality audit and main Batch were not started.
- `git diff --check`: passed; Git printed only line-ending conversion warnings for pre-existing tracked files.

## Safe resume after the failed comparison gate

No paid quality-audit or main-Batch command is authorized from this state. The new v3 low and medium settings both failed the fixed comparison gate, so rerunning them would only repeat paid work.

The current evidence can be reproduced without paid API calls:

```powershell
python scripts/run_phase2_v2_luna.py --config configs/phase2_v3_final_v1.json metrics --stage comparison --effort low
python scripts/run_phase2_v2_luna.py --config configs/phase2_v3_final_v1.json metrics --stage comparison --effort medium
python scripts/run_phase2_v2_luna.py --config configs/phase2_v3_final_v1.json select-effort
python scripts/finalize_phase2_v2.py --config configs/phase2_v3_final_v1.json
python scripts/verify_phase2_v2.py --config configs/phase2_v3_final_v1.json --env-file .env
```

A future paid resume requires an explicitly reviewed prompt/configuration or model strategy change in a new dataset version, followed again by smoke and comparison gates. Do not continue directly to the 100-row audit or main Batch.
