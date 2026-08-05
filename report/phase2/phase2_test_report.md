# Phase 2 test report

Executed on 2026-08-04 with local CPU Python `C:\Python313\python.exe`.

- Phase 2 focused tests: 13/13 passed (budget/request protection, validation, cost accounting, and external-contamination indexing).
- Locally runnable Phase 0 smoke + Phase 1 + Phase 2 suite: 40/40 passed.
- Phase 2 artifact integration verification: 25/25 passed.
- Full discovery found 44 test methods. Four Phase 0 environment methods cannot run in this local Windows CPU interpreter because their pinned runtime is the previously recorded Vast `/venv/main/bin/python` CUDA environment; the first discovery emitted eight import/CUDA error events from those methods. That remote GPU was intentionally not started for Phase 2.

The unavailable Phase 0 environment checks are not counted as Phase 2 failures and were not bypassed by installing or changing the pinned Phase 0 stack.
