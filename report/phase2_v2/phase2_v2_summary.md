# Phase 2 v2 summary

Status: **blocked_comparison_gate**. No low/medium comparison setting passed the required gate.

- Filtered canonical rows: 16528
- Eligible / selected / generated / accepted / unprocessed: 12255 / 0 / 0 / 0 / 12255
- A/B/C/D/unsuitable: 0 / 0 / 0 / 0 / 0
- Non-integer labels excluded: 0
- Suspected label/problem-quality exclusions: 2
- New noncanonical integer outputs: 0
- Cumulative paid cost / remaining: $0.5429184 / $3.9570816
- Final JSONL: `data\phase2\phase2_verified_cot_luna_3k_v2\phase2_verified_cot_luna_3k_v2.jsonl`
- Final SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Safe resume: `python scripts/run_phase2_v2_luna.py --config configs/phase2_v2.json metrics --stage comparison --effort low`
