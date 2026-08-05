# Phase 2 Luna quality report

The fixed 100-row audit failed at both allowed reasoning settings, so main generation was not started.

Both audits used `gpt-5.6-luna`, `tools=[]`, `store=false`, Structured Outputs, `text.verbosity=low`, and `max_output_tokens=4096`; only `reasoning.effort` changed from `low` to `medium`.

| effort | first exact | pass@2 | JSON parse | extraction failure | automated review error | gate |
|---|---:|---:|---:|---:|---:|---|
| low | 54.0% | 58.0% | 99.0% | 23.5% | 28.0% | failed |
| medium | 61.0% | 65.0% | 96.5% | 25.5% | 31.0% | failed |

Thresholds were 75%, 85%, 99%, <2%, and <5%, respectively. The review error is an automated deterministic validation signal, not a claim of human review. Audit rows are excluded from SFT. Diagnostic grades: `{"low": {"A": 25, "B": 14, "D": 61}, "medium": {"A": 32, "B": 13, "D": 55}}`.
