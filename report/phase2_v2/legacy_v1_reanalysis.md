# Phase 2 v1 raw-response reanalysis under the v2 integer contract

The original request and raw-response files were read without modification.
Only r4 audit requests whose reconstructed body hash proves an answer-hidden request, whose IDs are in the filtered train, and which are not Phase 1, leaderboard, or local-holdout protected were analyzed.
Reuse audit: 400 passed, 0 failed.

| effort | completion | truncation | completed JSON | completed schema | canonical integer | noncanonical | first exact | pass@2 | old verifier | corrected verifier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| low | 99.0% | 1.0% | 100.0% | 100.0% | 60.1% | 79 | 42.0% | 48.0% | 28.0% | 0.0% |
| medium | 96.5% | 3.5% | 100.0% | 100.0% | 59.1% | 79 | 49.0% | 53.0% | 31.0% | 0.0% |

Completed-response JSON and schema rates deliberately exclude incomplete responses from their denominators. Noncanonical final answers are rejected without numeric repair.
