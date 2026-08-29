# T11d label-blind prompt format canary

| Metric | Old prompt | New prompt |
|---|---:|---:|
| Strict final line | 64.06% | 72.07% |
| Exactly one marker | 91.21% | 78.52% |
| Last-integer path | 1.17% | 4.69% |
| Invalid output | 4.88% | 2.93% |
| Hit max | 0.78% | 0.98% |

Gate: `failed`. Gold accuracy was not loaded or used.
