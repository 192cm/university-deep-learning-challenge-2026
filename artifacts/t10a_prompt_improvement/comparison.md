# T10a prompt improvement comparison

| Arm | Prompt | Union accuracy | Δ vs A | McNemar p | Decision |
|---|---|---:|---:|---:|---|
| A | base | 69.31% | — | — | reference |
| B | strong_cot | 69.52% | +0.214pp | 0.625861 | hold |
| C | cot_boxed | 69.36% | +0.054pp | 0.942493 | hold |
| D | cot_brief | 68.85% | -0.455pp | 0.254247 | reject |

Final decision: **hold**. At least one candidate improved but no candidate passed every adoption gate.
T10b base template: `base`.

Predictions were frozen before labels were loaded. No generation pool was overwritten.
