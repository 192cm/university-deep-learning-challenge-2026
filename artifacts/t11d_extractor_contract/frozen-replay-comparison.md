# T11d frozen extractor replay

Status: `reused_holdout_diagnostic`. This repeatedly used T8 holdout is diagnostic only and does not adopt the extractor for T12.

| Metric | Old | New |
|---|---:|---:|
| Plurality@32 | 69.3069% (2590/3737) | 70.7252% (2643/3737) |
| Pass@32 | 84.3993% (3154/3737) | 82.9810% (3101/3737) |
| Invalid outputs | 0.7727% (924/119584) | 9.3374% (11166/119584) |

## Paired result

- Rescued/broken/net: 80/27/+53 questions (+1.4182pp).
- Exact McNemar p=2.87737e-07; paired bootstrap 95% CI [+0.9098, +1.9534]pp.
- Label-blind answer/path changes: 11488 candidates across 1368 questions.
- Explicit zero-decimal normalization: 1185 candidates across 86 questions.
- Explicit barrier removed old last-integer fallbacks: 9451 candidates across 1162 questions.

## train-012155 regression

Old plurality: `50`; new plurality: `400`.

The extractor grammar and tie-break were fixed before labels were loaded. Fresh validation is still required before T12 adoption.
