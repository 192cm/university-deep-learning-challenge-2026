# T10e three-view arm-normalized filtered voting@96

Each immutable T10d arm is filtered independently. Its valid answer histogram is normalized to total mass one, and the three arm distributions are summed.

## Holdout

- Accuracy: 71.29% (2664/3737).
- Versus T8 unfiltered: +1.98pp, recover 128 / regress 54, p=4.12e-08, bootstrap 95% CI [+1.28,+2.68]pp.
- Versus T10d flat: +0.11pp, recover 15 / regress 11, p=0.557, bootstrap 95% CI [-0.16,+0.37]pp.
- Splits: random_holdout 76.18% (+0.00pp vs T10d), template_holdout 75.63% (+0.24pp vs T10d), hard_diagnostic 42.18% (+0.36pp vs T10d), format_diagnostic 54.69% (-0.39pp vs T10d).
- Gate: exploratory_passes_numerical_t8_gate; the incremental gain remains exploratory.

## Leaderboard submission

- Input: `data/deep_chal_math_leaderboard_filtered.csv`; rows: 831.
- Changed answers versus frozen T10d flat submission: 9.
- Labels are unavailable; leaderboard accuracy was not computed.
