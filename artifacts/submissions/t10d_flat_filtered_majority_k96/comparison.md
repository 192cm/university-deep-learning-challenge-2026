# T10d three-view flat filtered majority@96

Three immutable same-base k=32 pools are filtered independently with the frozen T8-3 policy, including per-arm fallback, concatenated in base/cot-boxed/RFT order, and reduced with one ordinary answer-string majority vote.

## Holdout

- Accuracy: 71.18% (2660/3737).
- Versus T8 unfiltered: +1.87pp, recover 120 / regress 50, exact McNemar p=8.02e-08, paired bootstrap 95% CI [+1.20,+2.54]pp.
- Versus T8-3: +0.40pp, p=0.203.
- Split deltas versus T8: random_holdout +1.89pp, template_holdout +1.41pp, hard_diagnostic +2.18pp, format_diagnostic +4.69pp.
- Numerical preregistration-style gate: exploratory_passes_numerical_gate; this remains exploratory rather than independent confirmation.

## Leaderboard submission

- Rows: 831; labels unavailable and accuracy not computed.
- Changes versus frozen arm submissions: base 61, cot_boxed 66, rft_r1 70.

## Rules status

The recorded rules allow local same-base multi-sampling, majority voting, and same-base LoRA ensembles, and do not state a numerical sample cap. Written organizer confirmation is still requested for the exact 96-sample composition and extraction-path/termination-based vote exclusion.
