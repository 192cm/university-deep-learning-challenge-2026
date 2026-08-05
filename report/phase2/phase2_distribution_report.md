# Phase 2 distribution report

## Eligible competition rows (unprocessed)

```json
{
  "answer_magnitude": {
    "d1": 3577,
    "d2": 5109,
    "d3": 2349,
    "d4_5": 1281,
    "d6_plus": 112
  },
  "answer_sign": {
    "negative": 272,
    "positive": 12094,
    "zero": 62
  },
  "has_unit": {
    "False": 8479,
    "True": 3949
  },
  "is_hard_type": {
    "False": 10286,
    "True": 2142
  },
  "length_bucket": {
    "129_256": 6213,
    "257_512": 3430,
    "gt512": 335,
    "le128": 2450
  },
  "problem_type": {
    "algebra": 882,
    "arithmetic_word_problem": 9627,
    "combinatorics_probability": 284,
    "geometry": 1134,
    "number_theory": 501
  }
}
```

## External curriculum

```json
{
  "accepted_rows": 50000,
  "problem_sources": {
    "augmented_gsm8k": 8680,
    "augmented_math": 40372,
    "gsm8k": 312,
    "math": 636
  },
  "removal_reasons": {
    "competition_exact_duplicate": 685,
    "competition_near_duplicate": 182,
    "competition_template_duplicate": 237,
    "external_internal_exact_duplicate": 1372,
    "external_internal_template_duplicate": 486,
    "non_english_or_abnormal_unicode": 30,
    "non_single_numeric_answer": 12551,
    "question_length": 440,
    "solution_length": 317,
    "tool_or_code_dependent": 231
  },
  "removed_rows": 16531,
  "source_rows_seen": 66531
}
```
