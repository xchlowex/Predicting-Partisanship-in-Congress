# Week 3 handoff · Robustness and error analysis

**23 November 2026 · U.S. congressional speeches.** The first Week 3 deliverable compares matched RoBERTa predictions and publishes representative cases.

## Start here

Run:

```powershell
python -B scripts/week1_archive.py restore
python -B scripts/week3_analysis.py
python -B scripts/verify_week3.py
```

## Which file answers which question?

| Team task | Files to open | What to use |
|---|---|---|
| Write the robustness section | [Week 3 comparison](week3_comparison.md) | Matched text-versus-chamber results, confidence intervals and error groups. |
| Inspect exact decisions | [Development matches](../results/week3/error_analysis/dev_matched_predictions.csv) · [Test matches](../results/week3/error_analysis/test_matched_predictions.csv) | One row per frozen speech with both probabilities and the gold label. |
| Read representative speeches | [Representative cases](../results/week3/error_analysis/representative_cases.csv) | Development-selected examples and fixed test examples with source text. |
| Inspect lexical cues | [Lexical cues](../results/week3/error_analysis/lexical_cues.csv) | TF-IDF/logistic-regression cues, not RoBERTa attributions. |

## Use development choices to protect test validity

The category rules are determined from development speeches. Apply those fixed rules to test speeches without changing them in response to test performance.

## Remaining Week 3 work

Use the [checkpoint and inference guide](week3_weights.md) to install the selected models and score original, masked and phrase-removed speeches. Add the results to this report, preserve the held-out test as the final check, and prepare the final presentation.
