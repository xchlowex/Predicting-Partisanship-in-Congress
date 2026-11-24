# Week 3 · Robustness and error analysis

**23 November 2026 · U.S. congressional speeches.** This report compares matched text-only and text-plus-chamber predictions on the fixed development and test partitions. Development choices are separated from the held-out test check.

## Matched model comparison

![Matched temporal text-versus-chamber comparison](../results/week3/error_analysis/model_comparison.png)

Each cell gives macro-F1 [95% speaker-bootstrap interval] / accuracy.

| Partition | Text only | Text + chamber |
|---|---|---|
| Dev | 0.702 [0.679, 0.724] / 0.710 | 0.722 [0.703, 0.740] / 0.723 |
| Test | 0.622 [0.591, 0.647] / 0.632 | 0.658 [0.632, 0.681] / 0.659 |

## Matched error groups

| Partition | Both correct | Chamber only correct | Text only correct | Both wrong |
|---|---:|---:|---:|---:|
| Dev | 2512 | 357 | 306 | 793 |
| Test | 2196 | 419 | 313 | 1042 |

## Representative cases

The file `representative_cases.csv` contains 80 selected speech IDs with the source text and both model decisions. Selection used development categories first; test rows were included only after the category rules were fixed.

## Lexical cues

`lexical_cues.csv` ranks TF-IDF logistic-regression features separately for the temporal and speaker-disjoint protocols. These cues describe the baseline model and are not a direct explanation of RoBERTa.

## Scope and pending robustness work

This release completes matched error analysis and lexical-cue analysis. Party-name masking and phrase-removal tests require the formal RoBERTa checkpoints and training workstation; checkpoint files are not included in this clone.

## Reproduction

Run `python -B scripts/week3_analysis.py` from the repository root using the Week 1 analysis environment.

The machine-readable outputs contain exact matched speech IDs, probabilities, confusion matrices, bootstrap intervals and ranked lexical cues.
[Matched predictions](../results/week3/error_analysis/)
