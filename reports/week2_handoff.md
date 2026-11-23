# Week 2 handoff to the Week 3 team

**23 November 2026 · U.S. congressional speeches.** All 16 planned RoBERTa runs are complete. They predict a legislator’s documented Democratic or Republican affiliation from a speech. The deliverables support error analysis on the frozen evaluation sets and supply the main results figure for the presentation.

## Start on a teammate’s computer

From an existing clone, update `main` with `git pull --ff-only origin main` while on that branch. New team members can clone the repository or extract the handoff ZIP. Open a terminal in the folder containing `README.md` and `scripts/`. Run the following with Python 3.11 or newer.

```powershell
python -B scripts/verify_week1.py
python -B scripts/verify_week2_formal.py --require all
```

The first command restores `data/week1/corpus.jsonl` from five compressed Congress files and checks the Week 1 inputs and baselines. The second verifies all 16 selected epochs, prediction-file hashes, speech IDs and recomputed development and test scores. Both commands use the Python standard library and run offline on a CPU. The training workstation also passed the optional `--check-checkpoints` check.

## Which file answers which question?

| Team task | Files to open | What to use |
|---|---|---|
| Write the results section or make slides | [Formal comparison](week2_comparison.md) · [Presentation figure](../results/week2/formal_summary/model_comparison.png) | The figure shows seed 4011, test macro-F1 and 95% speaker-bootstrap intervals. The report also compares baselines and speakers seen or unseen in chronological training. |
| Retrieve exact values | [Comparison data](../results/week2/formal_summary/comparison.json) | `metrics` contains per-class scores and confusion matrices. `paired_differences` gives matched comparisons. `seed_summary` gives mean and sample SD. Select protocol, partition and subgroup before quoting a result. |
| Inspect individual decisions | [Formal run folders](../results/week2/formal/) | Each folder contains `run.json`, `dev_predictions.csv` and `test_predictions.csv`. Directory names encode protocol, variant and seed, e.g. `temporal_both_s4012`. Each CSV row has a speech ID, speaker ID, gold and predicted labels, and probabilities `p_D` and `p_R`. |
| Recover wording, chamber and period | [Frozen corpus shards](../data/week1/by_congress/) · [Split manifest](../data/week1/splits.csv) · [Window manifest](../data/week1/windows.csv) | The restored corpus supplies `text`, `year`, `congress` and `chamber`. The split file supplies both protocol assignments and temporal speaker history. For long speeches, consult the window manifest to see what RoBERTa could read. |
| Compare a baseline or inspect lexical features | [Week 1 predictions](../results/week1/predictions.csv) · [Baseline settings](../results/week1/run.json) · [Baseline code](../scripts/week1_baselines.py) | Filter the baseline CSV to the same protocol, partition and model. To inspect TF-IDF coefficients, fit the vectorizer and classifier on that protocol’s frozen training speeches with the saved settings. |
| Reproduce a model decision on edited text | [Run settings](../results/week2/formal/) · [Model code](../scripts/week2_model.py) · [Evaluation code](../scripts/week2_formal.py) | Use the selected checkpoint path and SHA-256 in the relevant `run.json`. Weights, base model and training environment are retained on the training workstation. Arrange access for new forward passes such as phrase removal or party-name masking. |

## Results to carry forward

| Test population | Test speeches | Highest primary-seed macro-F1 | Text only, three-seed mean | Text + year + chamber, three-seed mean |
|---|---:|---|---:|---:|
| Held-out legislators (speaker-disjoint split) | 2,966 | Text + year, **0.7367** | 0.7004 | 0.7074 |
| 114th Congress (chronological split) | 3,970 | Text + chamber, **0.6582** | 0.6372 | 0.6484 |

Seed 4011 covers all four input variants in both protocols. Seeds 4012 and 4013 repeat text only and text + year + chamber. Adding both metadata features improved test macro-F1 in two of three matched seeds in each protocol. The rankings of the individual year and chamber variants apply to seed 4011. The speaker-bootstrap intervals describe uncertainty for a fitted model. The three-seed SD describes variation across fits.

## Begin error analysis with a matched comparison

Start with `temporal_text_s4011/dev_predictions.csv` and `temporal_chamber_s4011/dev_predictions.csv` under `results/week2/formal/`. For each CSV, use `speech_id` as the unique key. Match the two predictions on this key and confirm the gold label and `canonical_speaker_id` agree. Join to the restored corpus and split manifest on the same key. This pair should yield **3,968 matched development speeches**, all with `temporal_split = dev`.

Group the joined records by whether both models were right, both were wrong, or only one was right. Within the disagreement groups, inspect speech text, chamber, year, gold label and both models’ probabilities. Save the speech IDs and the selection rule for any examples quoted. Use development speeches to choose categories or revise methods. Apply the fixed analysis to the held-out test speeches for reporting. The chronological test contains a single Congress; period comparisons should identify the Congresses and partitions used.

For the full Week 3 scope, see sections 4 and 5 of the [project proposal](../APAI4011%20Proposal.pdf). The assigned work includes robustness and error analysis, followed by the final report and presentation. The published probability files support case selection and subgroup summaries immediately. Measuring how predictions change after editing a speech requires another inference pass with the fitted model.
