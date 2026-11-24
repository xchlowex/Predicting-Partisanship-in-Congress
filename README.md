# Predicting-Partisanship-in-Congress
Group project for APAI4011. The goal is to predict the documented party affiliation of speakers from congressional speeches, allowing Democratic, Republican, Independent and other affiliations supported by the data.

## Week 3 team: start here

All 16 Week 2 runs are complete. The [Week 3 handoff guide](reports/week3_handoff.md) explains how to verify the matched error analysis and inspect representative cases. The [Week 3 comparison report](reports/week3_comparison.md) contains the matched text-versus-chamber comparison, error groups and lexical cues.

The Week 3 outputs are available offline:

```powershell
python -B scripts/week3_analysis.py
python -B scripts/verify_week3.py
python -B scripts/week3_plot.py
```

The analysis publishes matched development and test predictions, 80 representative speech cases and TF-IDF lexical cues under `results/week3/error_analysis/`. Party-name masking and phrase-removal inference are prepared in `scripts/week3_robustness.py`, but require the retained Week 2 checkpoints under `checkpoints/week2_formal/`.

## Week 1 handoff

The [handoff report](reports/week1_handoff.md) describes the dated affiliation checks, frozen study population, evaluation splits, baseline results and reproducibility commands. The repository distributes the [speech corpus in five compressed files](data/week1/by_congress/), one per Congress. Join the restored corpus to the [split manifest](data/week1/splits.csv) on `speech_id`. Use `canonical_speaker_id` to group speeches by legislator.

## What the team shares

| Stage | Commit to the private team repository | Keep on the workstation |
| --- | --- | --- |
| Week 1 | The 20,000-speech frozen corpus, split and window manifests, dated roster, tokenizer and licence, preprocessing and audit scripts, baseline settings, metrics, per-speech predictions and reports | Original Stanford ZIPs, extracted editions, downloaded roster snapshots, intermediate joins and the Python environments |
| Week 2 | Training and evaluation code, run configurations with seeds and model revision, the Week 1 input hashes, final metrics, per-speech predictions, error examples and figures used in the presentation | Downloaded model weights, intermediate and final checkpoints, optimizer states, experiment caches and detailed training logs |

After cloning, run `python -B scripts/verify_week1.py` from the repository root. It restores `data/week1/corpus.jsonl` if missing and verifies the corpus hash, splits and baseline results offline. The original Stanford archives are needed only to rebuild the dataset from source. For Week 2, save the small shared outputs in `results/week2/` and keep checkpoints under `checkpoints/`.

## Week 2 text-only pilot

From the repository root on the training workstation, run:

```powershell
& '..\.venv\Scripts\python.exe' -B scripts/week2_text_pilot.py
```

The entry point loads the pinned [RoBERTa-base weights](https://huggingface.co/FacebookAI/roberta-base/resolve/e2da8e2f811d1448a5b465c236feacd80ffbac7b/model.safetensors), verifies their SHA-256, and fine-tunes the encoder and a two-class head. For offline use, place that file in `models/roberta-base/model.safetensors`. The pilot uses 256 training and 128 development speeches sampled proportionally by party from the frozen speaker-disjoint partitions (seed 4011). It runs one epoch with AdamW at 2e-5 and an effective batch of 16 speeches. Training and evaluation both average window probabilities within each speech.

The command prints a new directory under `results/week2/` containing the sample IDs, development predictions, metrics and run settings. Verify it with `python -B scripts/week2_pilot_results.py --verify <run-directory>`. The checkpoint stays under `checkpoints/`. An explicit `--check-pipeline` run uses random initialization and writes its diagnostic outputs under `runs/week2/`. Both modes use train and development data only.

## Full Week 2 experiments

Run the full grid from the repository root with the same project environment:

```powershell
& '..\.venv\Scripts\python.exe' -u -B scripts/week2_formal.py
```

The queue first fits **text only, text + both, text + year and text + chamber** under the speaker-disjoint and chronological protocols with seed 4011. It then repeats text only and text + both with seeds 4012 and 4013. All 16 configurations start from the same verified pretrained encoder. Each run uses the full frozen training partition and three epochs. The checkpoint with the highest development macro-F1 is evaluated on that protocol's test speeches. The earliest epoch wins a tie.

Each speech has equal weight in the cross-entropy loss. The model averages its window probabilities before evaluating the loss. Year is standardized using training speeches for the current protocol. Chamber is encoded as House = 0 and Senate = 1. Each selected metadata feature adds a learned offset to each class logit. The two-class metadata weights start at zero. AdamW uses a constant learning rate of 2e-5 for RoBERTa and its text classification head, with weight decay 0.01. Metadata weights use 1e-3 and zero weight decay. Effective batches contain 16 speeches with up to four speeches per forward pass.

Use `--only speaker_text_s4011` to select a single run, or `--deadline` with a timezone-qualified ISO timestamp to limit runtime. Rerunning the same queue skips published results and resumes an interrupted run from its last complete epoch. Completed runs have configurations, epoch scores, hashes and per-speech development and test predictions in `results/week2/formal/`. The selected checkpoint and any unfinished optimizer state stay under `checkpoints/week2_formal/`.

The following commands use the Week 1 analysis environment to verify the results and build the comparison report. `--require core` checks the eight primary runs. Use `--require all` after all 16 runs finish.

```powershell
python -B scripts/verify_week2_formal.py --require core --check-checkpoints
python -B scripts/week2_compare.py
```

## Dataset
This project uses the [US-congress dataset](https://data.stanford.edu/congress_text) from Stanford University. The dataset contains congressional speeches and their corresponding party affiliations.  
The following data files are not uploaded to GitHub due to their size, but can be downloaded from the Stanford University website:
- `hein-bound`
- `hein-daily`
- `vocabulary`

