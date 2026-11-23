# Predicting-Partisanship-in-Congress
Group project for APAI4011. The goal is to predict the documented party affiliation of speakers from congressional speeches, allowing Democratic, Republican, Independent and other affiliations supported by the data.

## Week 1 handoff

The [handoff report](reports/week1_handoff.md) describes the dated affiliation checks, frozen study population, evaluation splits, baseline results and reproducibility commands. The repository distributes the [speech corpus in five compressed files](data/week1/by_congress/), one per Congress. Join the restored corpus to the [split manifest](data/week1/splits.csv) on `speech_id`. Use `canonical_speaker_id` to group speeches by legislator.

## What the team shares

| Stage | Commit to the private team repository | Keep on the workstation |
| --- | --- | --- |
| Week 1 | The 20,000-speech frozen corpus, split and window manifests, dated roster, tokenizer and licence, preprocessing and audit scripts, baseline settings, metrics, per-speech predictions and reports | Original Stanford ZIPs, extracted editions, downloaded roster snapshots, intermediate joins and the Python environments |
| Week 2 | Training and evaluation code, run configurations with seeds and model revision, the Week 1 input hashes, final metrics, per-speech predictions, error examples and figures used in the presentation | Downloaded model weights, intermediate and final checkpoints, optimizer states, experiment caches and detailed training logs |

After cloning, run `python -B scripts/verify_week1.py` from the repository root. It restores `data/week1/corpus.jsonl` if missing and verifies the corpus hash, splits and baseline results offline. The original Stanford archives are needed only to rebuild the dataset from source. For Week 2, save the small shared outputs in `results/week2/` and keep checkpoints under `checkpoints/`.

## Dataset
This project uses the [US-congress dataset](https://data.stanford.edu/congress_text) from Stanford University. The dataset contains congressional speeches and their corresponding party affiliations.  
The following data files are not uploaded to GitHub due to their size, but can be downloaded from the Stanford University website:
- `hein-bound`
- `hein-daily`
- `vocabulary`

