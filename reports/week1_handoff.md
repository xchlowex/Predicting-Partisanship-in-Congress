# Week 1 handoff · Predicting Partisanship in Congress

**23 November 2026.** Week 2 can start with 19,824 scored speeches and two fixed evaluation protocols. The full research corpus contains 20,000 speeches by 874 legislators. It includes 176 speeches from three legislators with other documented affiliations for descriptive analysis. Only the three CPU baselines have been fitted.

## Inputs for Week 2

| File | Use |
| --- | --- |
| [Frozen corpus by Congress](../data/week1/by_congress/) | Running `python -B scripts/verify_week1.py` restores `data/week1/corpus.jsonl` with one cleaned speech per line. `speech_id` is edition-qualified, `canonical_speaker_id` is a BioGuide ID and `label` is dated to the speech. Use only the text and the metadata allowed for the experiment as model inputs. |
| [splits.csv](../data/week1/splits.csv) | For D and R, join by `speech_id` and use either `speaker_split` or `temporal_split`. Each has fixed train, dev and test assignments. The temporal manifest also identifies speakers previously seen in training. |
| [windows.csv](../data/week1/windows.csv) | Content-token offsets for the frozen RoBERTa-base tokenizer. Every window inherits the partition of its speech. Average window probabilities to score the speech. |
| [metrics.csv](../results/week1/metrics.csv) and [predictions.csv](../results/week1/predictions.csv) | Baseline targets and per-speech probabilities for identical held-out IDs. [Class metrics](../results/week1/class_metrics.json) gives per-class precision/recall and confusion matrices. |

## Source and affiliation coverage

Use the Stanford daily edition for Congresses 110–114. The last available floor date in the 114th Congress data is **9 September 2016**. The Hugging Face release reformats Stanford's bound edition. Its preview matches the original archive, and we measured exact cross-edition text overlap in both overlapping study Congresses. Using one edition throughout the study prevents counting the same speech twice across sources. [Source evidence](week1_source_overlap.md) · [Raw archive audit](week1_data_readiness.md)

| Filtering stage for the five Congresses | Speeches |
| --- | ---: |
| Raw daily speech IDs | 752,103 |
| Eligible House/Senate floor speeches with text, a description and a valid date | 685,753 |
| Matched to a member in Stanford | 460,522 |
| Member and party resolved for the speech date under the published linkage rules | 456,477 |
| Passed the minimum length and text quality filters | 257,098 |
| Distinct normalized texts eligible for sampling | 255,932 |
| Frozen sample | 20,000 |

**Dated-label coverage is 66.57% of eligible floor speeches, or 99.12% of Stanford member matches.** The first denominator includes 225,231 speeches without a unique source member. An additional 4,045 records were excluded after matching: 3,621 nonvoting delegate speeches, 245 outside verified service dates, 178 without a matching roster surname/seat and one on an uncertain switch date. These are rules-based coverage rates. [Counts by Congress and chamber](../data/week1/preprocessing_counts.csv) · [Reason-specific counts](../data/week1/identity_status_counts.csv)

The dated join matches names and seat to a BioGuide ID using pinned term rosters, corroborating snapshots where available, and official records for party changes. In the final sample, Arlen Specter has 41 Republican and 13 Democratic speeches under the same BioGuide ID. Ordinary term-level matches require agreement with Stanford's source party initial. Some intervals rely on the term roster where an additional dated snapshot was unavailable. [Reference, evidence and exclusions](../data/reference/README.md)

A separate check sampled one frozen speech in each Congress/chamber combination. All ten matched the government sources examined. An additional check of all 404 Stanford speaker-mapped speeches by Specter and Parker Griffith in Congress 111 reproduced the expected dated labels and the single withheld transition-date record. [Independent identity checks](week1_identity_spotcheck.md) · [Detailed corpus methods](week1_corpus.md)

## Frozen preprocessing and splits

* Select 4,000 speeches per Congress, allocating chamber quotas proportionally to the eligible, exact-deduplicated pool. Use deterministic hash priorities and cap each legislator at 30 speeches per Congress. This defines a time-balanced study sample and affects representativeness.
* Normalize whitespace, remove an explicitly repeated structural speaker heading, require at least 40 whitespace-delimited words and 70% alphabetic nonspace characters. Remove 1,166 normalized exact duplicates. Screen final candidates for Jaccard similarity of at least 0.90 over five-token shingles; zero further removals were needed. In-speech party mentions and longer procedural passages remain for the later robustness analysis. OCR and transcription artifacts can survive these filters. For example, the frozen speech `hein-daily:1100157061` contains the fragmented wording “Sertoday … geant”.
* Score D (10,463) and R (9,361). Preserve I (117 speeches, two legislators) and Independent Democrat (59 speeches, one legislator) for descriptive analysis. These rare groups lack enough distinct legislators for the proposed cross-speaker evaluation.
* Frozen RoBERTa input comprises 31,213 windows for all 20,000 speeches. Each contains up to 254 content tokens plus two special tokens. The 6,455 speeches longer than 508 content tokens keep the first and last 254. The two-epoch training workload is 43,368 window presentations for the speaker split or 37,248 for the temporal split, before validation. The full-speech TF-IDF baseline sees more text on long speeches. [Tokenizer and window policy](../data/reference/roberta-base/README.md)

| Protocol | Train | Dev | Test | Validation |
| --- | ---: | ---: | ---: | --- |
| Speaker-disjoint | 13,893 | 2,965 | 2,966 | 610 / 130 / 131 legislators. Zero pairwise overlap by BioGuide ID. |
| Chronological | 11,886 | 3,968 | 3,970 | Train on Congresses 110–112, develop on 113, test on 114. All dates strictly ordered across partitions. |

Within the temporal test set, 3,118 speeches are from 353 legislators seen in training and 852 speeches are from 147 legislators unseen in training. Use the manifest's `temporal_speaker_history` column to distinguish members seen only in development. [Split counts, dates and hashes](../data/week1/split_summary.json)

## Held-out baseline results

The cells show **macro-F1** with 95% speaker-bootstrap intervals, followed by accuracy. Each evaluation uses the same speech IDs for all three models. Vectorizer, year scaling and model fitting use the training partition exclusively. Intervals resample legislators 1,000 times and are conditional on the fitted model.

| Baseline | Unseen legislators · macro-F1 [95% CI] / accuracy | Later Congress · macro-F1 [95% CI] / accuracy |
| --- | --- | --- |
| Training majority | 0.346 [0.285, 0.397] / 0.529 | 0.318 [0.289, 0.345] / 0.466 |
| TF-IDF + logistic regression | 0.710 [0.675, 0.738] / 0.713 | 0.607 [0.579, 0.631] / 0.615 |
| Year + chamber logistic regression | 0.524 [0.451, 0.597] / 0.548 | 0.318 [0.289, 0.345] / 0.466 |

TF-IDF leads on both protocols. Within Congress 114 its macro-F1 was 0.620 on returning legislators and 0.561 on legislators unseen in training. The two protocols use different training and test populations, so their score difference alone does not isolate a causal effect of time.

The metadata-only model predicted D for all 3,970 temporal test speeches, including 2,118 Republican speeches. For example, on the 8 January 2015 House speech about H.R. 39 (`hein-daily:1140000808`), it gave D probability 0.530. TF-IDF gave the documented R label probability 0.775. This pair illustrates what the baselines do. Week 2's text-versus-text-plus-metadata comparison will test how RoBERTa responds on the same speech. [Settings and hashes](../results/week1/run.json)

## Reproduction

From the repository root in PowerShell, the first three commands rebuild the cohort and model inputs. The project virtual environment supplies the pinned tokenizer. The CPU baseline uses the installed Python with scikit-learn 1.8.0, SciPy 1.17.1 and NumPy 2.4.3. The last command independently checks hashes, partition membership, window bounds, per-speech evaluation coverage and reported aggregate scores.

```powershell
python -B scripts/build_week1_corpus.py
python -B scripts/week1_splits.py
& '..\.venv\Scripts\python.exe' -B scripts/week1_windows.py
python -B scripts/week1_baselines.py
python -B scripts/verify_week1.py
```

An independent build from the cached raw joins reproduced the corpus, preprocessing and exclusion counts, duplicate list, split manifest and split summary byte for byte. The original Stanford archives are in the sibling `stanford_raw` directory. The frozen dated roster and tokenizer are in `data/reference`. The local SHA-256 of the study's `hein-daily.zip` is `0d44564769d826f3c1af40410e9d922d397f8143e2fa039364cbb010b0729c9d`. Gentzkow, Shapiro and Taddy compiled the *Congressional Record for the 43rd–114th Congresses: Parsed Speeches and Phrase Counts*, distributed by Stanford Libraries under ODC-BY 1.0. [Dataset and licence](https://data.stanford.edu/congress_text)
