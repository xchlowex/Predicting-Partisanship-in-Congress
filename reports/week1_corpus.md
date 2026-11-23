# Frozen U.S. speech corpus

The study uses Stanford's daily edition for Congresses 110–114. Each Congress contributes 4,000 speeches. The source includes House and Senate floor records from 4 January 2007 through 9 September 2016 across this period. The available 114th Congress data end on 9 September 2016. Choosing one edition prevents duplicate coverage of the same proceedings from the daily and bound editions. The released corpus contains Stanford records only. The proposal's Hugging Face source and conditional GovInfo records are excluded from this version.

## Selection and label coverage

| Stage in the selected five Congresses | Speeches |
| --- | ---: |
| Raw daily-edition speech IDs | 752,103 |
| Nonempty floor speech, valid date, H/S chamber | 685,753 |
| Unique Stanford member row in the same chamber | 460,522 |
| Unique canonical ID with a verified dated affiliation | 456,477 |
| After text-length and character-quality filters | 257,098 |
| After normalized exact-text deduplication | 255,932 |
| Final sample | 20,000 |

The dated linkage covers 66.57% of eligible floor speeches and 99.12% of the Stanford member-matched subset. Among the 460,522 source matches, 4,045 records were unresolved or excluded by the dated resolver. `identity_status_counts.csv` breaks these down by reason, Congress and chamber. Unknown affiliation never receives an Independent label. `preprocessing_counts.csv` gives stage and class counts by Congress and chamber; rows labeled `other` record texts outside the H/S denominator.

| Dated affiliation | Speeches | Distinct BioGuide IDs |
| --- | ---: | ---: |
| Democrat (D) | 10,463 | 417 |
| Republican (R) | 9,361 | 455 |
| Independent (I) | 117 | 2 |
| Independent Democrat (ID) | 59 | 1 |

There are 874 distinct BioGuide IDs overall. A person who changes party can appear in multiple class totals. The primary comparison uses 19,824 D/R speeches. I and ID remain available for descriptive reporting. The full file contains 12,567 House speeches and 7,433 Senate speeches.

## Text, sampling and duplicates

- Use the source's separate speech field. Collapse whitespace and strip a leading literal copy of the source speaker header when clearly terminated with a colon or period. Two source headers were stripped in the mapped cohort. Addresses such as “Mr. Speaker” and references to parties inside a speech remain available for the planned masking experiment.
- Require at least 40 whitespace-delimited tokens and at least 70% ASCII alphabetic characters among characters other than spaces. These reproducible quality filters remove 199,291 short speeches and 88 further speeches. Longer procedural interventions can still qualify. The sample therefore represents sustained floor interventions better than brief procedural exchanges.
- Normalize to lowercase ASCII letter/digit tokens for duplicate detection only. Deduplicate globally across the full eligible five-Congress pool by SHA-256 of the normalized tokens. Keep the first source occurrence, in Congress then file order. This removed 1,166 repeats before sampling.
- Allocate the 4,000 slots per Congress proportionally to its exact-unique House and Senate candidates, rounded to integers. Within those quotas, visit candidates in ascending SHA-256 priority of `APAI4011-week1-v1|hein-daily:<source_speech_id>`. Keep at most 30 speeches per canonical speaker per Congress. Of the priority candidates visited before filling quotas, 1,921 were skipped by this cap. Speeches outside the quotas or cap remain outside the experimental sample.
- Check every accepted speech against earlier accepted speeches across all Congresses and chambers **before splitting**. Two documents qualify as near duplicates if the Jaccard similarity of their sets of contiguous five-token shingles is at least 0.90. A shared-prefix index generates candidates under a single vocabulary order; its prefix-length bound preserves all pairs at the threshold. Candidate pairs are checked with exact set intersection. After exact-text deduplication, no additional accepted pair met the threshold. Shared slogans, quotations or boilerplate covering only part of a speech may remain.

Only `text` is presented to text-only models. The named metadata variants additionally receive `year` and/or `chamber`. The other fields in `data/week1/corpus.jsonl`, including both speaker IDs and the dated label, support splitting, provenance and evaluation.

## Reproduce

From the repository root, run `python scripts/build_week1_corpus.py`. It reads `../stanford_raw/hein-daily.zip` directly. The local join cache in `../stanford_raw/week1_build` avoids extracting the source ZIP. For repeated work on this exact archive, `--reuse-stage` reuses the cache after checking the source members' size/CRC inventory.

Data source: Matthew Gentzkow, Jesse M. Shapiro and Matt Taddy, *Congressional Record for the 43rd–114th Congresses: Parsed Speeches and Phrase Counts*. Stanford Libraries distributes the data under ODC-BY 1.0. [Stanford data and licence](https://data.stanford.edu/congress_text)
