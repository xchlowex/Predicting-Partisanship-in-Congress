# Week 1 data readiness

**23 November 2026 · Week 1 handoff complete.** All three archives in the README are present in the sibling `stanford_raw` directory. The required inventory and every ZIP member's CRC passed. Total archive size is 10,449,651,436 bytes. The [final handoff](week1_handoff.md) records the dated cohort, frozen splits and baseline scores.

## Published reference statistics and bundled audit samples

| Edition | Congresses in the official release | Local audit dates | Distinct speech IDs in local audit sample | Speeches mapped to a member in Stanford's published statistics |
| --- | --- | ---: | ---: | ---: |
| Bound | 43–111 | 56 | 2,482 | 13,311,255 / 13,623,972 (97.70%) |
| Daily | 97–114 | 28 | 324 | 2,585,807 / 2,614,163 (98.92%) |

The sample counts come from `data/audit/audit/parsing/auto/<edition>/speeches_*.txt`. The percentages use sums of the `mapped` and `unmapped` columns in `data/speakermap_stats/`, weighted by speech count. They measure Stanford's upstream member matching. The team's separate speech-date affiliation results appear in the handoff. Each edition has its own speech IDs, and Congresses 97–111 occur in both editions.

## Local full-archive audit

The audit completed across 87 edition–Congress combinations and produced 566 chamber/party support rows. Eligibility requires nonempty speech text, exactly one text row and description, a valid date and a House or Senate chamber. Titled speakers remain in the denominator. The final column counts eligible speeches with a unique source member match and exactly one party name for that Congress and party code.

| Edition | Eligible speeches | Source member matches | Unique session-end source party |
| --- | ---: | ---: | ---: |
| Bound | 16,766,379 | 13,216,831 | 12,700,992 |
| Daily | 3,592,118 | 2,572,390 | 2,411,166 |

## Week 1 acceptance

| Deliverable | Finding | Handoff evidence |
| --- | --- | --- |
| Stanford source files | Verified | Bound (345 required files), daily (90) and vocabulary (3) all passed the inventory and CRC checks. |
| Labeled study corpus | Complete for the selected cohort | 20,000 dated speeches, 874 BioGuide IDs, four documented source labels; D/R scored. |
| Preprocessing and overlap | Complete | One source edition, fixed text filters, exact/near duplicate checks, a frozen tokenizer and per-speech windows. |
| Frozen split manifests | Complete | Pairwise speaker overlap is zero in the speaker-disjoint protocol. Congress 113 is temporal development and 114 is temporal test. |
| Frozen baselines | Complete | CPU baseline settings, held-out scores and probabilities are in [results/week1](../results/week1/). |

## Two label checks that affect acceptance

1. Stanford's codebook documents session-end affiliation (p. 6) and an original speaker ID containing session, member identifier and chamber (p. 11). Verify switchers against dates or flag those records. Validate cross-session and cross-chamber identity before assigning people to splits.
2. A party initial can have several meanings within one Congress. In Congress 43, `data/party_full/party_full.txt` associates `I` with both `ind. democrat` and `ind. republican`. Resolve affiliation using member-level evidence before reporting an Independent class. Keep an unresolved label separate.

The [source comparison](week1_source_overlap.md) documents the Hugging Face conversion, measured overlap between Stanford editions and the decision to use the daily edition. The cohort falls within Stanford's date coverage.

## Reproduce the checks

Run these commands in PowerShell from the repository root. The downloader resumes partial downloads, checks the CRCs and inventory, and keeps the archives in the sibling `stanford_raw` directory. The audit reads the archives directly. It writes `session_linkage.csv` and `party_support.csv` under `stanford_raw/audit`. Use `--check-only` with the downloader to verify existing ZIPs offline.

```powershell
python scripts/download_stanford.py
if ($LASTEXITCODE -eq 0) { python scripts/audit_stanford.py }
```

The CSVs describe source-level joins by edition, Congress and chamber. The [cohort processing counts](../data/week1/preprocessing_counts.csv) give the corresponding dated-label and exclusion counts for the selected period.

Source: Gentzkow, Shapiro and Taddy, *Congressional Record for the 43rd–114th Congresses*. The landing page supplies the downloads, citation and ODC-BY 1.0 terms. [Stanford Libraries](https://data.stanford.edu/congress_text)
