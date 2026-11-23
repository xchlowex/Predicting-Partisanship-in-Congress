# Week 1 speech source and edition selection

**Decision, 23 November 2026.** Use Stanford's daily Congressional Record as the speech source for Congresses 110–114. Retain the Hugging Face release and Stanford bound edition as provenance and overlap checks. Each analysis speech enters the study through the daily edition only.

## Hugging Face provenance

Evžen Wybitul identifies `hein-bound.zip` from Stanford as the input for `Eugleo/us-congressional-speeches`. His preprocessing script reads speech text and descriptions, joins on `speech_id`, and uploads the resulting table. It does not read Stanford's member maps. The Hub schema accordingly has speech, date, chamber and speaker-description fields, but no party or canonical legislator ID. The author's general description mentions Congresses 43–114. Stanford specifies 43–111 for the bound edition and 97–114 for the daily edition. This study uses Stanford's edition-specific coverage. [Author's instructions](https://github.com/Eugleo/emotion-in-political-language/blob/a5f732e3e9187c661f136a4b31fdcb14492ad573/README.md) · [Conversion script](https://github.com/Eugleo/emotion-in-political-language/blob/a5f732e3e9187c661f136a4b31fdcb14492ad573/scripts/data_preprocess.py) · [Stanford source](https://data.stanford.edu/congress_text)

Two independent checks support the documented provenance.

1. The full Hub release lists **17,395,884 rows**. Summing the distinct speech IDs per session in the local bound archive gives **17,395,884**. All 69 bound sessions have a one-to-one ID join between descriptions and speech texts.
2. All 100 consecutive records in the Hub's public preview (IDs 430000001–430000100, Congress 43) matched the local bound edition on ID and all 13 description fields. All 72 fully displayed speech texts matched after whitespace normalization. For the other 28, the complete displayed prefix matched. This preview checks the published transformation at the beginning of the release. It does not establish row-by-row identity across every session. The decision draws on the conversion recipe, archive count and preview together. [Hub dataset and preview](https://huggingface.co/datasets/Eugleo/us-congressional-speeches) · [Hub revision](https://huggingface.co/datasets/Eugleo/us-congressional-speeches/commit/51776342f69868c2cc9a82dc74497d98f62008cb)

## Overlap inside the study period

For Congresses represented in both Stanford editions, we compared all House and Senate speech texts containing at least 30 whitespace-delimited words. We indexed each by date, chamber and SHA-256 of its complete text after ASCII lowercasing and whitespace normalization. Each matching daily row contributes once to the numerator. Different remarks phrased identically on the same day can match, and edits, OCR errors and segmentation changes can conceal links between editions. The figures quantify exact text reuse under this rule.

| Congress | Eligible bound rows | Eligible daily rows | Distinct shared date/chamber/text fingerprints | Daily rows matching a bound fingerprint | Share of eligible daily rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| 110 | 89,690 | 87,100 | 21,848 | 22,028 | 25.29% |
| 111 | 77,839 | 64,453 | 28,494 | 28,664 | 44.47% |

Congresses 112–114 have no bound-edition counterpart in the Stanford release. Keeping one edition throughout all five Congresses avoids duplicating these records across training and evaluation and gives stable source formatting across periods. Within-edition duplicate cleaning remains necessary. GovInfo is a conditional extension in the proposal. The Stanford daily release already covers the selected period through Congress 114, so no GovInfo speeches enter the study. [Official GovInfo daily record](https://www.govinfo.gov/app/collection/CREC)

## Reproduction

From the repository root, run `python scripts/audit_week1_sources.py` with the two verified archives in the sibling `stanford_raw` directory. The script compares the public preview snapshot against the local archive and recomputes the two cross-edition counts. The snapshot contains only fingerprints, IDs, dates, chambers and lengths. Preview text was stripped of the viewer truncation marker and whitespace-normalized. The metadata fingerprint uses UTF-8 compact JSON in the original description-column order, normalized to integer counts and YYYYMMDD dates. [Preview fingerprints](week1_hf_preview_fingerprints.csv)
