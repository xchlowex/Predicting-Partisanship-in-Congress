# Week 3 model weights and edited-speech inference

The handoff contains the development-selected checkpoint for each of the 16 formal Week 2 runs. Each file is an exact copy of the checkpoint recorded in its run's `run.json`, including the fitted classification head and any year or chamber coefficients. The tokenizer, architecture and training-time metadata transforms are already in this repository.

## Choose the runs

| Week 3 question | Run names |
|---|---|
| Does removing a cue affect the text-only and chamber models differently on the later Congress? | `temporal_text_s4011` and `temporal_chamber_s4011`. This pair matches the Week 3 error-analysis report. |
| Does that effect carry over to unseen legislators? | `speaker_text_s4011` and `speaker_chamber_s4011`. |
| Does the text versus year-and-chamber contrast persist across random seeds? | Use matching `text` and `both` runs for seeds 4011, 4012 and 4013 within the same protocol. |

Each selected file is about 499 MB. The first pair requires about 1.00 GB. All 16 together occupy 7,979,154,066 bytes. The [checksum list](../results/week2/formal_summary/checkpoint_sha256.txt) uses the release asset names. Each [formal run record](../results/week2/formal/) identifies the original local filename, selected epoch, class order and SHA-256.

## Install the first pair

From the repository root, use Python 3.11+ for the installer. GitHub release downloads also require a GitHub CLI login with read access to the private repository. Check whether the assets are published and then install the chosen pair:

```powershell
gh release view week2-formal-weights-v1 --repo xchlowex/Predicting-Partisanship-in-Congress
python -B scripts/week3_weights.py temporal_text_s4011 temporal_chamber_s4011
```

If using the workstation handoff folder, installation works offline:

```powershell
python -B scripts/week3_weights.py --from-dir ..\APAI4011_Week2_Weights temporal_text_s4011 temporal_chamber_s4011
```

Use `--all` in place of the run names to install all 16. The installer checks each SHA-256 before placing the file at the original `checkpoints/week2_formal/<run>/epoch<N>.pt` path. For a full installation, run `python -B scripts/verify_week2_formal.py --require all --check-checkpoints`.

## Score original and edited speeches

Use a Python environment with PyTorch and Transformers. The workstation used Python 3.14, torch 2.11.0+cu128, transformers 5.6.2 and tokenizers 0.22.2. For CPU execution, install a CPU-compatible PyTorch build. The saved checkpoint includes the encoder weights. The frozen tokenizer is loaded from the repository, so scoring can run offline.

Prepare one UTF-8 JSONL file per condition (original, party names masked, and each phrase-removal rule). Use the same rows in the same order, and carry the original year and chamber into every version. Each line needs a unique speech ID, text, integer year, and chamber `H` or `S`. For example, a single input line has this form:

```json
{"speech_id":"example-1","text":"My Republican colleagues supported the bill.","year":2014,"chamber":"H"}
```

For a masking condition, change only the relevant wording in `text` using the team's chosen masking rule. Record which speeches changed. If removing a phrase empties a speech, document how the case will be handled before evaluating test data. The scorer requires nonempty text.

```powershell
python -B scripts/week3_infer.py --run temporal_text_s4011 --input original.jsonl --output original_scores.csv
python -B scripts/week3_infer.py --run temporal_text_s4011 --input masked.jsonl --output masked_scores.csv
python -B scripts/week3_infer.py --run temporal_text_s4011 --input removed.jsonl --output removed_scores.csv
```

Repeat for `temporal_chamber_s4011` with distinct output filenames. The scorer tokenizes each edited speech afresh, applies the Week 2 windowing rule, averages probabilities across windows and writes `speech_id,predicted,p_D,p_R`. Join by `speech_id` within each run for paired comparisons. Fit any masking or phrase-selection decisions on development data and apply fixed rules to the test set. Running a checkpoint readiness check alone does not produce edited-speech predictions.

For a replay check, start with unedited development speeches and compare against that run's `dev_predictions.csv`. Use matching hardware, precision and batch order for bitwise comparisons. Four original development speeches from `speaker_both_s4011` reproduced both stored probabilities exactly on the training GPU. CPU float32 gave the same four labels with a maximum absolute probability difference of 0.00221. This check verifies the inference path; group-level masking and removal effects are for the Week 3 analysis.
