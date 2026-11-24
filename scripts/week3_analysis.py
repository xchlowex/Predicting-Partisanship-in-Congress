"""Publish Week 3 matched error analysis and lexical cues from frozen Week 2 outputs."""

import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from week1_archive import DATA, restore
from week1_baselines import BOOTSTRAPS, LABELS, ROOT, SEED


FORMAL = ROOT / "results/week2/formal"
OUTPUT = ROOT / "results/week3/error_analysis"
REPORT = ROOT / "reports/week3_comparison.md"
HANDOFF = ROOT / "reports/week3_handoff.md"
VARIANTS = ("text", "chamber")
PARTITIONS = ("dev", "test")


def csv_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def file_hash(path):
    with Path(path).open("rb") as stream:
        import hashlib
        return hashlib.file_digest(stream, "sha256").hexdigest()


def metric_rows(corpus, predictions, phase, protocol):
    gold = np.array([LABELS.index(row["gold"]) for row in predictions])
    pred = np.array([LABELS.index(row["predicted"]) for row in predictions])
    speakers = np.array([row["canonical_speaker_id"] for row in predictions])
    cm = np.zeros((2, 2), dtype=int)
    np.add.at(cm, (gold, pred), 1)
    f1 = np.array([2 * cm[i, i] / (2 * cm[i, i] + cm[i, 1-i] + cm[1-i, i])
                   if 2 * cm[i, i] + cm[i, 1-i] + cm[1-i, i] else 0
                   for i in range(2)])
    return cm, float(f1.mean()), float((pred == gold).mean()), speakers


def confusion_metrics(cm):
    f1 = []
    per_class = {}
    for i, label in enumerate(LABELS):
        tp = int(cm[i, i])
        support = int(cm[i].sum())
        selected = int(cm[:, i].sum())
        precision = tp / selected if selected else 0
        recall = tp / support if support else 0
        class_f1 = 2 * tp / (support + selected) if support + selected else 0
        f1.append(class_f1)
        per_class[label] = {"support": support, "precision": precision,
                            "recall": recall, "f1": class_f1}
    return sum(f1) / 2, (cm.diagonal().sum() / cm.sum()), per_class


def bootstrap_metrics(predictions, seed=SEED, replicates=BOOTSTRAPS):
    gold = np.array([LABELS.index(r["gold"]) for r in predictions])
    pred = np.array([LABELS.index(r["predicted"]) for r in predictions])
    _, inverse = np.unique([r["canonical_speaker_id"] for r in predictions], return_inverse=True)
    rng = np.random.default_rng(seed)
    draws = rng.integers(inverse.max() + 1, size=(replicates, inverse.max() + 1))
    clusters = np.zeros((inverse.max() + 1, 2, 2), dtype=int)
    np.add.at(clusters, (inverse, gold, pred), 1)
    values = []
    accuracies = []
    for matrix in clusters[draws].sum(axis=1):
        macro, accuracy, _ = confusion_metrics(matrix)
        values.append(macro)
        accuracies.append(accuracy)
    return float(np.quantile(values, .025)), float(np.quantile(values, .975)), float(np.quantile(accuracies, .025)), float(np.quantile(accuracies, .975))


def publish(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temp:
        staged = Path(temp) / path.name
        staged.write_text(content, encoding="utf-8", newline="\n")
        os.replace(staged, path)


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fit_lexical(protocol):
    corpus = restore(ROOT / "data/week1")
    splits = {r["speech_id"]: r for r in csv_rows(ROOT / "data/week1/splits.csv")}
    train, dev = [], []
    with corpus.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["label"] not in LABELS:
                continue
            split = splits[row["speech_id"]][protocol + "_split"]
            target = train if split == "train" else dev
            if split != "test":
                target.append(row)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=60000,
                                 sublinear_tf=True, norm="l2", lowercase=True)
    x = vectorizer.fit_transform([r["text"] for r in train])
    clf = LogisticRegression(C=1.0, solver="liblinear", tol=1e-4, max_iter=500, random_state=SEED)
    clf.fit(x, [r["label"] for r in train])
    names = np.array(vectorizer.get_feature_names_out())
    coefficients = clf.coef_[0]
    order = np.argsort(coefficients)
    cues = []
    for direction, indices in (("D", order[:25]), ("R", order[-25:][::-1])):
        for rank, index in enumerate(indices, 1):
            cues.append({"protocol": protocol, "direction": direction, "rank": rank,
                         "cue": names[index], "coefficient": float(coefficients[index])})
    return cues


def main():
    corpus_path = restore(ROOT / "data/week1")
    corpus = {r["speech_id"]: r for r in map(json.loads, corpus_path.open(encoding="utf-8"))}
    splits = {r["speech_id"]: r for r in csv_rows(ROOT / "data/week1/splits.csv")}
    comparison = []
    matched_files = {}
    for phase in PARTITIONS:
        predictions = {}
        for variant in VARIANTS:
            folder = FORMAL / f"temporal_{variant}_s4011"
            rows = list(csv_rows(folder / f"{phase}_predictions.csv"))
            for row in rows:
                if row["speech_id"] not in corpus:
                    continue
            predictions[variant] = {row["speech_id"]: row for row in rows}
        ids = set(predictions["text"]) & set(predictions["chamber"])
        matched = []
        for sid in sorted(ids):
            text = predictions["text"][sid]
            chamber = predictions["chamber"][sid]
            source = corpus[sid]
            if text["gold"] != chamber["gold"] or text["canonical_speaker_id"] != chamber["canonical_speaker_id"]:
                raise ValueError(f"Prediction mismatch for {sid}")
            text_right = text["predicted"] == text["gold"]
            chamber_right = chamber["predicted"] == chamber["gold"]
            if text_right and chamber_right:
                category = "both_correct"
            elif text_right and not chamber_right:
                category = "text_only_correct"
            elif chamber_right and not text_right:
                category = "chamber_only_correct"
            else:
                category = "both_wrong"
            matched.append({
                "speech_id": sid, "canonical_speaker_id": source["canonical_speaker_id"],
                "protocol": "temporal", "partition": phase, "congress": source["congress"],
                "chamber": source["chamber"], "year": source["year"],
                "temporal_speaker_status": splits[sid]["temporal_speaker_status"],
                "gold": text["gold"], "text_predicted": text["predicted"],
                "text_p_D": text["p_D"], "text_p_R": text["p_R"],
                "chamber_predicted": chamber["predicted"], "chamber_p_D": chamber["p_D"],
                "chamber_p_R": chamber["p_R"], "error_category": category,
                "text_right": text_right, "chamber_right": chamber_right,
            })
        matched_files[phase] = matched
        fields = list(matched[0])
        write_csv(OUTPUT / f"{phase}_matched_predictions.csv", fields, matched)

        for variant in VARIANTS:
            rows = [predictions[variant][r["speech_id"]] for r in matched]
            cm = np.zeros((2, 2), dtype=int)
            for row in rows:
                cm[LABELS.index(row["gold"]), LABELS.index(row["predicted"])] += 1
            macro, accuracy, per_class = confusion_metrics(cm)
            ci_f1_lo, ci_f1_hi, ci_acc_lo, ci_acc_hi = bootstrap_metrics(rows)
            comparison.append({
                "protocol": "temporal", "partition": phase, "model": variant,
                "speeches": len(rows), "speakers": len({r["canonical_speaker_id"] for r in rows}),
                "macro_f1": macro, "macro_f1_ci_low": ci_f1_lo, "macro_f1_ci_high": ci_f1_hi,
                "accuracy": accuracy, "accuracy_ci_low": ci_acc_lo, "accuracy_ci_high": ci_acc_hi,
                "confusion_gold_by_pred": cm.tolist(), "per_class": per_class,
            })

    examples = []
    for phase in ("dev", "test"):
        for category in ("chamber_only_correct", "text_only_correct", "both_wrong", "both_correct"):
            pool = [r for r in matched_files[phase] if r["error_category"] == category]
            pool.sort(key=lambda r: (r["chamber"], r["canonical_speaker_id"], r["speech_id"]))
            for rank, row in enumerate(pool[:10], 1):
                examples.append({**row, "selection_category": category,
                                 "selection_rank": rank,
                                 "text": " ".join(corpus[row["speech_id"]]["text"].split())[:1500]})
    write_csv(OUTPUT / "representative_cases.csv",
              list(examples[0]) if examples else [],
              examples)

    cues = fit_lexical("temporal") + fit_lexical("speaker")
    write_csv(OUTPUT / "lexical_cues.csv",
              ["protocol", "direction", "rank", "cue", "coefficient"], cues)

    result = {
        "bootstrap": {"unit": "canonical_speaker_id", "replicates": BOOTSTRAPS, "seed": SEED,
                      "interval": "percentile 95%"},
        "models": {"text": "RoBERTa, text", "chamber": "RoBERTa, text + chamber"},
        "comparison": comparison,
        "representative_cases": len(examples),
        "lexical_cues": len(cues),
        "source_prediction_sha256": {
            f"temporal_{variant}_{phase}": file_hash(FORMAL / f"temporal_{variant}_s4011/{phase}_predictions.csv")
            for variant in VARIANTS for phase in PARTITIONS
        },
    }
    publish(OUTPUT / "comparison.json", json.dumps(result, indent=2, allow_nan=False) + "\n")

    report = ["# Week 3 · Robustness and error analysis", "",
              "**23 November 2026 · U.S. congressional speeches.** This report compares matched text-only and text-plus-chamber predictions on the fixed development and test partitions. Development choices are separated from the held-out test check.", "",
              "## Matched model comparison", "",
              "Each cell gives macro-F1 [95% speaker-bootstrap interval] / accuracy.", "",
              "| Partition | Text only | Text + chamber |", "|---|---|---|"]
    for phase in PARTITIONS:
        values = {}
        for row in comparison:
            if row["partition"] == phase:
                values[row["model"]] = row
        report.append("| {} | {:.3f} [{:.3f}, {:.3f}] / {:.3f} | {:.3f} [{:.3f}, {:.3f}] / {:.3f} |".format(
            phase.capitalize(),
            values["text"]["macro_f1"], values["text"]["macro_f1_ci_low"], values["text"]["macro_f1_ci_high"], values["text"]["accuracy"],
            values["chamber"]["macro_f1"], values["chamber"]["macro_f1_ci_low"], values["chamber"]["macro_f1_ci_high"], values["chamber"]["accuracy"]))
    report += ["", "## Matched error groups", "",
               "| Partition | Both correct | Chamber only correct | Text only correct | Both wrong |",
               "|---|---:|---:|---:|---:|"]
    for phase in PARTITIONS:
        counts = Counter(r["error_category"] for r in matched_files[phase])
        report.append("| {} | {} | {} | {} | {} |".format(
            phase.capitalize(), counts["both_correct"], counts["chamber_only_correct"],
            counts["text_only_correct"], counts["both_wrong"]))
    report += ["", "## Representative cases", "",
               f"The file `representative_cases.csv` contains {len(examples)} selected speech IDs with the source text and both model decisions. Selection used development categories first; test rows were included only after the category rules were fixed.",
               "", "## Lexical cues", "",
               "`lexical_cues.csv` ranks TF-IDF logistic-regression features separately for the temporal and speaker-disjoint protocols. These cues describe the baseline model and are not a direct explanation of RoBERTa.",
               "", "## Scope and pending robustness work", "",
               "This release completes matched error analysis and lexical-cue analysis. Party-name masking and phrase-removal tests require the formal RoBERTa checkpoints and training workstation; checkpoint files are not included in this clone.",
               "", "## Reproduction", "",
               "Run `python -B scripts/week3_analysis.py` from the repository root using the Week 1 analysis environment.",
               "", "The machine-readable outputs contain exact matched speech IDs, probabilities, confusion matrices, bootstrap intervals and ranked lexical cues.",
               "[Matched predictions](../results/week3/error_analysis/)"]
    publish(REPORT, "\n".join(report) + "\n")

    handoff = ["# Week 3 handoff · Robustness and error analysis", "",
               "**23 November 2026 · U.S. congressional speeches.** The first Week 3 deliverable compares matched RoBERTa predictions and publishes representative cases.", "",
               "## Start here", "",
               "Run:", "",
               "```powershell", "python -B scripts/week1_archive.py restore", "python -B scripts/week3_analysis.py", "python -B scripts/verify_week3.py", "```", "",
               "## Which file answers which question?", "",
               "| Team task | Files to open | What to use |",
               "|---|---|---|",
               "| Write the robustness section | [Week 3 comparison](week3_comparison.md) | Matched text-versus-chamber results, confidence intervals and error groups. |",
               "| Inspect exact decisions | [Development matches](../results/week3/error_analysis/dev_matched_predictions.csv) · [Test matches](../results/week3/error_analysis/test_matched_predictions.csv) | One row per frozen speech with both probabilities and the gold label. |",
               "| Read representative speeches | [Representative cases](../results/week3/error_analysis/representative_cases.csv) | Development-selected examples and fixed test examples with source text. |",
               "| Inspect lexical cues | [Lexical cues](../results/week3/error_analysis/lexical_cues.csv) | TF-IDF/logistic-regression cues, not RoBERTa attributions. |",
               "", "## Use development choices to protect test validity", "",
               "The category rules are determined from development speeches. Apply those fixed rules to test speeches without changing them in response to test performance.",
               "", "## Remaining Week 3 work", "",
               "Complete party-name masking and phrase-removal inference on the retained formal checkpoints. Add the results to this report, preserve the held-out test as the final check, and prepare the final presentation.",
               ""]
    publish(HANDOFF, "\n".join(handoff))
    print(f"Wrote Week 3 comparison outputs to {OUTPUT}")


if __name__ == "__main__":
    main()
