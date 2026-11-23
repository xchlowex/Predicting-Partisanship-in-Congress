"""Compare completed Week 2 runs with the Week 1 baselines on the same speeches."""

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from week1_baselines import BOOTSTRAPS, LABELS, MODELS, PROTOCOLS, ROOT, SEED
from week1_baselines import checksum, evaluate, scores


VARIANTS = ("text", "year", "chamber", "both")
SEEDS = (4011, 4012, 4013)
PARTITIONS = ("dev", "test")
DISPLAY = {
    "majority": "Training majority", "tfidf_lr": "TF-IDF + logistic regression",
    "year_chamber_lr": "Year + chamber logistic regression", "text": "RoBERTa, text",
    "year": "RoBERTa, text + year", "chamber": "RoBERTa, text + chamber",
    "both": "RoBERTa, text + year + chamber",
}
FORMAL = ROOT / "results/week2/formal"
SUMMARY = ROOT / "results/week2/formal_summary"
REPORT = ROOT / "reports/week2_comparison.md"


def records(path):
    with path.open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def json_file(path):
    return json.loads(path.read_text(encoding="utf-8"))


def checked_predictions(rows, cohort):
    """Return class indices in the frozen cohort order after checking each speech."""
    by_id = {}
    for r in rows:
        sid = r["speech_id"]
        if sid in by_id:
            raise ValueError(f"Repeated prediction ID: {sid}")
        pd, pr = float(r["p_D"]), float(r["p_R"])
        if not (0 <= pd <= 1 and 0 <= pr <= 1 and math.isclose(pd + pr, 1.0, abs_tol=2e-6)):
            raise ValueError(f"Invalid probabilities for {sid}")
        if r["predicted"] != ("D" if pd >= pr else "R"):
            raise ValueError(f"Decision disagrees with class probabilities for {sid}")
        by_id[sid] = r
    if set(by_id) != {r["speech_id"] for r in cohort}:
        raise ValueError("Prediction IDs differ from the complete frozen evaluation partition")
    predictions = []
    for speech in cohort:
        pred = by_id[speech["speech_id"]]
        if pred["gold"] != speech["label"] or pred["canonical_speaker_id"] != speech["canonical_speaker_id"]:
            raise ValueError(f"Prediction gold label or speaker disagrees for {speech['speech_id']}")
        predictions.append(LABELS.index(pred["predicted"]))
    return np.asarray(predictions, dtype=int)


def frozen_cohorts(needed):
    split_path = ROOT / "data/week1/splits.csv"
    corpus_path = ROOT / "data/week1/corpus.jsonl"
    run = json_file(ROOT / "results/week1/run.json")
    for key, path in (("corpus", corpus_path), ("splits", split_path)):
        if checksum(path) != run["input_sha256"][key]:
            raise ValueError(f"Week 1 {key} SHA-256 mismatch")
    split_rows = {r["speech_id"]: r for r in records(split_path)}
    cohorts = {key: [] for key in needed}
    with corpus_path.open(encoding="utf-8") as stream:
        for line in stream:
            speech = json.loads(line)
            split = split_rows.get(speech["speech_id"])
            if split is None:
                continue
            row = {k: speech[k] for k in ("speech_id", "canonical_speaker_id", "label")}
            row["status"] = split["temporal_speaker_status"]
            for protocol in PROTOCOLS:
                key = protocol, split[protocol + "_split"]
                if key in cohorts:
                    cohorts[key].append(row)
    for key, rows in cohorts.items():
        rows.sort(key=lambda r: r["speech_id"])
        expected = {sid for sid, r in split_rows.items() if r[key[0] + "_split"] == key[1]}
        if len(rows) != len(expected) or {r["speech_id"] for r in rows} != expected:
            raise ValueError(f"Corpus does not match split manifest for {key}")
    return cohorts


def collect_runs(formal):
    entries = {}
    for protocol in PROTOCOLS:
        for variant in VARIANTS:
            for seed in SEEDS:
                folder = formal / f"{protocol}_{variant}_s{seed}"
                if not (folder / "run.json").is_file():
                    continue
                run = json_file(folder / "run.json")
                if (run["protocol"], run["variant"], run["seed"]) != (protocol, variant, seed):
                    raise ValueError(f"Run identity mismatch: {folder}")
                for partition in PARTITIONS:
                    name = f"{partition}_predictions.csv"
                    path = folder / name
                    if not path.is_file() and name not in run["output_sha256"]:
                        continue
                    if checksum(path) != run["output_sha256"][name]:
                        raise ValueError(f"Prediction SHA-256 mismatch: {path}")
                    entries[protocol, partition, variant, seed] = (run, path)
    return entries


def baseline_predictions(cohorts):
    path = ROOT / "results/week1/predictions.csv"
    expected_hash = json_file(ROOT / "results/week1/run.json")["output_sha256"]["predictions.csv"]
    if checksum(path) != expected_hash:
        raise ValueError("Week 1 baseline prediction SHA-256 mismatch")
    grouped = defaultdict(list)
    for r in records(path):
        protocol, partition = r["protocol"], r["partition"]
        if (protocol, partition) in cohorts:
            grouped[protocol, partition, r["model"]].append(r)
    return {(protocol, partition, model, None): checked_predictions(grouped[protocol, partition, model], rows)
            for (protocol, partition), rows in cohorts.items() for model in MODELS}


def bootstrap_series(rows, predicted):
    """Reuse the Week 1 draw order so differences compare the same sampled speakers."""
    _, speaker = np.unique([r["canonical_speaker_id"] for r in rows], return_inverse=True)
    n_speakers = int(speaker.max()) + 1
    cm = np.zeros((n_speakers, len(LABELS), len(LABELS)), dtype=int)
    gold = np.array([LABELS.index(r["label"]) for r in rows])
    np.add.at(cm, (speaker, gold, predicted), 1)
    draws = np.random.default_rng(SEED).integers(n_speakers, size=(BOOTSTRAPS, n_speakers))
    return scores(cm[draws].sum(axis=1))[0]


def evaluate_models(entries):
    cohorts = frozen_cohorts({(p, phase) for p, phase, _, _ in entries})
    predictions = baseline_predictions(cohorts)
    for key, (_, path) in entries.items():
        predictions[key] = checked_predictions(records(path), cohorts[key[:2]])
    metric_rows, differences = [], []
    for (protocol, phase), rows in sorted(cohorts.items()):
        subsets = {"all": np.arange(len(rows))}
        if (protocol, phase) == ("temporal", "test"):
            status = np.array([r["status"] for r in rows])
            if set(status) != {"seen_train", "unseen_train"}:
                raise ValueError("Unexpected temporal test speaker status")
            subsets.update({k: np.flatnonzero(status == k) for k in ("seen_train", "unseen_train")})
        for subgroup, index in subsets.items():
            observed = [rows[i] for i in index]
            blocks = {}
            for key, all_pred in predictions.items():
                if key[:2] != (protocol, phase):
                    continue
                model, seed = key[2:]
                values = all_pred[index]
                numbers, detail = evaluate(observed, values)
                if key in entries and subgroup == "all":
                    saved = entries[key][0][phase + "_metrics"]
                    if any(not math.isclose(numbers[k], saved[k], abs_tol=1e-6)
                           for k in ("macro_f1", "accuracy")):
                        raise ValueError(f"Recomputed metrics disagree with run.json for {key}")
                    if detail["confusion_gold_by_pred"] != saved["confusion_gold_by_pred"]:
                        raise ValueError(f"Recomputed confusion matrix disagrees with run.json for {key}")
                metric_rows.append({"protocol": protocol, "partition": phase, "subgroup": subgroup,
                                    "model": model, "seed": seed, **numbers, **detail})
                if model in (*VARIANTS, "tfidf_lr"):
                    blocks[model, seed] = (numbers["macro_f1"], bootstrap_series(observed, values))
            for (model, seed), (value, replicates) in blocks.items():
                if model not in VARIANTS:
                    continue
                comparator = ("tfidf_lr", None) if model == "text" else ("text", seed)
                if comparator not in blocks:
                    continue
                ref, ref_draws = blocks[comparator]
                lo, hi = np.quantile(replicates - ref_draws, [0.025, 0.975])
                differences.append({"protocol": protocol, "partition": phase, "subgroup": subgroup,
                                    "model": model, "reference": comparator[0], "seed": seed,
                                    "speeches": len(observed), "delta_macro_f1": value - ref,
                                    "ci_low": float(lo), "ci_high": float(hi)})
    return metric_rows, differences


def seed_summary(metrics):
    groups = defaultdict(list)
    for r in metrics:
        if r["model"] in VARIANTS:
            key = r["protocol"], r["partition"], r["subgroup"], r["model"]
            groups[key].append(r)
    result = []
    for (protocol, phase, subgroup, model), rows in sorted(groups.items()):
        row = {"protocol": protocol, "partition": phase, "subgroup": subgroup, "model": model,
               "seeds": sorted(r["seed"] for r in rows)}
        for k in ("macro_f1", "accuracy"):
            values = [r[k] for r in rows]
            row[k + "_mean"] = float(np.mean(values))
            row[k + "_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        result.append(row)
    return result


def fmt(r):
    if r is None:
        return "Pending"
    return (f"{r['macro_f1']:.3f} [{r['macro_f1_ci_low']:.3f}, {r['macro_f1_ci_high']:.3f}]"
            f" / {r['accuracy']:.3f}")


def report_text(entries, metrics, paired, seeds):
    table = {(r["protocol"], r["partition"], r["subgroup"], r["model"], r["seed"]): r for r in metrics}
    pairs = {(r["protocol"], r["partition"], r["subgroup"], r["model"], r["seed"]): r for r in paired}
    completed = {key for key in entries if key[1] == "test"}
    principal = sum(key[3] == SEEDS[0] for key in completed)
    repeats = sum(key[3] != SEEDS[0] and key[2] in ("text", "both") for key in completed)
    parts = ["# Week 2 · Party affiliation from congressional speeches", "",
             f"Completed test evaluations cover **{principal}/8 primary settings** (seed 4011) and "
             f"**{repeats}/8 additional runs** for the text and text + year + chamber comparison. "
             "Both protocols score the audited Democratic and Republican labels.", ""]
    completed_keys = {(p, v, s) for p, _, v, s in entries}
    epoch_counts = sorted({len(run["epochs"]) for run, _ in entries.values()})
    budgets = " or ".join(map(str, epoch_counts))
    parts.extend([f"Each run trained for {budgets} epochs. Development macro-F1 selected the "
                  "checkpoint, with the earliest epoch breaking ties. Test scores use the complete "
                  "frozen partition for each protocol. For the metadata variants, the model adds a "
                  "learned year or chamber term to the text logits before averaging window probabilities. "
                  "Year is standardized using the training speeches and chamber is coded as House 0 or Senate 1.",
                  ""])
    if principal == 8 and (SUMMARY / "model_comparison.png").is_file():
        parts.extend(["![Test macro-F1 by model and evaluation protocol]"
                      "(../results/week2/formal_summary/model_comparison.png)", ""])
    for protocol, title in (("speaker", "Unseen legislators"), ("temporal", "Later Congress")):
        if not any(p == protocol for p, _, _ in completed_keys):
            continue
        parts.extend([f"## {title}", "",
                      "Each cell gives macro-F1 [95% speaker-bootstrap interval] / accuracy.", "",
                      "| Model | Selected epoch | Development | Test |", "|---|---:|---:|---:|"])
        for model in (*MODELS, *VARIANTS):
            seed = None if model in MODELS else SEEDS[0]
            dev = table.get((protocol, "dev", "all", model, seed))
            test = table.get((protocol, "test", "all", model, seed))
            entry = entries.get((protocol, "dev", model, seed))
            epoch = str(entry[0]["best_epoch"]) if entry else "—"
            parts.append(f"| {DISPLAY[model]} | {epoch} | {fmt(dev)} | {fmt(test)} |")
        parts.append("")
        for model in ("text", "both"):
            result = pairs.get((protocol, "test", "all", model, SEEDS[0]))
            if result:
                parts.extend([f"{DISPLAY[model]} minus {DISPLAY[result['reference']]} gives "
                              f"**{result['delta_macro_f1']:+.3f}** test macro-F1 "
                              f"(paired 95% interval [{result['ci_low']:+.3f}, {result['ci_high']:+.3f}]).", ""])
    if ("temporal", "test", "all", "tfidf_lr", None) in table:
        parts.extend(["## Legislator history in the chronological test", "",
                      "| Model | Seen in training | Unseen in training |", "|---|---:|---:|"])
        for model in (*MODELS, *VARIANTS):
            seed = None if model in MODELS else SEEDS[0]
            seen = table.get(("temporal", "test", "seen_train", model, seed))
            unseen = table.get(("temporal", "test", "unseen_train", model, seed))
            parts.append(f"| {DISPLAY[model]} | {fmt(seen)} | {fmt(unseen)} |")
        seen = table["temporal", "test", "seen_train", "tfidf_lr", None]
        unseen = table["temporal", "test", "unseen_train", "tfidf_lr", None]
        parts.extend(["", f"The 114th Congress test contains {seen['speeches']:,} speeches from "
                      f"{seen['speakers']} training-seen legislators and {unseen['speeches']:,} from "
                      f"{unseen['speakers']} training-unseen legislators. The full protocols also use "
                      "different training populations, which limits interpretation of their score gap.", ""])
    parts.extend(["## Training-seed variation", "",
                  "Entries below give test macro-F1 across available seeds for the predefined key comparison. "
                  "SD is the sample standard deviation across independently fitted models.", "",
                  "| Protocol | Model | Seeds completed | Mean macro-F1 | SD |",
                  "|---|---|---|---:|---:|"])
    for r in seeds:
        if r["partition"] == "test" and r["subgroup"] == "all" and r["model"] in ("text", "both"):
            sd = r["macro_f1_sd"]
            sd_text = f"{sd:.3f}" if sd is not None else "—"
            parts.append(f"| {r['protocol']} | {DISPLAY[r['model']]} | "
                         f"{', '.join(map(str, r['seeds']))} | {r['macro_f1_mean']:.3f} | {sd_text} |")
    parts.append("")
    for protocol in PROTOCOLS:
        deltas = [r["delta_macro_f1"] for r in paired if
                  (r["protocol"], r["partition"], r["subgroup"], r["model"]) ==
                  (protocol, "test", "all", "both")]
        if len(deltas) >= 2:
            improved = sum(delta > 0 for delta in deltas)
            parts.extend([f"In the {protocol} protocol, adding year and chamber improved macro-F1 in "
                          f"{improved} of {len(deltas)} matched seeds. The mean change was "
                          f"{np.mean(deltas):+.3f} (range {min(deltas):+.3f} to {max(deltas):+.3f}).", ""])
    parts.extend(["", "## Reproduction", "",
                  f"Each model comparison uses the same frozen speech IDs, labels and canonical speaker IDs. "
                  f"Confidence intervals resample entire legislators {BOOTSTRAPS:,} times (seed {SEED}) "
                  "and condition on the fitted models. Paired differences use the same speaker draws for both "
                  "models. TF-IDF uses the full cleaned speech. RoBERTa averages probabilities across the "
                  "frozen token windows, which can omit the middle of long speeches.", "",
                  "Run `python -B scripts/week2_compare.py` from the repository root using the Week 1 "
                  "scikit-learn environment. The machine-readable results contain each run’s preferred epoch, "
                  "both classes’ precision, recall and F1, confusion matrices, subgroup results, paired "
                  "differences, and cross-seed summaries. "
                  "[Detailed results](../results/week2/formal_summary/comparison.json) · "
                  "[Per-run predictions and settings](../results/week2/formal/)", ""])
    return "\n".join(parts)


def publish(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as workdir:
        temp = Path(workdir) / path.name
        temp.write_text(contents, encoding="utf-8", newline="\n")
        os.replace(temp, path)


def generate(formal=FORMAL, summary=SUMMARY, report=REPORT):
    entries = collect_runs(formal)
    if not entries:
        print("No complete formal evaluation available yet.")
        return
    metrics, paired = evaluate_models(entries)
    seeds = seed_summary(metrics)
    selection = {}
    for (protocol, phase, variant, seed), (run, path) in entries.items():
        key = f"{protocol}_{variant}_s{seed}"
        if key not in selection:
            selection[key] = {"best_epoch": run["best_epoch"], "epochs": run["epochs"],
                              "prediction_sha256": {}}
        selection[key]["prediction_sha256"][phase] = run["output_sha256"][path.name]
    result = {"class_order": LABELS, "bootstrap": {"unit": "canonical_speaker_id",
              "replicates": BOOTSTRAPS, "seed": SEED, "interval": "percentile 95%"},
              "runs": selection, "metrics": metrics, "paired_differences": paired, "seed_summary": seeds}
    publish(summary / "comparison.json", json.dumps(result, indent=2, allow_nan=False) + "\n")
    publish(report, report_text(entries, metrics, paired, seeds))
    print(f"Reported {len(selection)} fitted runs, {len(metrics)} metric rows, "
          f"and {len(paired)} paired comparisons. {report}")


if __name__ == "__main__":
    generate()
