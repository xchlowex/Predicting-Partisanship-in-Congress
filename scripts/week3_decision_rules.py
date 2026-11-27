"""Decision rules for the temporal models, chosen on development only.

Predict Republican when p_R is at least a threshold. The default threshold is
0.5, which is the argmax because the two probabilities sum to 1. A lower
threshold changes labels. The selected threshold maximises development
macro-F1, and the closest value to 0.5 breaks ties. Test is scored after that
choice. Length and chamber rules use the same development cuts as the slices.
"""

import csv
import json
from pathlib import Path

from week1_windows import ROOT


SHORT_MAX = 146
TEXT = "temporal_text_s4011"
CHAMBER = "temporal_chamber_s4011"
GRID = [round(i / 100, 2) for i in range(5, 96)]


def load_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_rows(phase, run):
    gold = {row["speech_id"]: row for row in load_csv(
        ROOT / "results/week2/formal" / run / f"{phase}_predictions.csv")}
    scores = load_csv(ROOT / "results/week3/robustness" / phase / "scores" / f"{run}__original.csv")
    texts = {}
    with (ROOT / "results/week3/robustness" / phase / "original.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            texts[row["speech_id"]] = row
    rows = []
    for score in scores:
        source = texts[score["speech_id"]]
        label = gold[score["speech_id"]]
        rows.append({
            "gold": label["gold"],
            "p_R": float(score["p_R"]),
            "words": len(source["text"].split()),
            "chamber": source["chamber"],
        })
    return rows


def metrics(golds, preds):
    scores = []
    detail = {}
    for label in ("D", "R"):
        tp = sum(gold == label and pred == label for gold, pred in zip(golds, preds))
        fp = sum(gold != label and pred == label for gold, pred in zip(golds, preds))
        fn = sum(gold == label and pred != label for gold, pred in zip(golds, preds))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        scores.append(f1)
        detail[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
    correct = sum(gold == pred for gold, pred in zip(golds, preds))
    return {
        "n": len(golds),
        "accuracy": correct / len(golds),
        "macro_f1": sum(scores) / 2,
        "per_class": detail,
    }


def apply_threshold(rows, threshold, key=None):
    preds = []
    for row in rows:
        cutoff = threshold if key is None else threshold[key(row)]
        preds.append("R" if row["p_R"] >= cutoff else "D")
    return metrics([row["gold"] for row in rows], preds)


def best_threshold(rows, key=None, group=None):
    chosen = rows if group is None else [row for row in rows if key(row) == group]
    best = None
    for threshold in GRID:
        stats = apply_threshold(chosen, threshold)
        candidate = (stats["macro_f1"], -abs(threshold - 0.5), threshold, stats)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    return {"threshold": best[2], "dev": best[3]}


def pack(stats):
    return {
        "macro_f1": round(stats["macro_f1"], 4),
        "accuracy": round(stats["accuracy"], 4),
        "D_recall": round(stats["per_class"]["D"]["recall"], 4),
        "R_recall": round(stats["per_class"]["R"]["recall"], 4),
        "D_precision": round(stats["per_class"]["D"]["precision"], 4),
        "R_precision": round(stats["per_class"]["R"]["precision"], 4),
    }


def error_mass(rows):
    missed = [row["p_R"] for row in rows if row["gold"] == "R" and row["p_R"] < 0.5]
    missed.sort()
    def share(cutoff):
        return sum(value >= cutoff for value in missed) / len(missed)
    return {
        "missed_republicans": len(missed),
        "median_p_R": missed[len(missed) // 2],
        "share_p_R_at_least_0.40": round(share(0.40), 4),
        "share_p_R_at_least_0.30": round(share(0.30), 4),
        "share_p_R_at_least_0.20": round(share(0.20), 4),
        "share_p_R_below_0.10": round(sum(value < 0.10 for value in missed) / len(missed), 4),
    }


def main():
    report = {"grid": "0.05 to 0.95 by 0.01", "selection": "max development macro-F1, then closest to 0.5", "models": {}}
    for name, run in (("text", TEXT), ("chamber", CHAMBER)):
        dev, test = load_rows("dev", run), load_rows("test", run)
        single = best_threshold(dev)
        length_key = lambda row: "short" if row["words"] <= SHORT_MAX else "other"
        chamber_key = lambda row: row["chamber"]
        length = {group: best_threshold(dev, length_key, group) for group in ("short", "other")}
        chamber = {group: best_threshold(dev, chamber_key, group) for group in ("H", "S")}
        rules = {
            "argmax_0.5": {"threshold": 0.5},
            "one_threshold": {"threshold": single["threshold"]},
            "by_length": {"short": length["short"]["threshold"], "other": length["other"]["threshold"]},
            "by_chamber": {"H": chamber["H"]["threshold"], "S": chamber["S"]["threshold"]},
        }
        scored = {}
        for rule, spec in rules.items():
            if rule == "by_length":
                dev_stats = apply_threshold(dev, spec, length_key)
                test_stats = apply_threshold(test, spec, length_key)
            elif rule == "by_chamber":
                dev_stats = apply_threshold(dev, spec, chamber_key)
                test_stats = apply_threshold(test, spec, chamber_key)
            else:
                dev_stats = apply_threshold(dev, spec["threshold"])
                test_stats = apply_threshold(test, spec["threshold"])
            scored[rule] = {"spec": spec, "dev": pack(dev_stats), "test": pack(test_stats)}
        report["models"][name] = {
            "missed_on_dev": error_mass(dev),
            "missed_on_test": error_mass(test),
            "rules": scored,
        }
    destination = ROOT / "results/week3/robustness/decision_rules.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(destination)
    print(json.dumps(report["models"], indent=2))


if __name__ == "__main__":
    main()
