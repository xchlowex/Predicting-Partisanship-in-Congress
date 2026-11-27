"""Tables for the Week 3 follow-up.

Development length cuts stay fixed: short is 146 words or fewer, long is 543
or more. Party deletion is reported only after its score file exists. Seen and
unseen legislators use the frozen temporal speaker status.
"""

import csv
import json
from pathlib import Path

from week1_windows import ROOT
from week3_make_edits import PARTY_NAMES


SHORT_MAX = 146
LONG_MIN = 543
TEXT = "temporal_text_s4011"
CHAMBER = "temporal_chamber_s4011"


def load_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_jsonl(path):
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["speech_id"]] = row
    return rows


def class_counts(rows):
    labels = ("D", "R")
    out = {}
    for label in labels:
        tp = sum(row["gold"] == label and row["predicted"] == label for row in rows)
        fp = sum(row["gold"] != label and row["predicted"] == label for row in rows)
        fn = sum(row["gold"] == label and row["predicted"] != label for row in rows)
        support = tp + fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        out[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_negatives": fn,
        }
    correct = sum(row["gold"] == row["predicted"] for row in rows)
    recalls = [out[label]["f1"] for label in labels]
    return {
        "n": len(rows),
        "accuracy": correct / len(rows) if rows else None,
        "macro_f1": sum(recalls) / 2 if rows else None,
        "per_class": out,
    }


def joined(gold_rows, scores):
    by_id = {row["speech_id"]: row for row in scores}
    joined_rows = []
    for row in gold_rows:
        pred = by_id[row["speech_id"]]
        joined_rows.append({
            "speech_id": row["speech_id"],
            "speaker": row["canonical_speaker_id"],
            "gold": row["gold"],
            "predicted": pred["predicted"],
        })
    return joined_rows


def score_path(phase, run, condition):
    return ROOT / "results/week3/robustness" / phase / "scores" / f"{run}__{condition}.csv"


def recall_table():
    table = []
    for phase in ("dev", "test"):
        for model, run in (("text", TEXT), ("chamber", CHAMBER)):
            gold = load_csv(ROOT / "results/week2/formal" / run / f"{phase}_predictions.csv")
            stats = class_counts(joined(gold, load_csv(score_path(phase, run, "original"))))
            table.append({"phase": phase, "model": model, **stats})
    return table


def slice_table():
    rows = []
    for phase in ("dev", "test"):
        texts = load_jsonl(ROOT / "results/week3/robustness" / phase / "original.jsonl")
        status = {}
        with (ROOT / "data/week1/splits.csv").open(encoding="utf-8", newline="") as handle:
            for split in csv.DictReader(handle):
                status[split["speech_id"]] = split["temporal_speaker_status"]
        for model, run in (("text", TEXT), ("chamber", CHAMBER)):
            gold = load_csv(ROOT / "results/week2/formal" / run / f"{phase}_predictions.csv")
            scored = joined(gold, load_csv(score_path(phase, run, "original")))
            grouped = {"all": [], "short": [], "long": [], "house": [], "senate": [],
                       "party_mention": [], "no_party_mention": []}
            for row in scored:
                source = texts[row["speech_id"]]
                words = len(source["text"].split())
                named = bool(PARTY_NAMES.search(source["text"]))
                grouped["all"].append(row)
                if words <= SHORT_MAX:
                    grouped["short"].append(row)
                if words >= LONG_MIN:
                    grouped["long"].append(row)
                grouped["house" if source["chamber"] == "H" else "senate"].append(row)
                grouped["party_mention" if named else "no_party_mention"].append(row)
            if phase == "test":
                seen = [row for row in scored if status[row["speech_id"]] == "seen_train"]
                unseen = [row for row in scored if status[row["speech_id"]] == "unseen_train"]
                grouped["seen_train"] = seen
                grouped["unseen_train"] = unseen
            for name, group in grouped.items():
                stats = class_counts(group)
                speakers = len({row["speaker"] for row in group})
                rows.append({
                    "phase": phase,
                    "model": model,
                    "slice": name,
                    "speakers": speakers,
                    **stats,
                })
    return rows


def deletion_table():
    found = []
    for phase in ("dev", "test"):
        for run in (TEXT, "temporal_text_s4013"):
            gold = load_csv(ROOT / "results/week2/formal" / run / f"{phase}_predictions.csv")
            original_file = score_path(phase, run, "original")
            deleted_file = score_path(phase, run, "party_deleted")
            masked_file = score_path(phase, run, "party_masked")
            if not original_file.is_file() or not deleted_file.is_file():
                continue
            original = {row["speech_id"]: row for row in load_csv(original_file)}
            deleted = {row["speech_id"]: row for row in load_csv(deleted_file)}
            base = joined(gold, load_csv(original_file))
            deleted_rows = []
            flips = d_to_r = r_to_d = right_to_wrong = wrong_to_right = 0
            for row in base:
                pred = deleted[row["speech_id"]]["predicted"]
                deleted_rows.append({**row, "predicted": pred})
                if pred != original[row["speech_id"]]["predicted"]:
                    flips += 1
                    d_to_r += original[row["speech_id"]]["predicted"] == "D" and pred == "R"
                    r_to_d += original[row["speech_id"]]["predicted"] == "R" and pred == "D"
                    right_to_wrong += row["predicted"] == row["gold"] and pred != row["gold"]
                    wrong_to_right += row["predicted"] != row["gold"] and pred == row["gold"]
            item = {
                "phase": phase,
                "run": run,
                "original": class_counts(base),
                "party_deleted": class_counts(deleted_rows),
                "flips_vs_original": flips,
                "D_to_R": d_to_r,
                "R_to_D": r_to_d,
                "correct_to_wrong": right_to_wrong,
                "wrong_to_correct": wrong_to_right,
            }
            if masked_file.is_file():
                masked = {row["speech_id"]: row for row in load_csv(masked_file)}
                masked_rows = [{**row, "predicted": masked[row["speech_id"]]["predicted"]} for row in base]
                item["party_masked"] = class_counts(masked_rows)
            found.append(item)
    return found


def main():
    report = {
        "cuts": {"short_max_words": SHORT_MAX, "long_min_words": LONG_MIN},
        "recall": recall_table(),
        "slices": slice_table(),
        "party_deleted": deletion_table(),
    }
    destination = ROOT / "results/week3/robustness/followup.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
