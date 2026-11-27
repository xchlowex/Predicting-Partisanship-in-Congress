"""Slice development scores and list the most confident mistakes.

Cuts are fixed before looking at accuracy: short and long are the outer
quartiles of word count, a close call has confidence below 0.60, and a sure
guess has confidence of at least 0.90. Party mentions use the frozen mask.
"""

import csv
import json
import re
import statistics
from pathlib import Path

from week1_windows import ROOT
from week3_make_edits import PARTY_NAMES


RUNS = {
    "text": "temporal_text_s4011",
    "chamber": "temporal_chamber_s4011",
}
PHASE = "dev"
CONDITIONS = [
    "original",
    "party_masked",
    "heading_removed",
    "first_sentence_removed",
    "last_sentence_removed",
]


def load_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["speech_id"]: row for row in csv.DictReader(handle)}


def load_jsonl(path):
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["speech_id"]] = row
    return rows


def macro_f1(golds, preds):
    scores = []
    for label in ("D", "R"):
        tp = sum(gold == label and pred == label for gold, pred in zip(golds, preds))
        fp = sum(gold != label and pred == label for gold, pred in zip(golds, preds))
        fn = sum(gold == label and pred != label for gold, pred in zip(golds, preds))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return sum(scores) / 2


def summarize(rows):
    if not rows:
        return {"n": 0, "accuracy": None, "macro_f1": None, "errors": 0}
    golds = [row["gold"] for row in rows]
    preds = [row["predicted"] for row in rows]
    correct = sum(gold == pred for gold, pred in zip(golds, preds))
    return {
        "n": len(rows),
        "accuracy": correct / len(rows),
        "macro_f1": macro_f1(golds, preds),
        "errors": len(rows) - correct,
    }


def main():
    base = ROOT / "results/week3/robustness" / PHASE
    texts = load_jsonl(base / "original.jsonl")
    words = {sid: len(row["text"].split()) for sid, row in texts.items()}
    ordered_lengths = sorted(words.values())
    q1 = statistics.quantiles(ordered_lengths, n=4)[0]
    q3 = statistics.quantiles(ordered_lengths, n=4)[2]
    report = {
        "phase": PHASE,
        "word_count": {
            "min": ordered_lengths[0],
            "q1": q1,
            "median": statistics.median(ordered_lengths),
            "q3": q3,
            "max": ordered_lengths[-1],
        },
        "cuts": {
            "short": f"word count at or below quartile 1 ({q1:.0f})",
            "long": f"word count at or above quartile 3 ({q3:.0f})",
            "close": "confidence < 0.60",
            "sure": "confidence >= 0.90",
            "party_mention": "frozen party-name expression",
        },
        "models": {},
    }
    for label, run in RUNS.items():
        gold = load_csv(ROOT / "results/week2/formal" / run / f"{PHASE}_predictions.csv")
        scores = {
            condition: load_csv(base / "scores" / f"{run}__{condition}.csv")
            for condition in CONDITIONS
        }
        ids = list(gold)
        original = scores["original"]
        rows = []
        for sid in ids:
            confidence = max(float(original[sid]["p_D"]), float(original[sid]["p_R"]))
            text = texts[sid]["text"]
            rows.append({
                "speech_id": sid,
                "gold": gold[sid]["gold"],
                "predicted": original[sid]["predicted"],
                "confidence": confidence,
                "words": words[sid],
                "chamber": texts[sid]["chamber"],
                "party_mention": bool(PARTY_NAMES.search(text)),
                "text": text,
            })
        edits = {}
        for condition in CONDITIONS:
            paired = []
            flips = dto_r = rto_d = right_to_wrong = wrong_to_right = 0
            for row in rows:
                pred = scores[condition][row["speech_id"]]["predicted"]
                paired.append({**row, "predicted": pred})
                if pred != row["predicted"]:
                    flips += 1
                    dto_r += row["predicted"] == "D" and pred == "R"
                    rto_d += row["predicted"] == "R" and pred == "D"
                    right_to_wrong += row["predicted"] == row["gold"] and pred != row["gold"]
                    wrong_to_right += row["predicted"] != row["gold"] and pred == row["gold"]
            summary = summarize(paired)
            summary.update({
                "flips_vs_original": flips,
                "D_to_R": dto_r,
                "R_to_D": rto_d,
                "correct_to_wrong": right_to_wrong,
                "wrong_to_correct": wrong_to_right,
            })
            edits[condition] = summary
        groups = {
            "all": rows,
            "short": [row for row in rows if row["words"] <= q1],
            "long": [row for row in rows if row["words"] >= q3],
            "house": [row for row in rows if row["chamber"] == "H"],
            "senate": [row for row in rows if row["chamber"] == "S"],
            "party_mention": [row for row in rows if row["party_mention"]],
            "no_party_mention": [row for row in rows if not row["party_mention"]],
            "close": [row for row in rows if row["confidence"] < 0.60],
            "sure": [row for row in rows if row["confidence"] >= 0.90],
        }
        mistakes = [row for row in rows if row["predicted"] != row["gold"]]
        mistakes.sort(key=lambda row: row["confidence"], reverse=True)
        sure_errors = [row for row in mistakes if row["confidence"] >= 0.90]
        report["models"][label] = {
            "edits": edits,
            "slices": {name: summarize(group) for name, group in groups.items()},
            "sure_error_party_mentions": sum(row["party_mention"] for row in sure_errors),
            "sure_errors": len(sure_errors),
            "top_errors": [
                {
                    "speech_id": row["speech_id"],
                    "gold": row["gold"],
                    "predicted": row["predicted"],
                    "confidence": row["confidence"],
                    "words": row["words"],
                    "chamber": "House" if row["chamber"] == "H" else "Senate",
                    "party_mention": row["party_mention"],
                    "excerpt": re.sub(r"\s+", " ", row["text"]).strip()[:360],
                }
                for row in mistakes[:8]
            ],
        }
    destination = base / "error_slices.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
