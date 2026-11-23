"""Independently check the saved cohort, split manifest and CPU baseline predictions."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from week1_archive import restore


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/week1"
RESULTS = ROOT / "results/week1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    if not (DATA / "corpus.jsonl").is_file():
        restore(DATA)
    run = json.loads((RESULTS / "run.json").read_text(encoding="utf-8"))
    summary = json.loads((DATA / "split_summary.json").read_text(encoding="utf-8"))
    for key, filename in (("corpus", "corpus.jsonl"), ("splits", "splits.csv")):
        require(digest(DATA / filename) == summary[key + "_sha256"] == run["input_sha256"][key],
                f"Inputs changed after freezing: {filename}")
    for filename, expected in run["output_sha256"].items():
        require(digest(RESULTS / filename) == expected, f"Result changed: {filename}")

    with (DATA / "corpus.jsonl").open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    corpus = {row["speech_id"]: row for row in records}
    splits = csv_rows(DATA / "splits.csv")
    assignments = {row["speech_id"]: row for row in splits}
    require(len(corpus) == len(records) and len(assignments) == len(splits), "Duplicate speech ID")
    require(set(assignments) == {sid for sid, row in corpus.items() if row["label"] in ("D", "R")},
            "The design requires exactly one split assignment for every scored speech")
    windows = csv_rows(DATA / "windows.csv")
    require(len(windows) == len(corpus) and {w["speech_id"] for w in windows} == set(corpus),
            "The token windows must cover each corpus speech exactly once")
    for w in windows:
        n = int(w["content_tokens"])
        spans = [(int(w["window_1_start"]), int(w["window_1_stop"]))]
        if w["window_2_start"]:
            spans.append((int(w["window_2_start"]), int(w["window_2_stop"])))
        require(spans[0][0] == 0 and spans[-1][1] == n
                and all(0 < stop - start <= 254 for start, stop in spans)
                and (len(spans) == 1 or spans[0][1] <= spans[1][0])
                and sum(stop - start for start, stop in spans) == min(n, 508),
                f"Invalid token boundaries: {w['speech_id']}")
    people = defaultdict(set)
    heldout = defaultdict(set)
    for sid, assignment in assignments.items():
        row = corpus[sid]
        require(bool(row["canonical_speaker_id"]), f"Missing legislator: {sid}")
        for protocol in ("speaker", "temporal"):
            partition = assignment[protocol + "_split"]
            require(partition in ("train", "dev", "test"), f"Invalid split: {sid}")
            people[protocol, partition].add(row["canonical_speaker_id"])
            if partition != "train":
                heldout[protocol, partition].add(sid)
        period = {110: "train", 111: "train", 112: "train", 113: "dev", 114: "test"}
        require(assignment["temporal_split"] == period[row["congress"]], f"Wrong temporal split: {sid}")
    for pair in (("train", "dev"), ("train", "test"), ("dev", "test")):
        require(people["speaker", pair[0]].isdisjoint(people["speaker", pair[1]]), "Speaker leakage")

    forecasts = defaultdict(dict)
    for pred in csv_rows(RESULTS / "predictions.csv"):
        protocol, partition, model, sid = (pred[k] for k in ("protocol", "partition", "model", "speech_id"))
        group = forecasts[protocol, partition, model]
        require(sid in heldout[protocol, partition] and sid not in group, f"Invalid prediction key: {sid}")
        require(pred["gold"] == corpus[sid]["label"], f"Gold label mismatch: {sid}")
        probs = [float(pred["p_D"]), float(pred["p_R"])]
        require(all(0 <= x <= 1 for x in probs) and abs(sum(probs) - 1) < 1e-9, f"Bad probabilities: {sid}")
        require((pred["predicted"] == "D" and probs[0] >= probs[1] - 1e-12)
                or (pred["predicted"] == "R" and probs[1] >= probs[0] - 1e-12), f"Wrong prediction: {sid}")
        group[sid] = (pred["gold"], pred["predicted"])
    metrics = {(r["protocol"], r["partition"], r["model"]): r for r in csv_rows(RESULTS / "metrics.csv")
               if r["subgroup"] == "all"}
    expected = {(p, split, model) for p, split in heldout for model in ("majority", "tfidf_lr", "year_chamber_lr")}
    require(set(forecasts) == set(metrics) == expected, "Missing model or evaluation partition")
    for key, pairs in forecasts.items():
        require(set(pairs) == heldout[key[:2]], f"Unequal evaluation speech IDs: {key}")
        counts = Counter(pairs.values())
        f1 = sum(2 * counts[c, c] / (2 * counts[c, c] + counts["D", "R"] + counts["R", "D"])
                 for c in ("D", "R")) / 2
        accuracy = (counts["D", "D"] + counts["R", "R"]) / len(pairs)
        require(abs(float(metrics[key]["macro_f1"]) - f1) < 1e-10
                and abs(float(metrics[key]["accuracy"]) - accuracy) < 1e-10, f"Metric mismatch: {key}")
    print(f"Verified {len(corpus):,} corpus rows, {len(splits):,} scored IDs and {len(metrics)} model/partition results.")


if __name__ == "__main__":
    main()
