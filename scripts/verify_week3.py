"""Verify Week 3 matched predictions against the Week 2 prediction files."""

import csv
import hashlib
import json
from pathlib import Path

from week1_archive import DATA, restore
from week1_baselines import ROOT


ERRORS = ROOT / "results/week3/error_analysis"
FORMAL = ROOT / "results/week2/formal"


def rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    corpus = restore(ROOT / "data/week1")
    ids = {r["speech_id"] for r in map(json.loads, corpus.open(encoding="utf-8"))}
    comparison = json.loads((ERRORS / "comparison.json").read_text(encoding="utf-8"))
    for phase in ("dev", "test"):
        matched = list(rows(ERRORS / f"{phase}_matched_predictions.csv"))
        require(matched, f"Missing {phase} matched rows")
        seen = set()
        text_predictions = {r["speech_id"]: r for r in rows(FORMAL / "temporal_text_s4011" / f"{phase}_predictions.csv")}
        chamber_predictions = {r["speech_id"]: r for r in rows(FORMAL / "temporal_chamber_s4011" / f"{phase}_predictions.csv")}
        require(set(text_predictions) == set(chamber_predictions), f"Prediction ID sets differ in {phase}")
        for row in matched:
            sid = row["speech_id"]
            require(sid in ids and sid not in seen, f"Invalid/duplicate speech ID: {sid}")
            seen.add(sid)
            text, chamber = text_predictions[sid], chamber_predictions[sid]
            require(row["gold"] == text["gold"] == chamber["gold"], f"Gold mismatch: {sid}")
            require(row["canonical_speaker_id"] == text["canonical_speaker_id"] == chamber["canonical_speaker_id"], f"Speaker mismatch: {sid}")
            require(row["text_predicted"] == text["predicted"], f"Text prediction mismatch: {sid}")
            require(row["chamber_predicted"] == chamber["predicted"], f"Chamber prediction mismatch: {sid}")
        require(seen == set(text_predictions), f"Missing matched {phase} prediction")
    require(len(comparison["comparison"]) == 4, "Expected four primary comparison rows")
    print("Verified Week 3 matched development/test outputs and exact speech coverage.")


if __name__ == "__main__":
    main()
