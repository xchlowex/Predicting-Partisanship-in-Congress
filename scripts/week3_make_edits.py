"""Write edited JSONL copies of one formal run's development speeches.

An edit that would leave the text empty keeps the original wording and is
counted as unchanged. Every output file has the same speech ids in the same
order as the run's dev_predictions.csv.
"""

import argparse
import csv
import gzip
import json
import re
from pathlib import Path

from week1_windows import ROOT


PARTY_NAMES = re.compile(
    r"\b(?:Democrat(?:ic)?|Republican(?:s)?|GOP|Independent Democrat|Independent)\b",
    re.IGNORECASE,
)
HEADING = re.compile(
    r"^(?:Mr\.|Ms\.|Mrs\.|Madam|Mister)\s+(?:Speaker|Chairman|Chairwoman)\s*\.\s*",
    re.IGNORECASE,
)
SENTENCE_END = re.compile(r"[.!?](?:\s|$)")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def mask_party_names(text):
    return PARTY_NAMES.sub("[PARTY]", text)


def delete_party_names(text):
    """Drop the same frozen party-name expression, then collapse leftover spaces."""
    deleted = re.sub(r"[ \t]+", " ", PARTY_NAMES.sub(" ", text)).strip()
    return deleted


def remove_heading(text):
    return HEADING.sub("", text, count=1).strip()


def remove_first_sentence(text):
    match = SENTENCE_END.search(text)
    if match is None:
        return ""
    return text[match.end():].strip()


def remove_last_sentence(text):
    parts = SENTENCE_SPLIT.split(text.strip())
    if len(parts) < 2:
        return ""
    return " ".join(parts[:-1]).strip()


EDITS = {
    "original": lambda text: text,
    "party_masked": mask_party_names,
    "heading_removed": remove_heading,
    "first_sentence_removed": remove_first_sentence,
    "last_sentence_removed": remove_last_sentence,
}


def load_corpus(speech_ids):
    found = {}
    folder = ROOT / "data/week1/by_congress"
    for path in sorted(folder.glob("congress-*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["speech_id"] in speech_ids:
                    found[row["speech_id"]] = row
        if len(found) == len(speech_ids):
            break
    missing = speech_ids - found.keys()
    if missing:
        raise ValueError(f"{len(missing)} speeches are missing from the corpus.")
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="temporal_text_s4011")
    parser.add_argument("--phase", choices=("dev", "test"), default="dev")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--party-deleted",
        action="store_true",
        help="Write only party_deleted.jsonl. The match is deleted, not replaced with [PARTY].",
    )
    args = parser.parse_args()
    predictions = ROOT / "results/week2/formal" / args.run / f"{args.phase}_predictions.csv"
    output = args.output or (ROOT / "results/week3/robustness" / args.phase)
    output.mkdir(parents=True, exist_ok=True)
    with predictions.open(encoding="utf-8", newline="") as handle:
        order = [row["speech_id"] for row in csv.DictReader(handle)]
    corpus = load_corpus(set(order))
    edits = {"party_deleted": delete_party_names} if args.party_deleted else EDITS
    counts = {name: {"changed": 0, "kept_because_empty": 0} for name in edits}
    handles = {
        name: (output / f"{name}.jsonl").open("w", encoding="utf-8")
        for name in edits
    }
    try:
        for speech_id in order:
            source = corpus[speech_id]
            text = source["text"]
            year = source["year"]
            chamber = source["chamber"]
            if type(year) is not int or chamber not in {"H", "S"} or not text.strip():
                raise ValueError(f"Unusable source row: {speech_id}")
            for name, edit in edits.items():
                edited = text if name == "original" else edit(text)
                if not edited.strip():
                    edited = text
                    counts[name]["kept_because_empty"] += 1
                elif edited != text:
                    counts[name]["changed"] += 1
                handles[name].write(json.dumps({
                    "speech_id": speech_id,
                    "text": edited,
                    "year": year,
                    "chamber": chamber,
                }, ensure_ascii=False) + "\n")
    finally:
        for handle in handles.values():
            handle.close()
    report = {"run": args.run, "speeches": len(order), "edits": counts}
    counts_path = output / ("party_deleted_counts.json" if args.party_deleted else "edit_counts.json")
    counts_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
