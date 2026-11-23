"""Reproduce the two source-overlap checks reported for Week 1."""

import csv
import hashlib
import io
import json
from collections import Counter
from itertools import islice
from pathlib import Path
from zipfile import ZipFile

from stanford_zip import open_stanford_zip


REPO = Path(__file__).resolve().parents[1]
RAW = REPO.parent / "stanford_raw"


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_preview():
    """Compare the complete public 100-row preview with the matching Stanford rows."""
    with (REPO / "reports/week1_hf_preview_fingerprints.csv").open(encoding="utf-8") as file:
        preview = list(csv.DictReader(file))
    results = Counter()
    with open_stanford_zip(RAW / "hein-bound.zip") as archive:
        with archive.open("hein-bound/speeches_043.txt") as file:
            next(file)
            texts = dict(line.decode("cp1252").strip().split("|", 1) for line in islice(file, len(preview)))
        with io.TextIOWrapper(archive.open("hein-bound/descr_043.txt"), encoding="cp1252") as file:
            metadata = {r["speech_id"]: r for r in islice(csv.DictReader(file, delimiter="|"), len(preview))}
    for row in preview:
        key = row["speech_id"]
        text = " ".join(texts[key].split())
        if row["prefix_only"] == "1":
            text = text[:int(row["displayed_text_chars"])]
        r = metadata[key]
        fields = [
            r["chamber"], r["date"], int(r["number_within_file"]), r["speaker"],
            r["first_name"], r["last_name"], r["state"], r["gender"],
            int(r["line_start"]), int(r["line_end"]), r["file"], int(r["char_count"]), int(r["word_count"]),
        ]
        if sha256(text) != row["displayed_text_sha256"]:
            raise ValueError(f"Published text differs for {key}")
        if sha256(json.dumps(fields, ensure_ascii=False, separators=(",", ":"))) != row["metadata_sha256"]:
            raise ValueError(f"Published metadata differs for {key}")
        results["prefix" if row["prefix_only"] == "1" else "full_text"] += 1
    return dict(results)


def floor_fingerprints(archive: ZipFile, edition, congress):
    """Count same-day, same-chamber full-text hashes for speeches of 30+ words."""
    session = f"{congress:03}"
    with io.TextIOWrapper(archive.open(f"{edition}/descr_{session}.txt"), encoding="latin-1") as file:
        by_id = {
            r["speech_id"].encode("ascii"): (r["date"], r["chamber"])
            for r in csv.DictReader(file, delimiter="|")
            if r["chamber"] in {"H", "S"}
        }
    counts = Counter()
    with archive.open(f"{edition}/speeches_{session}.txt") as file:
        next(file)
        for line in file:
            key, text = line.split(b"|", 1)
            metadata = by_id.get(key)
            if metadata is None:
                continue
            words = text.split()
            if len(words) >= 30:
                digest = hashlib.sha256(b" ".join(words).lower()).digest()
                counts[(*metadata, digest)] += 1
    return counts


def main():
    print("Hugging Face public preview vs Stanford bound:", check_preview(), flush=True)
    with open_stanford_zip(RAW / "hein-bound.zip") as bound, open_stanford_zip(RAW / "hein-daily.zip") as daily:
        for congress in (110, 111):
            left = floor_fingerprints(bound, "hein-bound", congress)
            right = floor_fingerprints(daily, "hein-daily", congress)
            shared = left.keys() & right.keys()
            print(json.dumps(dict(congress=congress, bound_rows=left.total(), daily_rows=right.total(),
                                  shared_fingerprints=len(shared), daily_rows_matching_bound=sum(right[k] for k in shared))),
                  flush=True)


if __name__ == "__main__":
    main()
