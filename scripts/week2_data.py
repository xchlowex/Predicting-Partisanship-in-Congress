"""Load speech-level model inputs from the frozen Week 1 corpus and split manifest."""

import csv
import hashlib
import json
from dataclasses import dataclass

from week1_archive import DATA, restore
from week1_windows import load_tokenizer, window_ranges


@dataclass
class Speech:
    speech_id: str
    speaker_id: str
    label: int  # D = 0, R = 1
    windows: list[list[int]]  # Includes BOS/EOS, excludes batch padding.
    year: int = 0
    chamber: str = ""


def _sample(rows, size, partition, seed):
    """Preserve the D/R proportions, rounding the D quota to the nearest integer."""
    if not 2 <= size <= len(rows):
        raise ValueError(f"{partition} sample size must be between 2 and {len(rows)}.")
    groups = {party: [] for party in ("D", "R")}
    for row in rows:
        groups[row["label"]].append(row)
    d_quota = (2 * size * len(groups["D"]) + len(rows)) // (2 * len(rows))
    if d_quota == 0 or d_quota == size:
        raise ValueError(f"{partition} sample cannot represent both parties at size {size}.")

    def priority(row):
        key = f"week2-pilot|{seed}|{partition}|{row['speech_id']}"
        return hashlib.sha256(key.encode("utf-8")).digest()

    selected = []
    for party, count in (("D", d_quota), ("R", size - d_quota)):
        selected.extend(sorted(groups[party], key=priority)[:count])
    return sorted(selected, key=lambda row: row["speech_id"])


def _encoded_speech(row, manifest, tokens):
    sid = row["speech_id"]
    spans = [(int(manifest["window_1_start"]), int(manifest["window_1_stop"]))]
    if manifest["window_2_start"]:
        spans.append((int(manifest["window_2_start"]), int(manifest["window_2_stop"])))
    if len(tokens) != int(manifest["content_tokens"]) or spans != window_ranges(len(tokens)):
        raise ValueError(f"Week 1 tokenizer/window manifest mismatch for {sid}")
    windows = [[0, *tokens[start:stop], 2] for start, stop in spans]
    if row["chamber"] not in ("H", "S") or int(row["year"]) != int(row["date"][:4]):
        raise ValueError(f"Invalid chamber or year for {sid}.")
    return Speech(sid, row["canonical_speaker_id"], ("D", "R").index(row["label"]), windows,
                  int(row["year"]), row["chamber"])


def _read_rows(protocol, partitions):
    if protocol not in ("speaker", "temporal"):
        raise ValueError(f"Unknown protocol: {protocol}.")
    partitions = (partitions,) if isinstance(partitions, str) else tuple(partitions)
    if (not partitions or set(partitions) - {"train", "dev", "test"}
            or len(set(partitions)) != len(partitions)):
        raise ValueError(f"Invalid partitions: {partitions}.")
    corpus_path = restore(DATA)
    split_path = DATA / "splits.csv"
    summary = json.loads((DATA / "split_summary.json").read_text(encoding="utf-8"))
    with split_path.open("rb") as src:
        if hashlib.file_digest(src, "sha256").hexdigest() != summary["splits_sha256"]:
            raise ValueError("The split manifest differs from the frozen Week 1 version.")

    candidates = {phase: [] for phase in partitions}
    column = protocol + "_split"
    with split_path.open(encoding="utf-8", newline="") as src:
        assignments = {row["speech_id"]: row[column] for row in csv.DictReader(src)
                       if row[column] in candidates}
    with corpus_path.open(encoding="utf-8") as src:
        for line in src:
            row = json.loads(line)
            partition = assignments.get(row["speech_id"])
            if partition is not None:
                candidates[partition].append(row)
    previous_speakers = set()
    for phase, rows in candidates.items():
        expected = summary["support"][column][phase]
        speakers = {row["canonical_speaker_id"] for row in rows}
        if len(rows) != expected["speeches"] or len(speakers) != expected["speakers"]:
            raise ValueError(f"Frozen counts differ for {protocol}/{phase}.")
        if protocol == "speaker" and previous_speakers.intersection(speakers):
            raise ValueError("A legislator occurs in more than one speaker partition.")
        previous_speakers.update(speakers)
    return candidates


def _encode_partitions(candidates):
    selected_ids = {r["speech_id"] for rows in candidates.values() for r in rows}
    with (DATA / "windows.csv").open(encoding="utf-8", newline="") as src:
        manifest = {row["speech_id"]: row for row in csv.DictReader(src)
                    if row["speech_id"] in selected_ids}
    tokenizer = load_tokenizer()
    encoded = {}
    for phase, rows in candidates.items():
        speeches = []
        for offset in range(0, len(rows), 128):
            chunk = rows[offset:offset + 128]
            batch = tokenizer.encode_batch([row["text"] for row in chunk], add_special_tokens=False)
            speeches.extend(_encoded_speech(r, manifest[r["speech_id"]], b.ids)
                            for r, b in zip(chunk, batch, strict=True))
        encoded[phase] = speeches
    return encoded


def load_full(protocol, partitions):
    """Encode all speeches in the requested partitions, in sorted speech-ID order."""
    candidates = _read_rows(protocol, partitions)
    for rows in candidates.values():
        rows.sort(key=lambda row: row["speech_id"])
    return _encode_partitions(candidates)


def load_speaker_pilot(train_size, dev_size, seed=4011):
    """Return (train_speeches, dev_speeches), sampled by speech before windowing."""
    sizes = {"train": train_size, "dev": dev_size}
    candidates = _read_rows("speaker", sizes)
    sampled = {phase: _sample(rows, sizes[phase], phase, seed) for phase, rows in candidates.items()}
    encoded = _encode_partitions(sampled)
    return encoded["train"], encoded["dev"]
