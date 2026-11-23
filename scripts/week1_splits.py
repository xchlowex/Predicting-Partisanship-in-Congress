"""Freeze speaker-disjoint and chronological partitions for the audited Week 1 corpus."""

import argparse
import csv
import hashlib
import io
import json
import unicodedata
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from tempfile import NamedTemporaryFile


REPO = Path(__file__).resolve().parents[1]
SEED = 4011
PARTITIONS = ("train", "dev", "test")
PERCENT = {"train": 70, "dev": 15, "test": 15}
SCORED = ("D", "R")
PERIODS = {"train": [110, 111, 112], "dev": [113], "test": [114]}


def tie_key(*values):
    return hashlib.sha256("|".join(map(str, (SEED, *values))).encode()).digest()


def read_corpus(path):
    corpus_hash = hashlib.sha256()
    ids, texts = set(), set()
    rows = []
    with path.open("rb") as stream:
        for line in stream:
            corpus_hash.update(line)
            item = json.loads(line)
            sid, speaker, label = (item[name] for name in ("speech_id", "canonical_speaker_id", "label"))
            if not all(isinstance(value, str) and value.strip() for value in (sid, speaker, label)):
                raise ValueError(f"Invalid speech ID, canonical speaker ID or label: {sid!r}")
            text = " ".join(unicodedata.normalize("NFKC", item["text"]).casefold().split())
            text_key = hashlib.sha256(text.encode()).digest()
            if sid in ids or text_key in texts or not text:
                raise ValueError(f"Duplicate speech ID, duplicate normalized text or empty text: {sid}")
            ids.add(sid)
            texts.add(text_key)
            day = date.fromisoformat(item["date"])
            rows.append((sid, speaker, label, int(item["congress"]), day))
    return sorted(rows), corpus_hash.hexdigest()


def speaker_partitions(scored):
    counts = defaultdict(Counter)
    for _, speaker, label, _, _ in scored:
        counts[speaker][label] += 1
    strata = defaultdict(list)
    for speaker, histogram in counts.items():
        strata[tuple(sorted(histogram))].append(speaker)

    assigned = {}
    for labels, speakers in sorted(strata.items()):
        # Reserve an exact headcount per partition within each observed party combination.
        numerators = {p: len(speakers) * PERCENT[p] for p in PARTITIONS}
        quota = {p: numerators[p] // 100 for p in PARTITIONS}
        remainder_order = sorted(PARTITIONS, key=lambda p: (-(numerators[p] % 100), tie_key(labels, p)))
        for p in remainder_order[:len(speakers) - sum(quota.values())]:
            quota[p] += 1

        # Allocate prolific speakers first to balance speech volume within these quotas.
        occupancy, speeches = Counter(), Counter()
        for speaker in sorted(speakers, key=lambda s: (-counts[s].total(), tie_key(s))):
            size = counts[speaker].total()
            available = [p for p in PARTITIONS if occupancy[p] < quota[p]]
            p = min(available, key=lambda part: (
                (speeches[part] + size / 2) / PERCENT[part], tie_key(speaker, part)
            ))
            assigned[speaker] = p
            occupancy[p] += 1
            speeches[p] += size
    return assigned


def support(rows):
    by_label = defaultdict(set)
    counts = Counter()
    for _, speaker, label, _, _ in rows:
        by_label[label].add(speaker)
        counts[label] += 1
    return {
        "speeches": len(rows), "speakers": len({r[1] for r in rows}),
        "labels": {label: {"speeches": counts[label], "speakers": len(by_label[label])}
                   for label in sorted(counts)},
    }


def make_manifest(rows):
    scored = [r for r in rows if r[2] in SCORED]
    time_partition = {congress: p for p, congresses in PERIODS.items() for congress in congresses}
    if set(r[3] for r in scored) != set(time_partition):
        raise ValueError("The scored cohort must contain exactly Congresses 110 through 114.")
    assigned = speaker_partitions(scored)
    speaker_sets = {p: set() for p in PARTITIONS}
    partition_rows = {"speaker_split": defaultdict(list), "temporal_split": defaultdict(list)}
    for row in scored:
        _, speaker, _, congress, _ = row
        sp, tp = assigned[speaker], time_partition[congress]
        partition_rows["speaker_split"][sp].append(row)
        partition_rows["temporal_split"][tp].append(row)
        speaker_sets[sp].add(speaker)
    if any(speaker_sets[p] & speaker_sets[q] for i, p in enumerate(PARTITIONS) for q in PARTITIONS[i + 1:]):
        raise ValueError("Speaker overlap across canonical-speaker partitions.")
    for name, batches in partition_rows.items():
        for p in PARTITIONS:
            if {r[2] for r in batches[p]} != set(SCORED):
                raise ValueError(f"{name}/{p} has insufficient scored-class support.")

    temporal = partition_rows["temporal_split"]
    ranges = {p: [min(r[4] for r in temporal[p]), max(r[4] for r in temporal[p])] for p in PARTITIONS}
    if ranges["train"][1] > ranges["dev"][0] or ranges["dev"][1] > ranges["test"][0]:
        raise ValueError("Temporal date ranges cross a train/dev/test period boundary.")
    training_speakers = {r[1] for r in temporal["train"]}
    development_speakers = {r[1] for r in temporal["dev"]}
    manifest, test_slices = [], defaultdict(list)
    for row in scored:
        sid, speaker, _, congress, _ = row
        tp = time_partition[congress]
        status = "train" if tp == "train" else ("seen_train" if speaker in training_speakers else "unseen_train")
        history = status
        if status == "unseen_train":
            history = "seen_dev_only" if tp == "test" and speaker in development_speakers else "new_to_earlier_partitions"
        manifest.append((sid, assigned[speaker], tp, status, history))
        if tp == "test":
            test_slices[status].append(row)
    summary = {
        "scored_labels": list(SCORED), "corpus_support": support(rows),
        "descriptive_only": support([r for r in rows if r[2] not in SCORED]),
        "split_seed": SEED, "speaker_target_percent": PERCENT, "temporal_congresses": PERIODS,
        "temporal_dates_inclusive": {p: [d.isoformat() for d in dates] for p, dates in ranges.items()},
        "support": {name: {p: support(group[p]) for p in PARTITIONS} for name, group in partition_rows.items()},
        "temporal_test_speaker_status": {s: support(test_slices[s]) for s in ("seen_train", "unseen_train")},
        "pairwise_canonical_speaker_overlap_speaker_split": {
            f"{p}_{q}": len(speaker_sets[p] & speaker_sets[q])
            for i, p in enumerate(PARTITIONS) for q in PARTITIONS[i + 1:]
        },
        "partition_unit": "speech_id; every window inherits the speech partition",
        "temporal_speaker_status_definition": "Seen means present in the scored temporal training speeches.",
        "speaker_assignment": ("Within each speaker's set of observed scored parties, reserve speaker counts "
                               "by largest remainder; assign speakers by descending speech volume to balance "
                               "speech counts, breaking ties with SHA-256 and the split seed."),
    }
    return manifest, summary


def write_atomic(destination, data):
    with NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            stream.write(data)
        except BaseException:
            stream.close()
            temp.unlink(missing_ok=True)
            raise
    try:
        temp.replace(destination)
    finally:
        temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO / "data/week1")
    root = parser.parse_args().data_dir
    rows, corpus_digest = read_corpus(root / "corpus.jsonl")
    manifest, summary = make_manifest(rows)
    content = io.StringIO(newline="")
    writer = csv.writer(content, lineterminator="\n")
    writer.writerow(("speech_id", "speaker_split", "temporal_split", "temporal_speaker_status", "temporal_speaker_history"))
    writer.writerows(manifest)
    data = content.getvalue().encode()
    summary["corpus_sha256"] = corpus_digest
    summary["splits_sha256"] = hashlib.sha256(data).hexdigest()
    write_atomic(root / "splits.csv", data)
    write_atomic(root / "split_summary.json", (json.dumps(summary, indent=2) + "\n").encode())
    print(f"Froze {len(manifest):,} scored speeches. Manifest SHA-256: {summary['splits_sha256']}")


if __name__ == "__main__":
    main()
