"""Publish and independently verify the speech-level train/dev pilot outputs."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory


LABELS = ("D", "R")
SAMPLE = "sample.csv"
PREDICTIONS = "dev_predictions.csv"


def sample_row(split, speech):
    if speech.label not in (0, 1):
        raise ValueError(f"Invalid label for {speech.speech_id}")
    return {"split": split, "speech_id": speech.speech_id, "canonical_speaker_id": speech.speaker_id,
            "gold": LABELS[speech.label], "windows": len(speech.windows)}


def predicted_label(probabilities):
    p_d, p_r = map(float, probabilities)
    if not (0 <= p_d <= 1 and 0 <= p_r <= 1 and math.isclose(p_d + p_r, 1.0, abs_tol=1e-6)):
        raise ValueError(f"Invalid D/R probabilities: {p_d}, {p_r}")
    return "D" if p_d >= p_r else "R"


def summarize(sample, predictions):
    sample_by_id = {}
    speakers = {"train": set(), "dev": set()}
    classes = {"train": Counter(), "dev": Counter()}
    windows = Counter()
    for row in sample:
        sid, split = row["speech_id"], row["split"]
        speaker, label, count = row["canonical_speaker_id"], row["gold"], int(row["windows"])
        if not sid or sid in sample_by_id or not speaker or split not in speakers:
            raise ValueError(f"Invalid or repeated sample record: {sid}")
        if label not in LABELS or count not in (1, 2):
            raise ValueError(f"Invalid sample label/window count: {sid}")
        sample_by_id[sid] = row
        speakers[split].add(speaker)
        classes[split][label] += 1
        windows[split] += count
    if speakers["train"] & speakers["dev"]:
        raise ValueError("The speaker-disjoint pilot contains overlapping speakers.")
    counts = {}
    for split in speakers:
        if set(classes[split]) != set(LABELS):
            raise ValueError(f"The {split} sample must include both classes.")
        counts[split] = {"speeches": classes[split].total(), "speakers": len(speakers[split]),
                         "windows": windows[split], "labels": dict(classes[split])}

    seen = set()
    confusion = Counter()
    for row in predictions:
        sid = row["speech_id"]
        source = sample_by_id.get(sid)
        if (sid in seen or source is None or source["split"] != "dev"
                or any(row[field] != source[field] for field in ("gold", "canonical_speaker_id"))):
            raise ValueError(f"Prediction/sample mismatch: {sid}")
        decision = predicted_label((row["p_D"], row["p_R"]))
        if decision != row["predicted"]:
            raise ValueError(f"Prediction does not match class probabilities: {sid}")
        seen.add(sid)
        confusion[row["gold"], decision] += 1
    if len(seen) != counts["dev"]["speeches"]:
        raise ValueError("Development predictions do not cover the entire sample.")

    per_class = {}
    for label in LABELS:
        correct = confusion[label, label]
        support = classes["dev"][label]
        selected = sum(confusion[gold, label] for gold in LABELS)
        per_class[label] = {"support": support, "precision": correct / selected if selected else 0.0,
                            "recall": correct / support, "f1": 2 * correct / (selected + support)}
    metrics = {"accuracy": sum(confusion[k, k] for k in LABELS) / len(seen),
               "macro_f1": sum(row["f1"] for row in per_class.values()) / len(LABELS),
               "class_order": list(LABELS), "per_class": per_class,
               "confusion_gold_by_pred": [[confusion[g, p] for p in LABELS] for g in LABELS]}
    return counts, metrics


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def output_hashes(directory):
    result = {}
    for filename in (SAMPLE, PREDICTIONS):
        with (directory / filename).open("rb") as stream:
            result[filename] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def verify(directory):
    directory = Path(directory)
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    rows = []
    for filename in (SAMPLE, PREDICTIONS):
        with (directory / filename).open(encoding="utf-8", newline="") as stream:
            rows.append(list(csv.DictReader(stream)))
    counts, metrics = summarize(*rows)
    if (counts != run["counts"] or metrics != run["dev_metrics"]
            or output_hashes(directory) != run["output_sha256"]):
        raise ValueError("The published CSVs disagree with run.json.")
    return metrics


def write_results(directory, train, dev, probabilities, run):
    """Save one fresh pilot directory. Probabilities follow dev order, with D first."""
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"Pilot output already exists: {directory}")
    sample = [sample_row(split, speech) for split, group in (("train", train), ("dev", dev))
              for speech in group]
    predictions = []
    for speech, pair in zip(dev, probabilities, strict=True):
        p_d, p_r = map(float, pair)
        predictions.append({"speech_id": speech.speech_id, "canonical_speaker_id": speech.speaker_id,
                            "gold": LABELS[speech.label], "predicted": predicted_label((p_d, p_r)),
                            "p_D": p_d, "p_R": p_r})
    counts, metrics = summarize(sample, predictions)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=directory.parent) as temporary:
        stage = Path(temporary) / "result"
        stage.mkdir()
        write_csv(stage / SAMPLE, sample)
        write_csv(stage / PREDICTIONS, predictions)
        record = {**run, "counts": counts, "dev_metrics": metrics, "output_sha256": output_hashes(stage)}
        (stage / "run.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        verify(stage)
        stage.rename(directory)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", required=True, type=Path, metavar="RUN_DIRECTORY")
    print(json.dumps(verify(parser.parse_args().verify), indent=2))
