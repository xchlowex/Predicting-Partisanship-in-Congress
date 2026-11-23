"""Audit completed Week 2 RoBERTa runs against the frozen speech-level inputs."""

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from verify_week1 import DATA, ROOT, csv_rows, digest, require
from week1_archive import restore


FORMAL = ROOT / "results/week2/formal"
PHASES = ("train", "dev", "test")
PROTOCOLS = ("speaker", "temporal")
FEATURES = {"text": [], "year": ["year"], "chamber": ["chamber"], "both": ["year", "chamber"]}
REVISION = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
WEIGHTS = "5bde1d28afb363d0103324efeb5afc8b2b397fe5e04beabb9b1ef355255ade81"


def frozen_inputs():
    path = restore(DATA)
    week1 = json.loads((ROOT / "results/week1/run.json").read_text(encoding="utf-8"))
    hashes = {"corpus": digest(path), "splits": digest(DATA / "splits.csv"),
              "windows": digest(DATA / "windows.csv"),
              "tokenizer": digest(ROOT / "data/reference/roberta-base/tokenizer.json")}
    for key in ("corpus", "splits"):
        require(hashes[key] == week1["input_sha256"][key], f"Frozen {key} SHA-256 mismatch")
    with path.open(encoding="utf-8") as stream:
        corpus = {row["speech_id"]: (row["label"], row["canonical_speaker_id"],
                                     row["year"], row["congress"])
                  for row in map(json.loads, stream) if row["label"] in ("D", "R")}
    window_count = {r["speech_id"]: 1 + bool(r["window_2_start"])
                    for r in csv_rows(DATA / "windows.csv")}
    groups = defaultdict(dict)
    for r in csv_rows(DATA / "splits.csv"):
        sid = r["speech_id"]
        gold, speaker, year, congress = corpus[sid]
        for protocol in PROTOCOLS:
            phase = r[protocol + "_split"]
            require(phase in PHASES, f"Unknown {protocol} partition for {sid}")
            groups[protocol, phase][sid] = (gold, speaker, year, window_count[sid])
        expected_time = "train" if congress <= 112 else "dev" if congress == 113 else "test"
        require(r["temporal_split"] == expected_time, f"Wrong chronological partition: {sid}")
    frozen = {}
    for protocol in PROTOCOLS:
        rows = {phase: groups[protocol, phase] for phase in PHASES}
        require(sum(map(len, rows.values())) == len(corpus), f"Incomplete {protocol} partition")
        speakers = {phase: {r[1] for r in items.values()} for phase, items in rows.items()}
        if protocol == "speaker":
            require(sum(map(len, speakers.values())) == len(set.union(*speakers.values())),
                    "Frozen speaker partitions overlap")
        counts = {phase: {"speeches": len(items), "speakers": len(speakers[phase]),
                          "windows": sum(r[3] for r in items.values())} for phase, items in rows.items()}
        years = [r[2] for r in rows["train"].values()]
        frozen[protocol] = rows, counts, (statistics.fmean(years), statistics.pstdev(years) or 1.0)
    return frozen, hashes


def score_predictions(path, expected):
    seen = set()
    matrix = Counter()
    for r in csv_rows(path):
        sid = r["speech_id"]
        require(sid in expected and sid not in seen, f"{path.name}: unexpected/duplicate ID {sid}")
        seen.add(sid)
        require((r["gold"], r["canonical_speaker_id"]) == expected[sid][:2],
                f"{path.name}: wrong speaker or label for {sid}")
        p_d, p_r = float(r["p_D"]), float(r["p_R"])
        require(0 <= p_d <= 1 and 0 <= p_r <= 1 and abs(p_d + p_r - 1) < 1e-6,
                f"{path.name}: invalid D/R probabilities for {sid}")
        predicted = "D" if p_d >= p_r else "R"
        require(predicted == r["predicted"], f"{path.name}: incorrect argmax for {sid}")
        matrix[r["gold"], predicted] += 1
    require(len(seen) == len(expected), f"{path.name}: missing {len(expected) - len(seen)} speeches")
    errors = matrix["D", "R"] + matrix["R", "D"]
    accuracy = (len(seen) - errors) / len(seen)
    f1 = sum(2 * matrix[c, c] / (2 * matrix[c, c] + errors) for c in ("D", "R")) / 2
    confusion = [[matrix[g, p] for p in ("D", "R")] for g in ("D", "R")]
    return accuracy, f1, confusion


def close(actual, expected, name):
    require(math.isclose(float(actual), expected, abs_tol=1e-9, rel_tol=1e-9),
            f"{name}: recorded {actual}, expected {expected}")


def check_run(directory, frozen, hashes, check_checkpoint=False):
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    protocol, variant, seed = run["protocol"], run["variant"], run["seed"]
    require(directory.name == f"{protocol}_{variant}_s{seed}", "Run identity differs from directory")
    require(run["class_order"] == ["D", "R"], "Unexpected class order")
    rows, counts, (year_mean, year_scale) = frozen[protocol]
    require(run["week1_sha256"] == hashes, "Week 1 input hash mismatch")
    require((run["pretrained_revision"], run["pretrained_sha256"]) == (REVISION, WEIGHTS),
            "Pretrained model identity mismatch")
    stats = run["metadata_stats"]
    require(stats["features"] == FEATURES[variant], "Metadata inputs differ from the named variant")
    if "year" in stats["features"]:
        close(stats["year_mean"], year_mean, "Training-set year mean")
        close(stats["year_scale"], year_scale, "Training-set year scale")
    if "chamber" in stats["features"]:
        require(stats["chamber_encoding"] == {"H": 0, "S": 1}, "Unexpected chamber encoding")
    weights = run["metadata_weight"]
    if stats["features"]:
        require(len(weights) == len(stats["features"]) and
                all(len(w) == 2 and all(math.isfinite(x) for x in w) for w in weights),
                "Invalid fitted metadata weights")
    else:
        require(weights is None, "Text-only model contains metadata weights")
    for phase, entries in counts.items():
        for key, n in entries.items():
            require(run["counts"][phase][key] == n, f"{phase} {key} count mismatch")
    epochs = run["epochs"]
    require([r["epoch"] for r in epochs] == [1, 2, 3], "Expected three ordered training epochs")
    batch = run["training"]["effective_batch_speeches"]
    require(isinstance(batch, int) and batch >= 1, "Invalid effective batch size")
    expected_steps = math.ceil(counts["train"]["speeches"] / batch)
    for epoch in epochs:
        require(0 <= epoch["dev_macro_f1"] <= 1 and 0 <= epoch["dev_accuracy"] <= 1,
                "Invalid development metric in epoch history")
        require(math.isfinite(epoch["train_loss"]) and epoch["train_loss"] >= 0,
                "Nonfinite or negative training loss")
        require(math.isfinite(epoch["seconds"]) and epoch["seconds"] > 0, "Invalid epoch duration")
        require(epoch["optimizer_steps"] == expected_steps, "Optimizer steps disagree with full train size")
    best = max(epochs, key=lambda r: r["dev_macro_f1"])
    require(best["epoch"] == run["best_epoch"], "Checkpoint selection disagrees with development macro-F1")
    require(0 <= run["dev_checkpoint_replay_max_abs_error"] <= 1e-6, "Checkpoint replay drift")
    for phase in ("dev", "test"):
        filename = f"{phase}_predictions.csv"
        path = directory / filename
        require(digest(path) == run["output_sha256"][filename], f"{filename}: checksum mismatch")
        accuracy, f1, confusion = score_predictions(path, rows[phase])
        metrics = run[phase + "_metrics"]
        close(metrics["accuracy"], accuracy, f"{phase} accuracy")
        close(metrics["macro_f1"], f1, f"{phase} macro-F1")
        require(metrics["confusion_gold_by_pred"] == confusion, f"{phase} confusion matrix mismatch")
        if phase == "dev":
            close(best["dev_macro_f1"], f1, "Selected epoch macro-F1")
            close(best["dev_accuracy"], accuracy, "Selected epoch accuracy")
    if check_checkpoint:
        checkpoint = ROOT / run["checkpoint"]
        require(digest(checkpoint) == run["checkpoint_sha256"], "Local checkpoint SHA-256 mismatch")
    return protocol, variant, seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", metavar="RUN_DIRECTORY", type=Path, nargs="*",
                        help="Default: all published formal run directories.")
    parser.add_argument("--require", choices=("available", "core", "all"), default="available",
                        help="Require the eight core runs or all sixteen planned runs.")
    parser.add_argument("--check-checkpoints", action="store_true", help="Also hash each local checkpoint.")
    args = parser.parse_args()
    directories = args.runs or sorted(path.parent for path in FORMAL.glob("*/run.json")
                                      if not path.parent.name.startswith("."))
    if not directories:
        parser.exit(1, "No completed formal runs to verify.\n")
    frozen, hashes = frozen_inputs()
    verified = set()
    for directory in directories:
        try:
            identity = check_run(directory, frozen, hashes, args.check_checkpoints)
            require(identity not in verified, "Duplicate run identity")
        except (KeyError, TypeError, ValueError, OSError) as error:
            parser.exit(1, f"{directory}: {error}\n")
        verified.add(identity)
    required = {(p, v, 4011) for p in PROTOCOLS for v in FEATURES} if args.require != "available" else set()
    if args.require == "all":
        required.update((p, v, s) for p in PROTOCOLS for v in ("text", "both") for s in (4012, 4013))
    missing = sorted(required - verified)
    if missing:
        parser.exit(1, f"Missing {len(missing)} required run(s): {missing}\n")
    print(f"Verified {len(verified)} formal runs, their selected epochs, and complete dev/test predictions.")


if __name__ == "__main__":
    main()
