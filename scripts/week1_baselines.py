"""Reproduce the three fixed CPU baselines on the frozen Week 1 speech splits."""

import argparse
import csv
import hashlib
import json
import os
import platform
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import scipy
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
LABELS = ["D", "R"]
SEED = 4011
BOOTSTRAPS = 1000
MODELS = ("majority", "tfidf_lr", "year_chamber_lr")
PROTOCOLS = ("speaker", "temporal")
TFIDF = dict(ngram_range=(1, 2), min_df=3, max_features=60000,
             sublinear_tf=True, norm="l2", lowercase=True)
LR = dict(C=1.0, solver="liblinear", tol=1e-4, max_iter=500, random_state=SEED)


def load_inputs(corpus_path, splits_path):
    with corpus_path.open(encoding="utf-8") as stream:
        corpus = [json.loads(line) for line in stream]
    with splits_path.open(encoding="utf-8", newline="") as stream:
        splits = list(csv.DictReader(stream))
    manifests = {row["speech_id"]: row for row in splits}
    corpus_ids = {r["speech_id"] for r in corpus}
    if len(corpus_ids) != len(corpus) or len(manifests) != len(splits):
        raise ValueError("Duplicate speech ID in the corpus or split manifest.")
    scored = sorted((r for r in corpus if r["label"] in LABELS), key=lambda r: r["speech_id"])
    if not scored or set(manifests) != {r["speech_id"] for r in scored}:
        raise ValueError("Manifest does not cover exactly the D/R corpus.")
    for row in scored:
        split = manifests[row["speech_id"]]
        for protocol in PROTOCOLS:
            tag = split[f"{protocol}_split"]
            if tag not in {"train", "dev", "test"}:
                raise ValueError(f"Invalid {protocol} split: {tag}")
            row[f"{protocol}_split"] = tag
        row["temporal_speaker_status"] = split["temporal_speaker_status"]
        if not row["canonical_speaker_id"] or row["chamber"] not in {"H", "S"} or not row["text"].strip():
            raise ValueError(f"Incomplete data for {row['speech_id']}")
    return scored


def features(name, rows):
    if name == "tfidf_lr":
        return [r["text"] for r in rows]
    if name == "year_chamber_lr":
        return np.array([[int(r["year"]), int(r["chamber"] == "S")] for r in rows])
    return np.zeros((len(rows), 1))


def fit_model(name, train):
    if name == "majority":
        model = DummyClassifier(strategy="prior")
    elif name == "tfidf_lr":
        model = make_pipeline(TfidfVectorizer(**TFIDF), LogisticRegression(**LR))
    else:
        model = make_pipeline(StandardScaler(), LogisticRegression(**LR))
    model.fit(features(name, train), [r["label"] for r in train])
    if list(model.classes_) != LABELS:
        raise ValueError("Training partition must contain both frozen classes.")
    return model


def scores(matrices):
    # Last two axes index the gold class and the predicted class, respectively.
    tp = np.diagonal(matrices, axis1=-2, axis2=-1)
    support = matrices.sum(axis=-1)
    predicted = matrices.sum(axis=-2)
    denominator = support + predicted
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0)
    return f1.mean(axis=-1), tp.sum(axis=-1) / support.sum(axis=-1)


def evaluate(rows, predictions):
    speaker_ids, speaker_index = np.unique([r["canonical_speaker_id"] for r in rows], return_inverse=True)
    clusters = np.zeros((len(speaker_ids), len(LABELS), len(LABELS)), dtype=int)
    gold_index = np.array([LABELS.index(r["label"]) for r in rows])
    np.add.at(clusters, (speaker_index, gold_index, predictions), 1)
    observed = clusters.sum(axis=0)
    macro_f1, accuracy = scores(observed)
    # Draw whole legislators, retaining all their speeches and repeated draws.
    # The same ordered speaker list and seed pair the bootstrap draws across models.
    rng = np.random.default_rng(SEED)
    draws = rng.integers(len(clusters), size=(BOOTSTRAPS, len(clusters)))
    boot_f1, boot_acc = scores(clusters[draws].sum(axis=1))
    summary = {
        "speeches": len(rows), "speakers": len(clusters), "macro_f1": float(macro_f1),
        "macro_f1_ci_low": float(np.quantile(boot_f1, 0.025)),
        "macro_f1_ci_high": float(np.quantile(boot_f1, 0.975)),
        "accuracy": float(accuracy),
        "accuracy_ci_low": float(np.quantile(boot_acc, 0.025)),
        "accuracy_ci_high": float(np.quantile(boot_acc, 0.975)),
    }
    per_class = {}
    for i, label in enumerate(LABELS):
        gold, pred, tp = int(observed[i].sum()), int(observed[:, i].sum()), int(observed[i, i])
        per_class[label] = {
            "support": gold, "precision": tp / pred if pred else 0.0,
            "recall": tp / gold if gold else 0.0, "f1": 2 * tp / (pred + gold) if pred + gold else 0.0,
        }
    return summary, {"confusion_gold_by_pred": observed.tolist(), "per_class": per_class}


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(rows):
    forecasts, summary_rows, detail_rows, fitted = [], [], [], []
    for protocol in PROTOCOLS:
        partitions = {tag: [r for r in rows if r[f"{protocol}_split"] == tag] for tag in ("train", "dev", "test")}
        train = partitions["train"]
        train_speakers = {r["canonical_speaker_id"] for r in train}
        for partition in ("dev", "test"):
            for row in partitions[partition]:
                seen = row["canonical_speaker_id"] in train_speakers
                if protocol == "speaker" and seen:
                    raise ValueError("Speaker leakage into evaluation.")
                if protocol == "temporal" and row["temporal_speaker_status"] != ("seen_train" if seen else "unseen_train"):
                    raise ValueError("Temporal speaker status disagrees with the training partition.")
        for name in MODELS:
            print(f"Fitting {protocol}/{name} on {len(train):,} speeches", flush=True)
            model = fit_model(name, train)
            fit = {"protocol": protocol, "model": name, "train_class_counts": dict(Counter(r["label"] for r in train))}
            if name == "tfidf_lr":
                fit["vocabulary_size"] = len(model[0].vocabulary_)
            if name == "year_chamber_lr":
                fit["feature_order"] = ["year", "is_senate"]
                fit["training_mean"] = model[0].mean_.tolist()
                fit["training_scale"] = model[0].scale_.tolist()
                fit["republican_logit_coefficients"] = model[-1].coef_[0].tolist()
                fit["intercept"] = model[-1].intercept_.item()
            fitted.append(fit)
            for tag in ("dev", "test"):
                observed = partitions[tag]
                prob = model.predict_proba(features(name, observed))
                if not np.isfinite(prob).all() or not np.allclose(prob.sum(axis=1), 1):
                    raise ValueError("Invalid class probabilities.")
                predictions = prob.argmax(axis=1)
                for idx, r in enumerate(observed):
                    forecasts.append({
                        "protocol": protocol, "partition": tag, "model": name, "speech_id": r["speech_id"],
                        "canonical_speaker_id": r["canonical_speaker_id"], "congress": r["congress"],
                        "date": r["date"], "chamber": r["chamber"], "speaker_status": (
                            r["temporal_speaker_status"] if protocol == "temporal" else "unseen_train"),
                        "gold": r["label"], "predicted": LABELS[predictions[idx]],
                        "p_D": f"{prob[idx, 0]:.12f}", "p_R": f"{prob[idx, 1]:.12f}",
                    })
                groups = {"all": np.arange(len(observed))}
                if protocol == "temporal":
                    statuses = np.array([r["temporal_speaker_status"] for r in observed])
                    groups.update({status: np.flatnonzero(statuses == status) for status in ("seen_train", "unseen_train")})
                for subgroup, idx in groups.items():
                    if len(idx) == 0:
                        continue
                    keys = dict(protocol=protocol, partition=tag, model=name, subgroup=subgroup)
                    metrics, details = evaluate([observed[i] for i in idx], predictions[idx])
                    summary_rows.append({**keys, **metrics})
                    detail_rows.append({**keys, **details})
    return forecasts, summary_rows, detail_rows, fitted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/week1/corpus.jsonl")
    parser.add_argument("--splits", type=Path, default=ROOT / "data/week1/splits.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/week1")
    args = parser.parse_args()
    forecasts, summaries, details, fitted = run(load_inputs(args.corpus, args.splits))
    args.output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output) as tmp:
        stage = Path(tmp)
        write_csv(stage / "predictions.csv", forecasts)
        write_csv(stage / "metrics.csv", summaries)
        (stage / "class_metrics.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
        filenames = ["predictions.csv", "metrics.csv", "class_metrics.json"]
        settings = {
            "classes_in_order": LABELS,
            "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                         "numpy": np.__version__, "scipy": scipy.__version__},
            "input_sha256": {"corpus": checksum(args.corpus), "splits": checksum(args.splits)},
            "output_sha256": {name: checksum(stage / name) for name in filenames},
            "majority": "Most frequent train label; probabilities equal training class frequencies.",
            "tfidf_lr": {"input": "full cleaned speech", "vectorizer": TFIDF, "logistic_regression": LR},
            "year_chamber_lr": {"input": ["numeric year", "is_senate"], "scaling": "train mean and standard deviation",
                                "logistic_regression": LR},
            "selection": "Configurations fixed before scoring. Only training data fit all models and transforms.",
            "bootstrap": {"unit": "canonical_speaker_id", "replicates": BOOTSTRAPS, "seed": SEED,
                          "interval": "2.5th and 97.5th percentiles, conditional on fitted model"},
            "fitted": fitted,
        }
        (stage / "run.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
        for filename in [*filenames, "run.json"]:
            os.replace(stage / filename, args.output / filename)
    print(f"Wrote {len(forecasts):,} predictions and {len(summaries)} metric rows to {args.output}")


if __name__ == "__main__":
    main()
