"""Run the predeclared full-corpus Week 2 comparisons, selecting checkpoints on development only."""
import argparse
import csv
import gc
import json
import math
import platform
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import transformers
from filelock import FileLock
from torch.nn import functional as F

from week1_windows import ROOT, TOKENIZER_SHA256
from week2_data import load_full
from week2_model import build_for_reload, build_model, log_probs
from week2_text_pilot import EFFECTIVE_BATCH, LR, MICROBATCH, MODEL_ID, REVISION, WEIGHT_HASH, file_hash

EPOCHS = 3
META_LR = 0.001
DEVICE = torch.device("cuda")
OUTPUT = ROOT / "results/week2/formal"
LOCAL = ROOT / "checkpoints/week2_formal"
LABELS = ("D", "R")
MATRIX = [(p, v, 4011) for v in ("text", "both", "year", "chamber") for p in ("speaker", "temporal")]
MATRIX += [(p, v, seed) for seed in (4012, 4013) for p in ("speaker", "temporal") for v in ("text", "both")]


class BudgetExpired(Exception):
    pass


def stamp():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    print(stamp(), message, flush=True)


def check_budget(deadline):
    if deadline is not None and time.time() > deadline:
        raise BudgetExpired()


def save_torch(obj, path):
    part = path.with_suffix(".tmp")
    torch.save(obj, part)
    part.replace(path)


def train_epoch(model, optimizer, speeches, seed, epoch, bf16, deadline):
    model.train()
    ordered = random.Random(seed + epoch - 1).sample(speeches, k=len(speeches))
    total_loss = torch.zeros((), device=DEVICE)
    for i in range(0, len(ordered), EFFECTIVE_BATCH):
        if (i // EFFECTIVE_BATCH) % 25 == 0:
            check_budget(deadline)
        group = ordered[i:i + EFFECTIVE_BATCH]
        optimizer.zero_grad(set_to_none=True)
        for j in range(0, len(group), MICROBATCH):
            batch = group[j:j + MICROBATCH]
            lp = log_probs(model, batch, DEVICE, bf16)
            labels = torch.tensor([s.label for s in batch], device=DEVICE)
            loss = F.nll_loss(lp, labels, reduction="sum")
            (loss / len(group)).backward()
            total_loss += loss.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        step = i // EFFECTIVE_BATCH + 1
        if step % 200 == 0:
            log(f"  epoch {epoch} step {step}/{math.ceil(len(ordered) / EFFECTIVE_BATCH)}")
    mean = total_loss.item() / len(speeches)
    if not math.isfinite(mean):
        raise ValueError("Non-finite training loss.")
    return mean


@torch.inference_mode()
def predict(model, speeches, bf16):
    model.eval()
    out = []
    for i in range(0, len(speeches), MICROBATCH):
        out.append(log_probs(model, speeches[i:i + MICROBATCH], DEVICE, bf16).exp())
    probs = torch.cat(out).cpu()
    if not torch.isfinite(probs).all() or not torch.allclose(probs.sum(1), torch.ones(len(probs)), atol=1e-6):
        raise ValueError("Invalid speech probabilities.")
    return probs


def metrics(rows, probs):
    pairs = Counter((s.label, p) for s, p in zip(rows, probs.argmax(1).tolist(), strict=True))
    cm = [[pairs[a, b] for b in range(2)] for a in range(2)]
    f1 = [2 * cm[k][k] / (2 * cm[k][k] + cm[0][1] + cm[1][0]) for k in range(2)]
    return {"accuracy": (cm[0][0] + cm[1][1]) / len(rows), "macro_f1": sum(f1) / 2,
            "confusion_gold_by_pred": cm}


def export_csv(path, rows, probs):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["speech_id", "canonical_speaker_id", "gold", "predicted", "p_D", "p_R"])
        for s, (pd, pr) in zip(rows, probs.tolist(), strict=True):
            writer.writerow([s.speech_id, s.speaker_id, LABELS[s.label], LABELS[int(pr > pd)], pd, pr])


def run_one(protocol, variant, seed, data, deadline):
    name = f"{protocol}_{variant}_s{seed}"
    final = OUTPUT / name
    if final.exists():
        record = json.loads((final / "run.json").read_text(encoding="utf-8"))
        if any(file_hash(final / filename) != digest for filename, digest in record["output_sha256"].items()):
            raise ValueError(f"Saved prediction files changed for {name}")
        return
    check_budget(deadline)
    log(f"Starting {name}")
    start = time.monotonic()
    train, dev = data["train"], data["dev"]
    torch.manual_seed(seed)
    bf16 = torch.cuda.is_bf16_supported()
    model = build_model(variant, train).to(DEVICE)
    stats = model.metadata_stats
    groups = [{"params": model.encoder.parameters(), "lr": LR, "weight_decay": 0.01}]
    if model.metadata_weight is not None:
        groups.append({"params": [model.metadata_weight], "lr": META_LR, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(groups)
    ckpt = LOCAL / name
    ckpt.mkdir(parents=True, exist_ok=True)
    resume = ckpt / "resume.pt"
    history = []
    best_epoch = 0
    if resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history, best_epoch = state["history"], state["best_epoch"]
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["gpu_rng"])
        del state
        log(f"Resuming {name} after epoch {len(history)}")
    for epoch in range(len(history) + 1, EPOCHS + 1):
        t = time.monotonic()
        train_loss = train_epoch(model, optimizer, train, seed, epoch, bf16, deadline)
        dp = predict(model, dev, bf16)
        dev_score = metrics(dev, dp)
        improved = not history or dev_score["macro_f1"] > max(r["dev_macro_f1"] for r in history)
        history.append({"epoch": epoch, "dev_macro_f1": dev_score["macro_f1"],
                        "dev_accuracy": dev_score["accuracy"], "train_loss": train_loss,
                        "optimizer_steps": math.ceil(len(train) / EFFECTIVE_BATCH),
                        "seconds": round(time.monotonic() - t, 3)})
        if improved:
            best_epoch = epoch
            # Separate epoch files keep the previously published resume state valid during a write.
            save_torch({"model": model.state_dict(), "dev_probs": dp}, ckpt / f"epoch{epoch}.pt")
        save_torch({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "history": history, "best_epoch": best_epoch, "cpu_rng": torch.get_rng_state(),
                    "gpu_rng": torch.cuda.get_rng_state()}, resume)
        for old in ckpt.glob("epoch*.pt"):
            if old.name != f"epoch{best_epoch}.pt":
                old.unlink()
        log(f"{name} epoch complete: {json.dumps(history[-1])}")
    del optimizer, model, groups
    gc.collect()
    torch.cuda.empty_cache()
    best_path = ckpt / f"epoch{best_epoch}.pt"
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    model = build_for_reload(variant, stats)
    model.load_state_dict(best["model"])
    model.to(DEVICE)
    dp = predict(model, dev, bf16)
    drift = (dp - best["dev_probs"]).abs().max().item()
    if drift > 1e-6 or not torch.equal(dp.argmax(1), best["dev_probs"].argmax(1)):
        raise ValueError(f"Checkpoint replay differs for {name}: {drift}")
    # The full configuration, selection rule and seed grid are fixed before any test score is read.
    test = load_full(protocol, ["test"])["test"]
    tp = predict(model, test, bf16)
    summary = json.loads((ROOT / "data/week1/split_summary.json").read_text(encoding="utf-8"))
    with TemporaryDirectory(dir=OUTPUT.parent) as temp:
        stage = Path(temp) / name
        stage.mkdir()
        for phase, rows, probs in (("dev", dev, dp), ("test", test, tp)):
            export_csv(stage / f"{phase}_predictions.csv", rows, probs)
        record = {"protocol": protocol, "variant": variant, "seed": seed, "best_epoch": best_epoch,
                  "epochs": history, "class_order": LABELS, "metadata_stats": stats,
                  "metadata_weight": None if model.metadata_weight is None else model.metadata_weight.detach().cpu().tolist(),
                  "counts": {p: {"speeches": len(ss), "speakers": len({s.speaker_id for s in ss}),
                                 "windows": sum(len(s.windows) for s in ss)}
                             for p, ss in (("train", train), ("dev", dev), ("test", test))},
                  "dev_metrics": metrics(dev, dp), "test_metrics": metrics(test, tp),
                  "dev_checkpoint_replay_max_abs_error": drift,
                  "pretrained_model": MODEL_ID, "pretrained_revision": REVISION, "pretrained_sha256": WEIGHT_HASH,
                  "training": {"selection": "Maximum development speech macro-F1, earliest epoch on ties.",
                               "objective": "Mean speech cross-entropy on arithmetic mean window probabilities.",
                               "optimizer": "AdamW", "encoder_lr": LR, "encoder_weight_decay": 0.01,
                               "metadata_lr": META_LR if variant != "text" else None, "metadata_weight_decay": 0.,
                               "schedule": "constant", "gradient_clip_norm": 1., "effective_batch_speeches": EFFECTIVE_BATCH,
                               "microbatch_speeches": MICROBATCH, "autocast": "bfloat16" if bf16 else "disabled",
                               "shuffle": "Python Random(seed + epoch - 1), canonical speech-ID order before sampling."},
                  "week1_sha256": {"corpus": summary["corpus_sha256"], "splits": summary["splits_sha256"],
                                   "windows": file_hash(ROOT / "data/week1/windows.csv"), "tokenizer": TOKENIZER_SHA256},
                  "checkpoint": best_path.relative_to(ROOT).as_posix(), "checkpoint_sha256": file_hash(best_path),
                  "output_sha256": {f.name: file_hash(f) for f in stage.glob("*.csv")},
                  "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                               "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name()},
                  "finished_utc": stamp(), "this_invocation_seconds": round(time.monotonic() - start, 2)}
        (stage / "run.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        stage.rename(final)
    resume.unlink(missing_ok=True)
    log(f"Published {name} best_epoch={best_epoch} dev_macro_f1={record['dev_metrics']['macro_f1']:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline", help="ISO 8601 wall-clock deadline with timezone, for unattended training.")
    parser.add_argument("--only", choices=[f"{p}_{v}_s{s}" for p,v,s in MATRIX], help="Run one declared configuration.")
    args = parser.parse_args()
    deadline = None
    if args.deadline:
        dt = datetime.fromisoformat(args.deadline)
        if dt.tzinfo is None:
            parser.error("--deadline must include the timezone.")
        deadline = dt.timestamp()
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device available for the formal training queue.")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    cache = {}
    with FileLock(str(LOCAL / "training.lock"), timeout=0):
        try:
            for protocol, variant, seed in MATRIX:
                name = f"{protocol}_{variant}_s{seed}"
                if args.only and name != args.only:
                    continue
                check_budget(deadline)
                if protocol not in cache:
                    cache[protocol] = load_full(protocol, ("train", "dev"))
                run_one(protocol, variant, seed, cache[protocol], deadline)
        except BudgetExpired:
            log("Deadline reached. Finished epochs remain resumable.")
            return
    log("Training queue complete.")


if __name__ == "__main__":
    main()
