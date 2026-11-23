"""Fine-tune text-only RoBERTa-base on a small speaker-disjoint train/dev sample."""

import argparse
import hashlib
import json
import math
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from huggingface_hub import hf_hub_download
from torch.nn import functional as F
from transformers import RobertaConfig, RobertaForSequenceClassification

from week1_windows import ROOT, TOKENIZER_SHA256
from week2_data import load_speaker_pilot
from week2_pilot_results import write_results


MODEL_ID = "FacebookAI/roberta-base"
REVISION = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
WEIGHT_HASH = "5bde1d28afb363d0103324efeb5afc8b2b397fe5e04beabb9b1ef355255ade81"
MODEL_DIR = ROOT / "models/roberta-base"
SAMPLE_SEED = 4011
TRAIN_SEED = 4011
TRAIN_SIZE, DEV_SIZE = 256, 128
MICROBATCH, EFFECTIVE_BATCH = 4, 16  # Measured in speeches, each containing one or two windows.
LR = 2e-5


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def make_model(check_pipeline):
    # The original encoder uses a single token type and 514 positions including padding offsets.
    config = RobertaConfig(vocab_size=50265, hidden_size=768, num_hidden_layers=12,
                           num_attention_heads=12, intermediate_size=3072, type_vocab_size=1,
                           max_position_embeddings=514, layer_norm_eps=1e-5,
                           hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
                           pad_token_id=1, bos_token_id=0, eos_token_id=2,
                           id2label={0: "D", 1: "R"}, label2id={"D": 0, "R": 1})
    if check_pipeline:
        return RobertaForSequenceClassification(config)

    weights = MODEL_DIR / "model.safetensors"
    if not weights.is_file():
        try:
            weights = Path(hf_hub_download(MODEL_ID, weights.name, revision=REVISION,
                                           local_dir=MODEL_DIR))
        except Exception as error:
            raise SystemExit(f"Could not retrieve {MODEL_ID}@{REVISION}. "
                             f"Place the official model.safetensors at {weights}. "
                             f"Download error: {error}") from error
    if file_hash(weights) != WEIGHT_HASH:
        raise ValueError(f"Pretrained weights do not match the official SHA-256: {weights}")
    model, info = RobertaForSequenceClassification.from_pretrained(
        weights.parent, config=config, local_files_only=True, use_safetensors=True,
        output_loading_info=True)
    if (any(not key.startswith("classifier.") for key in info["missing_keys"])
            or info.get("mismatched_keys") or info.get("error_msgs")):
        raise ValueError(f"Incomplete or incompatible pretrained encoder: {info}")
    return model


def speech_log_probs(model, speeches, device, bf16):
    windows = [window for speech in speeches for window in speech.windows]
    lengths = [len(speech.windows) for speech in speeches]
    width = 8 * math.ceil(max(map(len, windows)) / 8)
    ids = torch.tensor([w + [1] * (width - len(w)) for w in windows], device=device)
    # Mask by length, so a literal special-token ID in the original text is still attended.
    mask = torch.arange(width, device=device)[None, :] < torch.tensor(
        [len(w) for w in windows], device=device)[:, None]
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
        logits = model(input_ids=ids, attention_mask=mask).logits
    per_window = F.log_softmax(logits.float(), dim=-1)
    # Stable log of the arithmetic mean of probabilities. Each speech has equal loss weight.
    return torch.stack([chunk.logsumexp(0) - math.log(len(chunk))
                        for chunk in per_window.split(lengths)])


def train_one_epoch(model, speeches, device, bf16):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    ordered = random.Random(TRAIN_SEED).sample(speeches, k=len(speeches))
    probe = model.roberta.encoder.layer[0].attention.self.query.weight
    before = probe.detach().clone()
    loss_sum = 0.0
    steps = 0
    for i in range(0, len(ordered), EFFECTIVE_BATCH):
        group = ordered[i:i + EFFECTIVE_BATCH]
        optimizer.zero_grad(set_to_none=True)
        for j in range(0, len(group), MICROBATCH):
            batch = group[j:j + MICROBATCH]
            logp = speech_log_probs(model, batch, device, bf16)
            labels = torch.tensor([s.label for s in batch], device=device)
            loss = F.nll_loss(logp, labels, reduction="sum")
            (loss / len(group)).backward()
            loss_sum += loss.detach().item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
        optimizer.step()
        steps += 1
    encoder_change = (probe.detach() - before).abs().max().item()
    if not math.isfinite(loss_sum) or not math.isfinite(encoder_change) or encoder_change == 0:
        raise RuntimeError("Training did not produce a finite loss and an encoder weight update.")
    return {"optimizer_steps": steps, "mean_train_speech_loss": loss_sum / len(ordered),
            "max_abs_change_first_layer_query_weight": encoder_change}


@torch.inference_mode()
def predict(model, dev, device, bf16):
    model.eval()
    output = []
    for i in range(0, len(dev), MICROBATCH):
        output.extend(speech_log_probs(model, dev[i:i + MICROBATCH], device, bf16).exp().cpu().tolist())
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-pipeline", action="store_true",
                        help="Explicit engineering check with random initialization, written under runs/.")
    args = parser.parse_args()
    torch.manual_seed(TRAIN_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    model = make_model(args.check_pipeline).to(device)
    train, dev = load_speaker_pilot(TRAIN_SIZE, DEV_SIZE, SAMPLE_SEED)
    mode = "random_init_pipeline_check" if args.check_pipeline else "pretrained_speaker_text_pilot"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = ROOT / ("runs/week2" if args.check_pipeline else "results/week2") / f"{mode}_{stamp}"
    start = time.monotonic()
    print(f"{mode}: {len(train)} train, {len(dev)} dev on {device}.", flush=True)
    train_metrics = train_one_epoch(model, train, device, bf16)
    probabilities = predict(model, dev, device, bf16)
    summary = json.loads((ROOT / "data/week1/split_summary.json").read_text(encoding="utf-8"))
    run = {"mode": mode, "protocol": "speaker_split", "inputs": ["text"],
           "classes_in_order": ["D", "R"], "test_evaluated": False,
           "pretrained_model": None if args.check_pipeline else MODEL_ID,
           "pretrained_revision": None if args.check_pipeline else REVISION,
           "pretrained_sha256": None if args.check_pipeline else WEIGHT_HASH,
           "week1_sha256": {"corpus": summary["corpus_sha256"], "splits": summary["splits_sha256"],
                            "windows": file_hash(ROOT / "data/week1/windows.csv"),
                            "tokenizer": TOKENIZER_SHA256},
           "sample_seed": SAMPLE_SEED, "training_seed": TRAIN_SEED, "epochs": 1,
           "microbatch_speeches": MICROBATCH, "effective_batch_speeches": EFFECTIVE_BATCH,
           "optimizer": "AdamW", "learning_rate": LR, "weight_decay": 0.01, "gradient_clip_norm": 1.0,
           "loss": "Cross-entropy on mean window probabilities with equal weight per speech.",
           "device": str(device), "precision": "bf16 autocast" if bf16 else "float32",
           "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
           "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "transformers": transformers.__version__},
           "training": train_metrics, "train_and_dev_seconds": round(time.monotonic() - start, 2)}
    if not args.check_pipeline:
        checkpoint = ROOT / "checkpoints" / directory.name
        model.save_pretrained(checkpoint)
        run["checkpoint"] = checkpoint.relative_to(ROOT).as_posix()
    metrics = write_results(directory, train, dev, probabilities, run)
    print(directory, flush=True)
    print(json.dumps({"mode": mode, **train_metrics, "dev_macro_f1": metrics["macro_f1"],
                      "dev_accuracy": metrics["accuracy"]}), flush=True)


if __name__ == "__main__":
    main()
