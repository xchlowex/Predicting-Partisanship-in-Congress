"""Score JSONL speeches with an original, selected Week 2 checkpoint."""

import argparse
import csv
import hashlib
import json
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from week1_windows import ROOT, TOKENIZER_SHA256, encode_windows, load_tokenizer
from week2_data import Speech
from week2_model import CHAMBER, build_for_reload, log_probs
from week2_text_pilot import MICROBATCH


def load_selected(name, device):
    if Path(name).name != name or name in ("", ".", ".."):
        raise ValueError("Use the run directory name, for example speaker_both_s4011.")
    run_file = ROOT / "results/week2/formal" / name / "run.json"
    run = json.loads(run_file.read_text(encoding="utf-8"))
    identity = f"{run['protocol']}_{run['variant']}_s{run['seed']}"
    relative = Path("checkpoints/week2_formal") / name / f"epoch{run['best_epoch']}.pt"
    if identity != name or Path(run["checkpoint"]) != relative:
        raise ValueError(f"Run identity or selected checkpoint path differs in {run_file}")
    if run["class_order"] != ["D", "R"] or run["week1_sha256"]["tokenizer"] != TOKENIZER_SHA256:
        raise ValueError("Run class order or tokenizer differs from the Week 2 configuration.")

    with (ROOT / relative).open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != run["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint SHA-256 differs from {run_file}")
        stream.seek(0)
        state = torch.load(stream, map_location="cpu", weights_only=True)
    model = build_for_reload(run["variant"], run["metadata_stats"])
    model.load_state_dict(state["model"], strict=True)
    bf16 = (run["training"]["autocast"] == "bfloat16" and device.type == "cuda"
            and torch.cuda.is_bf16_supported())
    return model.eval().to(device), bf16


def read_speeches(path, tokenizer):
    seen = set()
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
                sid, text, year, chamber = (row[key] for key in ("speech_id", "text", "year", "chamber"))
                if not isinstance(sid, str) or not sid.strip() or sid in seen:
                    raise ValueError("speech_id must be a unique nonempty string")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("text must be a nonempty string")
                if type(year) is not int or chamber not in CHAMBER:
                    raise ValueError("year must be an integer and chamber must be H or S")
                windows = encode_windows(text, tokenizer)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}, line {line_number}: {exc}") from exc
            seen.add(sid)
            # Gold label and legislator ID are unused by the forward pass.
            yield Speech(sid, "", -1, windows, year, chamber)
    if not seen:
        raise ValueError(f"No speeches in {path}")


@torch.inference_mode()
def score(run, source, output, device):
    if output.exists():
        raise FileExistsError(f"Choose a new output CSV: {output}")
    model, bf16 = load_selected(run, device)
    speeches = read_speeches(source, load_tokenizer())
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with TemporaryDirectory(dir=output.parent) as folder:
        staged = Path(folder) / "predictions.csv"
        with staged.open("w", encoding="utf-8", newline="") as dst:
            writer = csv.writer(dst, lineterminator="\n")
            writer.writerow(["speech_id", "predicted", "p_D", "p_R"])
            while batch := list(islice(speeches, MICROBATCH)):
                probs = log_probs(model, batch, device, bf16).exp().cpu()
                if (not torch.isfinite(probs).all()
                        or not torch.allclose(probs.sum(1), torch.ones(len(batch)), atol=1e-6)):
                    raise ValueError("Invalid speech probabilities")
                for speech, (pd, pr) in zip(batch, probs.tolist(), strict=True):
                    writer.writerow([speech.speech_id, "R" if pr > pd else "D", pd, pr])
                count += len(batch)
        # Reserve the output name even when two jobs finish at the same time.
        output.touch(exist_ok=False)
        try:
            staged.replace(output)
        except OSError:
            output.unlink(missing_ok=True)
            raise
    precision = "bfloat16 autocast" if bf16 else "float32"
    print(f"Scored {count} speeches with {run} on {device} ({precision}): {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Published Week 2 run, e.g. speaker_both_s4011")
    parser.add_argument("--input", type=Path, required=True, help="UTF-8 JSONL: speech_id, text, year, chamber")
    parser.add_argument("--output", type=Path, required=True, help="New predictions CSV file")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Choose --device mps, cpu, or auto.")
    if device == "mps" and not torch.backends.mps.is_available():
        parser.error("The Apple GPU is unavailable. Choose --device cpu or auto.")
    score(args.run, args.input, args.output, torch.device(device))


if __name__ == "__main__":
    main()

