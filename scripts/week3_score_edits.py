"""Score prepared edits with the two temporal checkpoints.

Prints a line after every batch so a terminal shows that scoring is moving.
The saved weights are not updated. A finished score file is left in place.
"""

import argparse
import csv
import time
from pathlib import Path

import torch

from week1_windows import ROOT, load_tokenizer
from week2_model import log_probs
from week3_infer import load_selected, read_speeches
CONDITIONS = [
    "original",
    "party_masked",
    "heading_removed",
    "first_sentence_removed",
    "last_sentence_removed",
]
RUNS = ["temporal_text_s4011", "temporal_chamber_s4011"]


def score_one(model, bf16, condition, run, tokenizer, device, batch_size, source, out):
    speeches = list(read_speeches(source / f"{condition}.jsonl", tokenizer))
    output = out / f"{run}__{condition}.csv"
    if output.is_file():
        with output.open(encoding="utf-8") as handle:
            rows = sum(1 for line in handle if line.strip())
        if rows == len(speeches) + 1:
            print(f"skip complete {run} {condition}", flush=True)
            return
    started = time.perf_counter()
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["speech_id", "predicted", "p_D", "p_R"])
        for start in range(0, len(speeches), batch_size):
            batch = speeches[start:start + batch_size]
            with torch.inference_mode():
                probs = log_probs(model, batch, device, bf16).exp().cpu()
            if device.type == "mps":
                torch.mps.empty_cache()
            if not torch.isfinite(probs).all():
                raise RuntimeError("non-finite probabilities")
            for speech, (pd, pr) in zip(batch, probs.tolist()):
                writer.writerow([speech.speech_id, "R" if pr > pd else "D", f"{pd:.17g}", f"{pr:.17g}"])
            handle.flush()
            done = min(start + batch_size, len(speeches))
            print(f"{run} {condition}: {done}/{len(speeches)}", flush=True)
    print(f"finished {run} {condition} in {time.perf_counter() - started:.0f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("dev", "test"), default="dev")
    parser.add_argument("--run", action="append", dest="runs")
    parser.add_argument("--condition", action="append", dest="conditions")
    args = parser.parse_args()
    runs = args.runs or RUNS
    conditions = args.conditions or CONDITIONS
    source = ROOT / "results/week3/robustness" / args.phase
    out = source / "scores"
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # Batch 16 is the follow-up timing run. Completed score files are left as they are.
    batch_size = 16 if device.type == "mps" else 32
    print(f"device {device} batch {batch_size}", flush=True)
    for run in runs:
        started = time.perf_counter()
        model, bf16 = load_selected(run, device)
        print(f"loaded {run} in {time.perf_counter() - started:.1f}s", flush=True)
        for condition in conditions:
            score_one(model, bf16, condition, run, tokenizer, device, batch_size, source, out)
        del model
    print("done", flush=True)


if __name__ == "__main__":
    main()
