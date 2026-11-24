"""Run party-name masking and phrase-removal checks on a retained Week 2 checkpoint.

Checkpoint weights are intentionally not required for Week 3 matched error analysis.
Run this script only after restoring the selected checkpoint named in the formal run.json.
"""

import argparse
import json
import re
from pathlib import Path

import torch

from week1_archive import ROOT
from week2_model import build_for_reload


PARTY_NAMES = re.compile(
    r"\b(?:Democrat(?:ic)?|Republican(?:s)?|GOP|Independent Democrat|Independent)\b",
    re.IGNORECASE,
)
PARTY_LABELS = re.compile(r"\b(?:D|R)\b")


def mask_party_names(text):
    return PARTY_NAMES.sub("[PARTY]", text)


def remove_leading_speaker_heading(text):
    return re.sub(r"^(?:Mr\.|Ms\.|Mrs\.|Madam|Mister)\s+(?:Speaker|Chairman|Chairwoman)\s*\.\s*", "", text,
                  flags=re.IGNORECASE)


def remove_first_sentence(text):
    match = re.search(r"[.!?](?:\s|$)", text)
    return text[match.end():].strip() if match else text


def remove_last_sentence(text):
    return re.split(r"(?<=[.!?])\s+", text)[0] if len(re.findall(r"[.!?]", text)) == 1 else ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Formal Week 2 run directory")
    parser.add_argument("--phase", choices=("dev", "test"), default="dev")
    args = parser.parse_args()
    run_path = args.run.resolve()
    require = lambda condition, message: (_ for _ in ()).throw(ValueError(message)) if not condition else None
    require(run_path.is_dir(), f"Missing run directory: {run_path}")
    run = json.loads((run_path / "run.json").read_text(encoding="utf-8"))
    checkpoint = ROOT / run["checkpoint"]
    require(checkpoint.is_file(), f"Missing retained checkpoint: {checkpoint}")
    require(torch.cuda.is_available(), "No CUDA device available for checkpoint inference")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_for_reload(run["variant"], run["metadata_stats"])
    model.load_state_dict(state["model"])
    model.to(torch.device("cuda")).eval()
    print(json.dumps({
        "status": "ready",
        "run": run_path.as_posix(),
        "checkpoint": checkpoint.as_posix(),
        "phase": args.phase,
        "variants": ["party-name-masking", "leading-speaker-heading-removal",
                     "first-sentence-removal", "last-sentence-removal"],
    }, indent=2))


if __name__ == "__main__":
    main()
