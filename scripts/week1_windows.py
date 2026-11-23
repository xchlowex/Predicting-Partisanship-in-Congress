"""Build the per-speech RoBERTa-base token-window manifest without loading model weights."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "data/reference/roberta-base/tokenizer.json"
TOKENIZER_SHA256 = "847bbeab6174d66a88898f729d52fa8d355fafe1bea101cf960dd404581df70e"
MAX_CONTENT = 254


def load_tokenizer(path=TOKENIZER):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != TOKENIZER_SHA256:
        raise ValueError("Tokenizer does not match the frozen RoBERTa-base tokenizer.")
    return Tokenizer.from_str(data.decode("utf-8"))


def window_ranges(n_tokens):
    """Non-overlapping [start, stop) spans of the full speech's BPE tokens."""
    if n_tokens <= 0:
        raise ValueError("A speech must have at least one content token.")
    if n_tokens <= MAX_CONTENT:
        return [(0, n_tokens)]
    if n_tokens <= 2 * MAX_CONTENT:
        midpoint = (n_tokens + 1) // 2
        return [(0, midpoint), (midpoint, n_tokens)]
    return [(0, MAX_CONTENT), (n_tokens - MAX_CONTENT, n_tokens)]


def encode_windows(text, tokenizer):
    """Return model input IDs for a speech, with BOS and EOS on each window."""
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    return [[0, *ids[start:stop], 2] for start, stop in window_ranges(len(ids))]


def write_manifest(corpus, output):
    tokenizer = load_tokenizer()
    output.parent.mkdir(parents=True, exist_ok=True)
    speech_ids = set()
    window_count = truncated = token_total = retained_total = 0
    with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=output.parent,
                            delete=False) as dst:
        temp_path = Path(dst.name)
    try:
        with corpus.open(encoding="utf-8") as src, temp_path.open("w", newline="", encoding="utf-8") as dst:
            writer = csv.writer(dst, lineterminator="\n")
            writer.writerow(["speech_id", "content_tokens", "window_1_start", "window_1_stop",
                             "window_2_start", "window_2_stop"])
            for line in src:
                speech = json.loads(line)
                key = speech["speech_id"]
                if key in speech_ids:
                    raise ValueError(f"Duplicate speech_id: {key}")
                speech_ids.add(key)
                n = len(tokenizer.encode(speech["text"], add_special_tokens=False).ids)
                ranges = window_ranges(n)
                second = ranges[1] if len(ranges) == 2 else ("", "")
                writer.writerow([key, n, *ranges[0], *second])
                window_count += len(ranges)
                truncated += n > 2 * MAX_CONTENT
                token_total += n
                retained_total += sum(stop - start for start, stop in ranges)
        if not speech_ids:
            raise ValueError("Corpus is empty.")
        temp_path.replace(output)
    finally:
        temp_path.unlink(missing_ok=True)
    print(f"{len(speech_ids):,} speeches, {window_count:,} windows, "
          f"{truncated:,} speeches with omitted middle tokens; "
          f"{retained_total:,}/{token_total:,} content tokens retained.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/week1/corpus.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/week1/windows.csv")
    args = parser.parse_args()
    write_manifest(args.corpus, args.output)
