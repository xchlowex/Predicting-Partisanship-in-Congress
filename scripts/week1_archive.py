"""Restore the frozen Week 1 JSONL from one gzip file per Congress, or pack it."""

import argparse
import gzip
import hashlib
import json
from itertools import groupby
from pathlib import Path
from tempfile import TemporaryDirectory


DATA = Path(__file__).resolve().parents[1] / "data" / "week1"
CONGRESSES = tuple(range(110, 115))


def shard_name(congress):
    return f"congress-{congress}.jsonl.gz"


def expected_digest(root):
    summary = json.loads((root / "split_summary.json").read_text(encoding="utf-8"))
    return summary["corpus_sha256"]


def require_digest(actual, expected):
    if actual != expected:
        raise ValueError(f"Corpus SHA-256 mismatch: {actual} (expected {expected})")


def pack(root):
    corpus = root / "corpus.jsonl"
    directory = root / "by_congress"
    expected = expected_digest(root)
    digest = hashlib.sha256()
    with TemporaryDirectory(dir=root) as temp, corpus.open("rb") as source:
        staged = Path(temp)
        seen = []
        for congress, lines in groupby(source, key=lambda line: json.loads(line)["congress"]):
            if congress not in CONGRESSES or congress in seen:
                raise ValueError(f"Unexpected or out-of-order Congress {congress}")
            seen.append(congress)
            with (staged / shard_name(congress)).open("wb") as file:
                with gzip.GzipFile(filename="", fileobj=file, mode="wb", mtime=0, compresslevel=6) as out:
                    for line in lines:
                        digest.update(line)
                        out.write(line)
        if tuple(seen) != CONGRESSES:
            raise ValueError(f"Expected Congresses {CONGRESSES}, got {seen}")
        require_digest(digest.hexdigest(), expected)
        directory.mkdir(exist_ok=True)
        for congress in CONGRESSES:
            name = shard_name(congress)
            (staged / name).replace(directory / name)


def restore(root):
    expected = expected_digest(root)
    corpus = root / "corpus.jsonl"
    if corpus.exists():
        with corpus.open("rb") as source:
            require_digest(hashlib.file_digest(source, "sha256").hexdigest(), expected)
        return corpus
    digest = hashlib.sha256()
    with TemporaryDirectory(dir=root) as temp:
        staged = Path(temp) / corpus.name
        with staged.open("wb") as out:
            for congress in CONGRESSES:
                path = root / "by_congress" / shard_name(congress)
                count = 0
                with gzip.open(path, "rb") as source:
                    for line in source:
                        if json.loads(line)["congress"] != congress:
                            raise ValueError(f"Speech assigned to the wrong Congress in {path.name}")
                        out.write(line)
                        digest.update(line)
                        count += 1
                if count == 0:
                    raise ValueError(f"Empty shard: {path}")
        require_digest(digest.hexdigest(), expected)
        staged.replace(corpus)
    return corpus


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("restore", "pack"), default="restore")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    args = parser.parse_args()
    if args.action == "pack":
        pack(args.data_dir)
        print(f"Packed and verified five Congresses in {args.data_dir / 'by_congress'}")
    else:
        print(f"Verified {restore(args.data_dir)}")
