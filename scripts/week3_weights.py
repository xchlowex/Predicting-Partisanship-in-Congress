"""Install verified Week 2 checkpoints from the team release or a local copy."""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from verify_week1 import ROOT, digest


REPOSITORY = "xchlowex/Predicting-Partisanship-in-Congress"
RELEASE = "week2-formal-weights-v1"


def verify(path, expected):
    if digest(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")


def install(manifest, source):
    record = json.loads(manifest.read_text(encoding="utf-8"))
    name = manifest.parent.name
    checkpoint = ROOT / record["checkpoint"]
    run_dir = ROOT / "checkpoints/week2_formal" / name
    if checkpoint.parent.resolve() != run_dir.resolve() or checkpoint.name != f"epoch{record['best_epoch']}.pt":
        raise ValueError(f"Unexpected checkpoint path in {manifest}")
    expected = record["checkpoint_sha256"]
    if checkpoint.exists():
        verify(checkpoint, expected)
        print(f"Already verified: {name}")
        return

    asset = f"{name}.pt"
    run_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=run_dir) as tmp:
        staged = Path(tmp) / asset
        if source is None:
            subprocess.run(
                ["gh", "release", "download", RELEASE, "--repo", REPOSITORY,
                 "--pattern", asset, "--dir", tmp],
                check=True,
            )
        else:
            shutil.copyfile(source / asset, staged)
        verify(staged, expected)
        # Both operations publish the complete file and fail if a concurrent installer won.
        try:
            if os.name == "nt":
                os.rename(staged, checkpoint)
            else:
                os.link(staged, checkpoint)
        except FileExistsError:
            verify(checkpoint, expected)
    print(f"Verified: {checkpoint.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", metavar="RUN", help="Formal run folder name, e.g. temporal_text_s4011")
    parser.add_argument("--all", action="store_true", help="Install all published formal runs")
    parser.add_argument("--from-dir", type=Path, help="Read <run>.pt assets from this folder instead of GitHub")
    args = parser.parse_args()
    if bool(args.runs) == args.all:
        parser.error("Specify one or more run names, or use --all.")
    manifests = {p.parent.name: p for p in sorted((ROOT / "results/week2/formal").glob("*/run.json"))}
    if not manifests:
        parser.error("No formal run manifests found in results/week2/formal.")
    selected = list(manifests) if args.all else list(dict.fromkeys(args.runs))
    unknown = [name for name in selected if name not in manifests]
    if unknown:
        parser.error(f"Unknown formal runs: {', '.join(unknown)}")
    for name in selected:
        try:
            install(manifests[name], args.from_dir)
        except (OSError, KeyError, ValueError, subprocess.CalledProcessError) as error:
            parser.exit(1, f"{name}: {error}\n")


if __name__ == "__main__":
    main()
