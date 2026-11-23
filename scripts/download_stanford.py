"""Download and validate the Stanford archives named in the project README."""

import argparse
from collections import Counter
from contextlib import contextmanager
import http.client
import os
from pathlib import Path, PurePosixPath
import re
import sys
import time
from urllib.request import Request, urlopen
from zipfile import BadZipFile, is_zipfile
import zlib

from stanford_zip import open_stanford_zip


BASE_URL = "https://stacks.stanford.edu/file/druid%3Amd374tz9962/"
DEFAULT_DIR = Path(__file__).resolve().parents[2] / "stanford_raw"
ARCHIVES = ("vocabulary", "hein-daily", "hein-bound")
CHUNK = 2**20


def expected_files(name):
    if name == "vocabulary":
        return {"master_list.txt", "procedural.txt", "vocab.txt"}
    sessions = range(43, 112) if name == "hein-bound" else range(97, 115)
    return {
        filename
        for session in sessions
        for filename in (
            f"speeches_{session:03}.txt", f"descr_{session:03}.txt",
            f"{session:03}_speakermap.txt", f"byparty_2gram_{session:03}.txt",
            f"byspeaker_2gram_{session:03}.txt",
        )
    }


def verify(path, name):
    expected = expected_files(name)
    with open_stanford_zip(path) as archive:
        members = Counter()
        for item in archive.infolist():
            member = PurePosixPath(item.filename)
            if not item.is_dir() and item.file_size and "__MACOSX" not in member.parts:
                members[member.name.lower()] += 1
        invalid = sorted(key for key in expected if members[key] != 1)
        if invalid:
            raise ValueError("Missing, empty or duplicated expected files: " + ", ".join(invalid[:5]))
        print(f"{name}: checking CRCs for the entire ZIP...", flush=True)
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"CRC failure in {bad}")
    print(f"{name}: verified {len(expected)} required files and ZIP integrity.", flush=True)


@contextmanager
def download_lock(directory):
    # The OS releases the byte-range lock even if the process is interrupted.
    with (directory / ".stanford-download.lock").open("a+b") as handle:
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise OSError(f"Cannot lock {directory} for downloading: {error}") from error
        yield


def transfer(url, path):
    offset = path.stat().st_size if path.exists() else 0
    headers = {"Accept-Encoding": "identity", "User-Agent": "APAI4011-dataset-download/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    with urlopen(Request(url, headers=headers), timeout=90) as response:
        if response.status == 206:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if not match or int(match[1]) != offset or int(match[2]) != int(match[3]) - 1:
                raise ValueError("Unexpected HTTP byte range; partial file retained.")
            total = int(match[3])
        elif response.status == 200:
            size = response.headers.get("Content-Length")
            total = int(size) if size else None
            if offset:
                # Some servers ignore Range. Compare their complete response with our
                # existing prefix before appending, preserving both data and progress.
                with path.open("rb") as existing:
                    while prefix := existing.read(CHUNK):
                        if response.read(len(prefix)) != prefix:
                            raise ValueError("Remote data differs from the partial file; move it aside to restart.")
        else:
            raise ValueError(f"Unexpected HTTP status {response.status}")
        if not offset and response.read(4) != b"PK\x03\x04":
            raise ValueError("Server returned a non-ZIP response.")
        received = offset
        last_update = time.monotonic()
        with path.open("ab") as dest:
            if not offset:
                dest.write(b"PK\x03\x04")
                received = 4
            while block := response.read(CHUNK):
                dest.write(block)
                received += len(block)
                if time.monotonic() - last_update >= 15:
                    print(f"  {received / 2**20:,.0f} MiB received", flush=True)
                    last_update = time.monotonic()
        if total is not None and received != total:
            raise OSError(f"Expected {total} bytes, received {received}; partial file retained.")


def prepare(directory, name, check_only):
    final = directory / f"{name}.zip"
    if final.is_file() or check_only:
        verify(final, name)
        return
    partial = final.with_suffix(".zip.part")
    if not partial.is_file() or not is_zipfile(partial):
        print(f"{name}: downloading from Stanford to {partial}", flush=True)
        transfer(BASE_URL + final.name, partial)
    verify(partial, name)
    partial.rename(final)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--check-only", action="store_true", help="Check all three local archives without downloading.")
    args = parser.parse_args()
    errors = (OSError, ValueError, BadZipFile, zlib.error, EOFError, http.client.HTTPException)
    if args.check_only:
        valid = True
        for name in ARCHIVES:
            try:
                prepare(args.data_dir, name, True)
            except errors as error:
                print(f"{name}: {error}", file=sys.stderr)
                valid = False
        return 0 if valid else 1
    try:
        args.data_dir.mkdir(parents=True, exist_ok=True)
        with download_lock(args.data_dir):
            for name in ARCHIVES:
                prepare(args.data_dir, name, False)
    except errors as error:
        print(f"Download/validation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
