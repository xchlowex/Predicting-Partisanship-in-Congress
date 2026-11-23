"""Measure Stanford source linkage, using one Congress and edition at a time.

Inputs: hein-bound.zip and hein-daily.zip, or extracted directories with those
names. The eligible denominator is distinct, nonempty speech text with exactly
one text row and one description, a valid calendar date, and chamber H or S.
Titled speakers remain in this denominator. A source member match requires one
map row, a numeric source speaker ID, and matching chamber. A resolved source
party additionally has exactly one party_full lookup for its code and Congress.

These are end-of-session party values. Speaker IDs are local to a session and
chamber. Totals across editions can include duplicate speeches. This audit does
not certify dated labels, identity across sessions, or train/test splits.
"""

import argparse
import csv
import io
import os
import sys
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import date
from functools import lru_cache
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile

from stanford_zip import open_stanford_zip


REPO = Path(__file__).resolve().parents[1]
EDITIONS = {"hein-bound": range(43, 112), "hein-daily": range(97, 115)}


def source_index(root, edition, stack):
    folder, archive = root / edition, root / (edition + ".zip")
    if archive.is_file():
        zipped = stack.enter_context(open_stanford_zip(archive))
        entries = (name for name in zipped.namelist() if not name.endswith("/"))
        opener = zipped.open
    elif folder.is_dir():
        entries = (path.relative_to(folder).as_posix() for path in folder.rglob("*.txt"))
        def opener(name):
            return (folder / name).open("rb")
    else:
        raise FileNotFoundError(f"Missing {archive} or extracted directory {folder}")
    members = defaultdict(list)
    for name in entries:
        path = PurePosixPath(name)
        if "__MACOSX" not in path.parts and not path.name.startswith("."):
            members[path.name.lower()].append(name)
    return members, opener


def session_paths(members, edition, congress):
    n = f"{congress:03}"
    paths = []
    for key in (f"speeches_{n}.txt", f"descr_{n}.txt", f"{n}_speakermap.txt"):
        matches = members.get(key, [])
        if len(matches) != 1:
            raise ValueError(f"{edition}: expected one {key}, found {len(matches)}")
        paths.append(matches[0])
    return paths


def rows(opener, path, fields, counts, kind, speech=False):
    with io.TextIOWrapper(opener(path), encoding="latin-1") as stream:
        header = stream.readline().removeprefix("ï»¿").strip().lower().split("|")
        if not all(field in header for field in fields):
            raise ValueError(f"{path}: missing required column; expected {', '.join(fields)}")
        positions = [header.index(field) for field in fields]
        for line in stream:
            counts[kind + "_rows"] += 1
            # Speech text may itself contain a pipe.
            parts = line.rstrip("\r\n").split("|", 1 if speech else -1)
            if len(parts) != len(header):
                counts[kind + "_malformed_rows"] += 1
                continue
            record = [parts[pos].strip() for pos in positions]
            if not record[0].isascii() or not record[0].isdigit():
                counts[kind + "_invalid_id_rows"] += 1
                continue
            yield record


@lru_cache(maxsize=8192)
def valid_date(value):
    if len(value) != 8 or not value.isascii() or not value.isdigit():
        return False
    try:
        date(int(value[:4]), int(value[4:6]), int(value[6:]))
    except ValueError:
        return False
    return True


def read_lookup(path):
    labels = defaultdict(set)
    with path.open(encoding="latin-1", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="|")
        if not {"session", "party", "party_full"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: invalid party lookup header")
        for row in reader:
            label = row["party_full"].strip().lower()
            if label and label not in {"unknown", "unresolved"}:
                labels[int(row["session"]), row["party"].strip()].add(label)
    return labels


def audit_session(edition, congress, paths, opener, labels):
    counts = Counter()
    text_rows, descr, mapped = {}, {}, {}
    duplicate_text, duplicate_descr, duplicate_map, conflicting_map = set(), set(), set(), set()
    for key, speech in rows(opener, paths[0], ["speech_id", "speech"], counts, "text", True):
        if key in text_rows:
            duplicate_text.add(key)
        text_rows[key] = bool(speech)
    for key, chamber, day in rows(opener, paths[1], ["speech_id", "chamber", "date"], counts, "descr"):
        if key in descr:
            duplicate_descr.add(key)
        descr[key] = (chamber, day)
    for key, speaker, chamber, party in rows(
        opener, paths[2], ["speech_id", "speakerid", "chamber", "party"], counts, "map"
    ):
        target = (speaker, chamber, party)
        if key in mapped:
            duplicate_map.add(key)
            if mapped[key] != target:
                conflicting_map.add(key)
        mapped[key] = target
    counts.update(
        text_distinct_ids=len(text_rows), text_duplicate_ids=len(duplicate_text),
        descr_duplicate_ids=len(duplicate_descr), map_duplicate_ids=len(duplicate_map),
        map_conflicting_targets=len(conflicting_map), text_missing_descr=len(text_rows.keys() - descr.keys()),
        descr_ids_without_text=len(descr.keys() - text_rows.keys()), map_ids_without_text=len(mapped.keys() - text_rows.keys()),
    )
    totals = {k: Counter() for k in ("H", "S")}
    support, members = Counter(), defaultdict(set)
    for key, nonempty in text_rows.items():
        if key in duplicate_text or key in duplicate_descr or key not in descr:
            continue
        chamber, day = descr[key]
        if not nonempty:
            counts["empty_text_exclusions"] += 1
            continue
        if chamber not in totals or not valid_date(day):
            counts["invalid_date_or_nonfloor_chamber_exclusions"] += 1
            continue
        c = totals[chamber]
        c["eligible"] += 1
        if key not in mapped:
            c["missing_map"] += 1
            continue
        if key in duplicate_map:
            c["duplicate_map"] += 1
            continue
        speaker, mapped_chamber, party = mapped[key]
        if mapped_chamber != chamber or not speaker.isascii() or not speaker.isdigit():
            c["invalid_member_id_or_chamber"] += 1
            continue
        c["member_matches"] += 1
        candidates = labels.get((congress, party), set())
        if len(candidates) == 1:
            c["unique_source_party"] += 1
        elif candidates:
            c["ambiguous_party_code"] += 1
        else:
            c["unmapped_party_code"] += 1
        support[chamber, party] += 1
        members[chamber, party].add(speaker)
    stats = {"edition": edition, "congress": congress, **counts}
    for chamber, c in totals.items():
        for field in ("eligible", "member_matches", "unique_source_party", "missing_map",
                      "duplicate_map", "invalid_member_id_or_chamber", "ambiguous_party_code", "unmapped_party_code"):
            stats[f"{chamber}_{field}"] = c[field]
        stats[f"{chamber}_unresolved_source_party"] = c["eligible"] - c["unique_source_party"]
    for kind in ("text", "descr", "map"):
        for field in ("rows", "malformed_rows", "invalid_id_rows"):
            stats.setdefault(f"{kind}_{field}", 0)
    for field in ("empty_text_exclusions", "invalid_date_or_nonfloor_chamber_exclusions"):
        stats.setdefault(field, 0)
    party_rows = []
    for (chamber, party), amount in sorted(support.items()):
        candidates = sorted(labels.get((congress, party), set()))
        party_rows.append(dict(edition=edition, congress=congress, chamber=chamber, source_party_code=party,
                               candidate_count=len(candidates), candidate_names=" | ".join(candidates),
                               source_member_matched_speeches=amount,
                               distinct_session_chamber_speaker_ids=len(members[chamber, party])))
    return stats, party_rows


def write_csv(path, fields, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=REPO.parent / "stanford_raw")
    parser.add_argument("--output", type=Path, help="default: DATA_ROOT/audit (filtered requests get a subdirectory)")
    parser.add_argument("--party-lookup", type=Path, default=REPO / "data/party_full/party_full.txt")
    parser.add_argument("--edition", choices=tuple(EDITIONS), action="append", help="repeat to select editions")
    parser.add_argument("--congress", type=int, nargs="+", help="limit audit to specified Congresses")
    args = parser.parse_args()
    editions = list(dict.fromkeys(args.edition or EDITIONS))
    supported = set().union(*(EDITIONS[e] for e in editions))
    if args.congress and set(args.congress) - supported:
        parser.error("requested Congress outside selected edition coverage")
    stats, parties = [], []
    # Preflight all expected members before opening the large speech streams.
    with ExitStack() as stack:
        worklist = []
        for edition in editions:
            sessions = [n for n in EDITIONS[edition] if not args.congress or n in args.congress]
            if not sessions:
                continue
            index, opener = source_index(args.data_root, edition, stack)
            for n in sessions:
                worklist.append((edition, n, session_paths(index, edition, n), opener))
        labels = read_lookup(args.party_lookup)
        for edition, n, paths, opener in worklist:
            summary, support = audit_session(edition, n, paths, opener, labels)
            stats.append(summary)
            parties.extend(support)
            print(f"Audited {edition} {n}", file=sys.stderr, flush=True)
    target = args.output or args.data_root / "audit"
    if args.output is None and (args.edition or args.congress):
        subset = "_".join(sorted(editions))
        if args.congress:
            subset += "_" + "-".join(str(n) for n in sorted(set(args.congress)))
        target = target / subset
    fields = ["edition", "congress", *sorted(stats[0].keys() - {"edition", "congress"})]
    write_csv(target / "session_linkage.csv", fields, stats)
    write_csv(target / "party_support.csv", ["edition", "congress", "chamber", "source_party_code",
              "candidate_count", "candidate_names", "source_member_matched_speeches",
              "distinct_session_chamber_speaker_ids"], parties)
    print(f"Wrote {len(stats)} edition–Congress rows to {target}. Party linkage is end-of-session only.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, BadZipFile, ValueError, EOFError) as exc:
        sys.exit(f"Audit failed: {exc}")
