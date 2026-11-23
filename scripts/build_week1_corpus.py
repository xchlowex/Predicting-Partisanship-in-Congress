"""Build the frozen 110th–114th Congress Stanford daily speech sample.

Only standard-library modules and the project's dated member resolver are used.
The optional stage stores local, intermediate joins outside the Git repository.
"""

import argparse
import csv
import gzip
import hashlib
import heapq
import json
import re
from collections import Counter, defaultdict
from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path

from audit_stanford import rows, session_paths, source_index, valid_date


REPO = Path(__file__).resolve().parents[1]
EDITION = "hein-daily"
CONGRESSES = range(110, 115)
PER_CONGRESS = 4_000
RESERVOIR_SIZE = 6_000  # Per Congress and chamber, leaving room to replace rejected candidates.
SPEAKER_CAP = 30
SEED = "APAI4011-week1-v1"
TOKEN = re.compile(r"[a-z0-9]+")
NON_ALPHA = re.compile(r"[^A-Za-z]+")
MEMBER_FIELDS = ["speakerid", "lastname", "firstname", "chamber", "state", "party", "district",
                 "gender", "nonvoting"]
LABEL = {"Democrat": "D", "Democratic": "D", "Republican": "R", "Independent": "I",
         "Independent Democrat": "ID"}
STAGE_SCHEMA = 1


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, records):
    fields = list(dict.fromkeys(k for row in records for k in row))
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


@lru_cache(maxsize=4096)
def speaker_prefix(speaker):
    return re.compile(r"^" + re.escape(speaker) + r"\s*[:.]\s+", re.IGNORECASE)


def clean(text, speaker):
    text = " ".join(text.split())
    # The source already separates the speaker column. Strip an explicitly
    # repeated source heading, while preserving forms of address, e.g. Mr. Speaker.
    stripped = 0
    if speaker and speaker.lower() != "unknown":
        text, stripped = speaker_prefix(speaker).subn("", text, count=1)
    count = text.count(" ") + bool(text)
    if count < 40:
        return "short", "", count, stripped
    nonspace = len(text) - text.count(" ")
    if len(NON_ALPHA.sub("", text)) * 10 < 7 * nonspace:
        return "low_alpha", "", count, stripped
    return "kept", text, count, stripped


def stage_stamp(data_root):
    with ExitStack() as stack:
        index, _ = source_index(data_root, EDITION, stack)
        # Compact fingerprint of the original three source members per session.
        from stanford_zip import open_stanford_zip
        with open_stanford_zip(data_root / (EDITION + ".zip")) as archive:
            return {"schema": STAGE_SCHEMA, "files": {
                name: [archive.getinfo(name).CRC, archive.getinfo(name).file_size]
                for c in CONGRESSES for name in session_paths(index, EDITION, c)}}


def build_stage(data_root, stage):
    stage.mkdir(parents=True, exist_ok=True)
    stats, roster = defaultdict(Counter), {}
    temp = stage / "joined.jsonl.gz.tmp"
    with ExitStack() as stack, gzip.open(temp, "wt", encoding="utf-8", compresslevel=1) as out:
        index, opener = source_index(data_root, EDITION, stack)
        for congress in CONGRESSES:
            print(f"Reading Stanford daily {congress}", flush=True)
            speech_path, descr_path, map_path = session_paths(index, EDITION, congress)
            source_counts = Counter()
            descriptions, mappings = {}, {}
            for sid, chamber, day, speaker in rows(opener, descr_path,
                    ["speech_id", "chamber", "date", "speaker"], source_counts, "descr"):
                if sid in descriptions:
                    raise ValueError(f"Duplicate description in {congress}: {sid}")
                descriptions[sid] = (chamber, day, speaker)
            for sid, *values in rows(opener, map_path, ["speech_id", *MEMBER_FIELDS], source_counts, "map"):
                if sid in mappings:
                    raise ValueError(f"Duplicate map row in {congress}: {sid}")
                person = dict(zip(MEMBER_FIELDS, values))
                member = person["speakerid"]
                if roster.setdefault(member, person) != person:
                    raise ValueError(f"Conflicting metadata for Stanford ID {member}")
                mappings[sid] = member
            seen = set()
            for sid, text in rows(opener, speech_path, ["speech_id", "speech"], source_counts, "text", True):
                if sid in seen:
                    raise ValueError(f"Duplicate speech in {congress}: {sid}")
                seen.add(sid)
                chamber, day, speaker = descriptions.get(sid, ("", "", ""))
                chamber = chamber if chamber in {"H", "S"} else "other"
                bucket = stats[congress, chamber]
                bucket["raw_text_ids"] += 1
                if chamber == "other" or not valid_date(day) or not text:
                    continue
                bucket["eligible_floor_speeches"] += 1
                member = mappings.get(sid)
                if not member or roster[member]["chamber"] != chamber:
                    bucket["no_unique_source_member"] += 1
                    continue
                bucket["source_member_matched"] += 1
                reason, cleaned, count, stripped = clean(text, speaker)
                bucket["raw_headers_removed"] += stripped
                record = [congress, chamber, f"{day[:4]}-{day[4:6]}-{day[6:]}", sid,
                          member, reason, count, cleaned]
                out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            if any(v for k, v in source_counts.items() if "malformed" in k or "invalid_id" in k):
                raise ValueError(f"Malformed input in Congress {congress}: {source_counts}")
    temp.replace(stage / "joined.jsonl.gz")
    write_json(stage / "sources.json", {"stamp": stage_stamp(data_root), "roster": roster,
               "counts": [[c, ch, dict(v)] for (c, ch), v in sorted(stats.items())]})


def tokens(text):
    return TOKEN.findall(text.lower())


def get_candidates(stage, metadata):
    from week1_identity import load_resolver

    roster = metadata["roster"]
    resolver = load_resolver()
    @lru_cache(maxsize=200_000)
    def resolve(congress, day, mid):
        return resolver.resolve(congress, day, roster[mid])

    stats = defaultdict(Counter, {(c, ch): Counter(s) for c, ch, s in metadata["counts"]})
    pools = defaultdict(list)
    content_hashes = set()
    identity_status, verified_members = Counter(), defaultdict(set)
    with gzip.open(stage / "joined.jsonl.gz", "rt", encoding="utf-8") as records:
        for line in records:
            c, ch, day, sid, mid, reason, length, text = json.loads(line)
            stat = stats[c, ch]
            bio, party, status = resolve(c, day, mid)
            identity_status[c, ch, status] += 1
            if status != "verified":
                stat["dated_unresolved_or_excluded_after_source_match"] += 1
                continue
            if not bio or not party:
                raise ValueError(f"Incomplete verified identity for {sid}")
            stat["dated_verified_all_parties"] += 1
            label = LABEL.get(party)
            if label is None:
                stat["dated_other_party_excluded"] += 1
                continue
            verified_members[c, ch, label].add(bio)
            stat["dated_" + label] += 1
            if reason != "kept":
                stat[reason + "_excluded"] += 1
                continue
            stat["quality_eligible"] += 1
            wordseq = tokens(text)
            fingerprint = hashlib.sha256(" ".join(wordseq).encode("ascii")).digest()
            if fingerprint in content_hashes:
                stat["normalized_exact_duplicates"] += 1
                continue
            content_hashes.add(fingerprint)
            stat["exact_unique_eligible"] += 1
            priority = int.from_bytes(hashlib.sha256(f"{SEED}|{EDITION}:{sid}".encode()).digest(), "big")
            row = {"speech_id": f"{EDITION}:{sid}", "source_speech_id": sid, "source": EDITION,
                   "canonical_speaker_id": bio, "source_speaker_id": mid, "date": day, "year": int(day[:4]),
                   "congress": c, "chamber": ch, "label": label, "text": text, "word_count": length}
            pool = pools[c, ch]
            item = (-priority, sid, row)
            if len(pool) < RESERVOIR_SIZE:
                heapq.heappush(pool, item)
            elif item[0] > pool[0][0]:
                heapq.heapreplace(pool, item)
    for (c, ch, label), people in verified_members.items():
        stats[c, ch]["dated_distinct_speakers_" + label] = len(people)
    return pools, stats, identity_status


class NearDuplicateIndex:
    """Exact 5-token shingle Jaccard search over accepted speeches at J >= 0.90."""

    def __init__(self, pools):
        frequency = Counter()
        for pool in pools.values():
            for _, _, row in pool:
                frequency.update(tokens(row["text"]))
        self.word_id = {word: i for i, word in enumerate(sorted(frequency, key=lambda w: (frequency[w], w)))}
        self.bits = max(1, len(self.word_id).bit_length())
        self.mask = (1 << (self.bits * 5)) - 1
        self.prefix_index = defaultdict(list)
        self.accepted = []

    def find_or_add(self, row):
        # Losslessly pack 5 word IDs. The global ascending ordering puts rare
        # initial words first; any global order is valid for the prefix bound.
        shingles = set()
        rolling = 0
        for i, word in enumerate(tokens(row["text"])):
            rolling = ((rolling << self.bits) | self.word_id[word]) & self.mask
            if i >= 4:
                shingles.add(rolling)
        size = len(shingles)
        if not size:
            raise ValueError("Eligible speech has fewer than five tokens")
        # J >= .90 implies overlap >= ceil(.90 * size) for BOTH sets.
        # Their first shared shingle must therefore be in both prefixes.
        prefix = heapq.nsmallest(size - (9 * size + 9) // 10 + 1, shingles)
        candidates = {idx for shingle in prefix for idx in self.prefix_index.get(shingle, ())}
        for idx in sorted(candidates):
            previous, sid = self.accepted[idx]
            n = len(previous)
            if 10 * min(n, size) < 9 * max(n, size):
                continue
            overlap = len(previous & shingles)
            union = n + size - overlap
            if 10 * overlap >= 9 * union:
                return sid, overlap / union
        idx = len(self.accepted)
        self.accepted.append((shingles, row["speech_id"]))
        for shingle in prefix:
            self.prefix_index[shingle].append(idx)
        return None


def select(pools, stats):
    quotas = {}
    for c in CONGRESSES:
        h, s = (stats[c, ch]["exact_unique_eligible"] for ch in ("H", "S"))
        if h + s < PER_CONGRESS:
            raise ValueError(f"Too few eligible speeches for Congress {c}")
        quotas[c, "H"] = (2 * PER_CONGRESS * h + h + s) // (2 * (h + s))
        quotas[c, "S"] = PER_CONGRESS - quotas[c, "H"]
    near = NearDuplicateIndex(pools)
    order = sorted(((-p, sid, row) for pool in pools.values() for p, sid, row in pool))
    speaker_counts, accepted_counts = Counter(), Counter()
    corpus, duplicate_records = [], []
    for _, _, row in order:
        c, ch = row["congress"], row["chamber"]
        if accepted_counts[c, ch] >= quotas[c, ch]:
            continue
        member = (c, row["canonical_speaker_id"])
        if speaker_counts[member] >= SPEAKER_CAP:
            stats[c, ch]["priority_speaker_cap_skips"] += 1
            continue
        duplicate = near.find_or_add(row)
        if duplicate:
            previous, similarity = duplicate
            stats[c, ch]["priority_near_duplicate_skips"] += 1
            duplicate_records.append({"excluded": row["speech_id"], "representative": previous,
                                      "shingle_jaccard": round(similarity, 6)})
            continue
        corpus.append(row)
        accepted_counts[c, ch] += 1
        speaker_counts[member] += 1
    if accepted_counts != quotas:
        raise ValueError(f"Could not fill predeclared quotas {quotas}, achieved {accepted_counts}")
    corpus.sort(key=lambda row: (row["congress"], int(row["source_speech_id"])))
    for key, quota in quotas.items():
        stats[key]["sampled"] = quota
    for row in corpus:
        stats[row["congress"], row["chamber"]]["sampled_" + row["label"]] += 1
    return corpus, duplicate_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO.parent / "stanford_raw")
    parser.add_argument("--output", type=Path, default=REPO / "data/week1")
    stage_mode = parser.add_mutually_exclusive_group()
    stage_mode.add_argument("--stage-only", action="store_true")
    stage_mode.add_argument("--reuse-stage", action="store_true")
    args = parser.parse_args()
    stage = args.data_root / "week1_build"
    if not args.reuse_stage:
        build_stage(args.data_root, stage)
    if args.stage_only:
        return
    metadata = json.loads((stage / "sources.json").read_text(encoding="utf-8"))
    if metadata["stamp"] != stage_stamp(args.data_root):
        raise ValueError("Source archive or stage format changed. Re-run without --reuse-stage.")
    print("Resolving dated identities and filtering for the sample", flush=True)
    pools, stats, status = get_candidates(stage, metadata)
    print("Checking near duplicates and selecting the frozen sample", flush=True)
    corpus, duplicates = select(pools, stats)
    args.output.mkdir(parents=True, exist_ok=True)
    temporary = args.output / "corpus.jsonl.tmp"
    with temporary.open("w", encoding="utf-8", newline="\n") as out:
        for row in corpus:
            out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(args.output / "corpus.jsonl")
    count_fields = sorted({field for s in stats.values() for field in s})
    write_csv(args.output / "preprocessing_counts.csv", [
        {"congress": c, "chamber": ch, **{f: s[f] for f in count_fields}} for (c, ch), s in sorted(stats.items())])
    write_csv(args.output / "identity_status_counts.csv", [
        {"congress": c, "chamber": ch, "resolution_status": k, "speeches": v}
        for (c, ch, k), v in sorted(status.items())])
    write_json(args.output / "near_duplicates_removed.json", duplicates)
    print(f"Wrote {len(corpus):,} speeches to {args.output}", flush=True)


if __name__ == "__main__":
    main()
