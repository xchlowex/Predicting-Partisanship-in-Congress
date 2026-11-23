"""Rebuild the 110th–114th Congress reference from pinned source snapshots.

Run with the project Python (PyYAML) and a local cache directory. Missing inputs
are fetched from the pinned GitHub revisions. The published CSV suffices offline.
"""
import argparse
import csv
import hashlib
import re
from pathlib import Path
from urllib.request import urlopen

import yaml


ROOT = Path(__file__).resolve().parents[1]
ROSTER_REV = "e6dc9b4adcd34af74aa9199bb98b2f43787d0b74"
SNAPSHOTS = {
    "latest": "2b1fe728159fe5f5e145277794106e800719eb50",
    "2016": "0a4bdd15cf9dbae4bca1be114a81499daa762be7",
    "2014": "b57cc0ccfe90015b6252ab8b78e0c689e6f8755a",
    "2012": "9a5da322b1ee0d2f7fba282559c4d1aaf0def715",
}
SOURCE_BLOBS = {
    "current-latest.yaml": "ce5518ea8ccb3bd9a2d607cdcc23e25a85ffc348",
    "current-2016.yaml": "d9d591a736abadc273be4a207de04786e2bcb17d",
    "current-2014.yaml": "9f7c36574e4164813dcd560f6f055cbce33dc267",
    "current-2012.yaml": "434c8c7f2638845359f5c2e5882cda8c9fbb09f4",
    "House-110.csv": "4e651c17d7aaa093056b80be393957ab53652353",
    "House-111.csv": "54a67b0888f9dab06ca5eabe67f8a8bef6dda757",
    "House-112.csv": "73d8086dd63d5947d9bd803a8420f5fa48af1af2",
    "House-113.csv": "26a466de2cad21e4ef7bdcc514e07f4e81db72e5",
    "House-114.csv": "e5aec51a477c8a149fa2a8e104b7a3850a0d50c9",
    "Senate-110.csv": "676e2b7b896f1688e363a7f66813736e28e285b1",
    "Senate-111.csv": "a94e820b47bfc185ecb5c74885d539d8b2306d32",
    "Senate-112.csv": "bb68a199a5e68a263eeb062ab98584e066f270eb",
    "Senate-113.csv": "e52ef914d44a30e8c0478aed3f72d4398d89fcba",
    "Senate-114.csv": "d938f7461ee24c62adf415530b6462bfb969ed36",
}


def input_path(cache, name, github_path):
    path = cache / name
    if path.exists():
        data = path.read_bytes()
    else:
        with urlopen("https://raw.githubusercontent.com/" + github_path, timeout=90) as response:
            data = response.read()
    actual = hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()
    if actual != SOURCE_BLOBS[name]:
        raise ValueError(f"Unexpected Git blob for {name}: {actual}")
    if not path.exists():
        path.write_bytes(data)
    return path


def reference_periods(record, congress, chamber, snapshots):
    year = 2007 + 2 * (congress - 110)
    begin = max(record["start_date"] or "0000", f"{year}-01-03")
    end = min(record["end_date"] or "9999", f"{year + 2}-01-03")
    # Conservatively exclude the dated departure day where sources give no time.
    bioguide = re.search(r"/([A-Z][0-9]{6})\.jpg$", record["image"]).group(1)
    if chamber == "S":
        state, district = record["area_id"].rsplit(":", 1)[-1].upper(), ""
    else:
        state, district = record["area_id"].split("-")
    surname, _, given = record["sort_name"].partition(",")
    surnames = {surname.strip()}
    given_names = {given.strip(), record["name"].split()[0].strip('",')}
    base = dict(congress=congress, bioguide=bioguide, chamber=chamber, state=state, district=district)
    # Pick the newest available snapshot with a term in this session. A member
    # retired before the first snapshot can still appear in the session roster.
    term_type = "sen" if chamber == "S" else "rep"
    terms = []
    source = "EveryPolitician"
    for snapshot, people in snapshots.items():
        person = people.get(bioguide)
        if person is None:
            continue
        names = [person["name"], *person.get("other_names", [])]
        for name in names:
            surnames.add(name.get("last", ""))
            given_names.update([name.get("first", ""), name.get("nickname", ""), name.get("middle", "")])
        terms = [t for t in person["terms"] if t["type"] == term_type and t["state"] == state
                 and (chamber == "S" or str(t.get("district", "")) == district)
                 and str(t["start"]) < end and begin < str(t["end"])]
        if terms:
            source = f"EveryPolitician+congress-legislators-{snapshot}"
            break
    base.update(last_names="|".join(sorted(surnames - {""})), given_names="|".join(sorted(given_names - {""})))
    intervals = []
    for term in terms:
        for period in term.get("party_affiliations", [term]):
            left = max(begin, str(term["start"]), str(period["start"]))
            right = min(end, str(term["end"]), str(period["end"]))
            if left < right:
                intervals.append((left, right, period.get("party", ""), source))
    if not terms:
        intervals = [(begin, end, record["group"], source)]
    for left, right, party, evidence in intervals:
        periods = [(left, right, party)]
        if bioguide == "S000709":
            periods = [(left, min(right, "2009-04-28"), "Republican"),
                       (max(left, "2009-04-28"), min(right, "2009-04-30"), "Unresolved"),
                       (max(left, "2009-04-30"), right, "Democrat")]
            evidence += "+Senate-switch-table"
        elif bioguide == "G000557":
            periods = [(left, min(right, "2009-12-22"), "Democrat"),
                       (max(left, "2009-12-22"), right, "Republican")]
            evidence += "+House-biography"
        elif bioguide == "L000304":
            periods = [(left, right, "Independent Democrat")]
            evidence += "+Senate-switch-table"
        for start, stop, label in periods:
            if start < stop:
                yield dict(**base, start=start, end_exclusive=stop, party=label,
                           roster_party=record["group"], evidence=evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT.parent / "stanford_raw/roster_inputs")
    parser.add_argument("--out", type=Path, default=ROOT / "data/reference/member_party_110_114.csv")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    snapshots = {}
    for year, commit in SNAPSHOTS.items():
        path = input_path(args.cache, f"current-{year}.yaml",
                          f"unitedstates/congress-legislators/{commit}/legislators-current.yaml")
        with path.open(encoding="utf-8") as src:
            people = yaml.load(src, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
        snapshots[year] = {person["id"]["bioguide"]: person for person in people}
    rows = []
    for congress in range(110, 115):
        for name, chamber in (("House", "H"), ("Senate", "S")):
            path = input_path(args.cache, f"{name}-{congress}.csv", f"everypolitician/everypolitician-data/"
                              f"{ROSTER_REV}/data/United_States_of_America/{name}/term-{congress}.csv")
            with path.open(encoding="utf-8", newline="") as src:
                for member in csv.DictReader(src):
                    rows.extend(reference_periods(member, congress, chamber, snapshots))
    rows.sort(key=lambda r: (r["congress"], r["chamber"], r["state"], r["bioguide"], r["start"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows):,} member/party intervals to {args.out}")


if __name__ == "__main__":
    main()
