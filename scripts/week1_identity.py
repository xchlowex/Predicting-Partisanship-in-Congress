"""Resolve a Stanford member row to a public BioGuide ID and dated affiliation."""
import csv
import re
import unicodedata
from collections import defaultdict
from functools import lru_cache
from pathlib import Path


REFERENCE = Path(__file__).resolve().parents[1] / "data/reference/member_party_110_114.csv"
SWITCH_CHECKED = {"S000709", "G000557", "L000304"}


@lru_cache(maxsize=18000)
def normalized(value):
    text = unicodedata.normalize("NFKD", value.casefold()).encode("ascii", "ignore").decode()
    text = re.sub(r"[,\s]+(?:jr|sr|ii|iii|iv)\.?$", "", text)
    return "".join(re.findall(r"[a-z]+", text))


def first_tokens(value):
    # Retain both spellings for e.g. "Mary Ann" and the single-token given name.
    return {normalized(value), normalized(value.split()[0])} if value.strip() else set()


class MemberResolver:
    def __init__(self, reference):
        self.index = defaultdict(list)
        with Path(reference).open(encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source):
                row["given"] = set().union(*(first_tokens(x) for x in row["given_names"].split("|")))
                for surname in {normalized(x) for x in row["last_names"].split("|")}:
                    key = (int(row["congress"]), row["chamber"], row["state"], row["district"], surname)
                    self.index[key].append(row)

    @lru_cache(maxsize=12000)
    def candidates(self, congress, chamber, state, district, last, first):
        within_seat = self.index.get((congress, chamber, state, district, normalized(last)), ())
        if not within_seat:
            return (), "no_roster_surname_seat"
        first = normalized(first)
        if not first or first == "unknown":
            return (), "missing_given_name"
        matches = [r for r in within_seat if first in r["given"]]
        # An initial alone is accepted only for a uniquely identified surname
        # in the same state, chamber, district and Congress.
        if not matches and len({r["bioguide"] for r in within_seat}) == 1:
            matches = [r for r in within_seat if any(
                first[0] == g[:1] and (len(first) == 1 or len(g) == 1) for g in r["given"])]
        if not matches:
            return (), "given_name_disagreement"
        return matches, ""

    def resolve(self, congress, day, member):
        if member.get("nonvoting", "").strip().lower() == "nonvoting":
            return "", "", "nonvoting_delegate"
        chamber = member["chamber"]
        district = member["district"].strip() if chamber == "H" else ""
        candidates, reason = self.candidates(congress, chamber, member["state"], district,
                                             member["lastname"], member["firstname"])
        if reason:
            return "", "", reason
        dated = {(r["bioguide"], r["party"], r["roster_party"]) for r in candidates
                 if r["start"] <= day < r["end_exclusive"]}
        if not dated:
            return "", "", "outside_verified_service_dates"
        if len(dated) > 1:
            return "", "", "ambiguous_roster_match"
        bioguide, party, roster_party = dated.pop()
        if party == "Unresolved":
            return "", "", "switch_date_uncertain"
        if party not in {"Democrat", "Republican", "Independent", "Independent Democrat"}:
            return "", "", "unresolved_party"
        if bioguide not in SWITCH_CHECKED and (party != roster_party or member["party"] != party[:1]):
            return "", "", "party_sources_disagree"
        return bioguide, party, "verified"


def load_resolver(reference=REFERENCE):
    return MemberResolver(reference)
