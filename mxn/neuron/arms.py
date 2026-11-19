"""Label sources of the measurement arms (design reports/design_ku_neurons.md §3). Arm S, the system, uses only
the cue labels of mxn/neuron/cloze.py; nothing here is on its path.

- L (ruling N4, measurement only): arm S plus text-layout conventions: a life-span parenthesis "(date – date)" gives
  date / place of birth (left side) and of death (right side); "a 1933 American film" gives publication date (the
  year) and country of origin / country (the capitalized words after it); "was an American actor" gives country of
  citizenship.
- R (ruling N2, measurement only): our chapter 2 rules as labeler: the date rules, the life-span place rule and the
  mined pattern triggers (extractor v3), whose values are matched to the KU's own slots.
- O (oracle, measurement only): 2Wiki evidence triples for the KU, matched to its slots.

Each returns {slot index: {relations}} for cloze.examples(extra=...).
"""

from __future__ import annotations

import re
from collections import defaultdict

from mxn.expert.extractor import dates_rule, place_rule
from mxn.expert.patterns import pattern_rule
from mxn.neuron.match import find_slot
from mxn.neuron.signal import RELATIONS

_DASH = re.compile(r"\s*[–—]\s*|\s+-\s+|(?<=\d)-(?=\d)")
_WORK = re.compile(r"\b(?:a|an)\s+((?:1[0-9]{3}|20[0-9]{2}))\s+((?:[A-Z][\w-]*\s+)+)?(?=[a-z])")
_NAT = re.compile(r"\b(?:is|was)\s+(?:a|an)\s+((?:[A-Z][\w-]*\s+)+)(?=[a-z])")


def _slots_in(slots, i: int, a: int, b: int, kinds) -> list[int]:
    return [k for k, s in enumerate(slots) if s.kind in kinds and any(j == i and a <= x and y <= b for j, x, y in s.occ)]


def layout_labels(sentences: list[str], slots) -> dict:
    out = defaultdict(set)
    for i, s in enumerate(sentences):
        for m in re.finditer(r"\(([^()]*)\)", s):
            inner, off = m.group(1), m.start(1)
            parts = _DASH.split(inner, maxsplit=1)
            if len(parts) < 2:
                continue
            cut = off + len(parts[0])
            left, right = (off, cut), (cut, m.end(1))
            if not (_slots_in(slots, i, *left, ("date", "year")) or _slots_in(slots, i, *right, ("date", "year"))):
                continue
            for (a, b), side in ((left, "birth"), (right, "death")):
                for k in _slots_in(slots, i, a, b, ("date", "year")):
                    out[k].add(f"date of {side}")
                for k in _slots_in(slots, i, a, b, ("name",)):
                    out[k].add(f"place of {side}")
            break                                   # the first life-span parenthesis only
        for m in _WORK.finditer(s):
            for k in _slots_in(slots, i, m.start(1), m.end(1), ("year",)):
                out[k].add("publication date")
            if m.group(2):
                for k in _slots_in(slots, i, m.start(2), m.end(2), ("name",)):
                    out[k] |= {"country of origin", "country"}
        for m in _NAT.finditer(s):
            for k in _slots_in(slots, i, m.start(1), m.end(1), ("name",)):
                out[k].add("country of citizenship")
    return dict(out)


def rule_labels(sentences: list[str], slots, triggers: dict) -> dict:
    out = defaultdict(set)
    for r in RELATIONS:
        for v in (dates_rule(r, sentences), place_rule(r, sentences), pattern_rule(r, sentences, triggers)):
            k = find_slot(v, slots)
            if k is not None:
                out[k].add(r)
    return dict(out)


def oracle_labels(gold: list[dict], slots, equiv: dict | None = None) -> dict:
    """gold: this KU's lookups [{relation, object}]."""
    out = defaultdict(set)
    for g in gold:
        k = find_slot(g["object"], slots, equiv)
        if k is not None:
            out[k].add(g["relation"])
    return dict(out)


def load_triggers(path) -> dict:
    import json
    return {r: [tuple(t["trigger"]) for t in ts] for r, ts in json.load(open(path)).items()}
