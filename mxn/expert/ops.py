"""Deterministic operations and output templates (step 5 of 2026-10-24b). No learning, no generation.

A plan (mxn/expert/planner.py form) is executed over values that FIND returns (lists of strings, one
list per variable; a FIND may return several values, e.g. two citizenships). COMPARE:
  earlier / later     parsed dates (year, month, day; missing parts = 0); the first value of each side
  larger / smaller    parsed numbers; the first value of each side
  same / different    normalized string sets; same = any shared value, different = none shared
Templates: chain → the value text; earlier/later/larger/smaller → the winning side's entity (its KU
title); same/different → "yes"/"no". A side without a parsable value, or a tie → None (abstain).
"""

from __future__ import annotations

import re
import string

MONTHS = {m: i + 1 for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                          "september", "october", "november", "december"])}


def normalize(s: str) -> str:
    """The official 2Wiki answer normalization."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def parse_date(s: str):
    """(year, month, day) from "13 June 1946", "June 13, 1946", "June 1946", "1946"; else None."""
    s = s.strip().lower().replace(",", "").rstrip(".)")
    m = re.fullmatch(r"(\d{1,2}) ([a-z]+) (\d{1,4})", s)
    if m and m.group(2) in MONTHS:
        return (int(m.group(3)), MONTHS[m.group(2)], int(m.group(1)))
    m = re.fullmatch(r"([a-z]+) (\d{1,2}) (\d{1,4})", s)
    if m and m.group(1) in MONTHS:
        return (int(m.group(3)), MONTHS[m.group(1)], int(m.group(2)))
    m = re.fullmatch(r"([a-z]+) (\d{1,4})", s)
    if m and m.group(1) in MONTHS:
        return (int(m.group(2)), MONTHS[m.group(1)], 0)
    m = re.fullmatch(r"(\d{1,4})", s)
    return (int(m.group(1)), 0, 0) if m else None


def parse_number(s: str):
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", s)
    return float(m.group().replace(",", "")) if m else None


def compare(a: list[str], b: list[str], op: str, equiv: dict | None = None):
    """0 if side a wins, 1 if side b wins (ordering ops); True/False (same/different); None = abstain."""
    if op in ("same", "different"):
        if not a or not b:
            return None
        key = lambda x: (equiv or {}).get(normalize(x), normalize(x))
        shared = bool({key(x) for x in a} & {key(x) for x in b})
        return shared if op == "same" else not shared
    if not a or not b:
        return None
    parse = parse_date if op in ("earlier", "later") else parse_number
    x, y = parse(a[0]), parse(b[0])
    if x is None or y is None or x == y:
        return None
    first_smaller = x < y
    if op in ("earlier", "smaller"):
        return 0 if first_smaller else 1
    return 1 if first_smaller else 0


def answer(plan: dict, values: dict, entities: list[str], equiv: dict | None = None):
    """Output template. values: {"v1": [...], ...}; entities: the resolved KU titles of E1 (, E2)."""
    t = plan["template"]
    if t == "chain":
        v = values.get("v2") or []
        return v[0] if v else None
    a, b = (values.get("v1"), values.get("v2")) if t == "compare" else (values.get("v3"), values.get("v4"))
    r = compare(a or [], b or [], plan["op"], equiv)
    if r is None:
        return None
    if plan["op"] in ("same", "different"):
        return "yes" if r else "no"
    return entities[r]
