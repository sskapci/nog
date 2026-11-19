"""Whole-KU fact edit (step 9 UPDATABLE v2): replace every whole-word mention of a fact's value in a KU.

For country relations, every mention of any form in the old value's country-table class (e.g. "France", "French")
is replaced, case-insensitive, longest form first, so no equivalent surface form of the old fact is left behind.
"""

from __future__ import annotations

import re

from mxn.expert.ops import normalize

COUNTRY_RELATIONS = ("country of citizenship", "country")


def _pattern(forms: list[str], flags: int = 0) -> re.Pattern:
    alts = sorted({f for f in forms if f}, key=len, reverse=True)
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(f) for f in alts) + r")(?!\w)", flags)


def class_forms(value: str, table: dict[str, int]) -> list[str]:
    """All forms (normalized) in value's country-table class; [] if value is not in the table."""
    c = table.get(normalize(value))
    return [] if c is None else [f for f, k in table.items() if k == c]


def mentions(sentences: list[str], value: str, relation: str = "", table: dict | None = None) -> int:
    """Number of whole-word mentions of the value (and, for country relations, its class forms)."""
    pat = _edit_pattern(value, relation, table)
    return sum(len(pat.findall(s)) for s in sentences)


def _edit_pattern(value: str, relation: str, table: dict | None) -> re.Pattern:
    if relation in COUNTRY_RELATIONS and table and class_forms(value, table):
        return _pattern(class_forms(value, table) + [value], re.IGNORECASE)
    return _pattern([value])


def edit_ku(sentences: list[str], old: str, new: str, relation: str = "", table: dict | None = None) -> tuple[list[str], int]:
    """New sentences with every mention of the old value replaced by the new one, and the number replaced."""
    pat, n = _edit_pattern(old, relation, table), 0
    out = []
    for s in sentences:
        s2, k = pat.subn(lambda m: new, s)
        out.append(s2); n += k
    return out, n


def leading_name_match(answer: str | None, value: str) -> bool:
    """UPDATABLE v2 scoring: the answer equals the value's leading full name part, the segment before the first comma
    ("Kirksville" for "Kirksville, Missouri"); a value without a comma has no leading part ("New" ≠ "New York")."""
    return bool(answer) and "," in value and normalize(answer) == normalize(value.split(",")[0])


# UPDATABLE attempt 3 (decisions 2026-10-27f §3): edit only where every mention of the old value stands alone.
PLACE_RELATIONS = ("place of birth", "place of death", "place of burial")
PARTICLES = {"of", "de", "du", "da", "di", "del", "della", "des", "von", "van", "der", "la", "le"}
_REGION = re.compile(r", ((?:[A-Z][\w'’]*)(?: [A-Z][\w'’]*)*)")


def _cap(w: str) -> bool:
    return w[:1].isupper()


def standalone(s: str, a: int, b: int, relation: str) -> int | None:
    """End of the text to replace for the mention s[a:b] (past its ", Region" for a place relation), or None when the
    mention is part of something larger: joined by a hyphen or "'s", next to a capitalized word, after
    "Capitalized," (it is itself a region), or inside a "Capitalized of/de/... X" name."""
    if relation in PLACE_RELATIONS:
        m = _REGION.match(s, b)
        if m:
            b = m.end()
    left, right = s[:a], s[b:]
    if left.endswith(("-", "‐")) or right.startswith(("-", "‐", "'s", "’s")):
        return None
    nxt = re.match(r" (\S+)", right)
    if nxt and (_cap(nxt.group(1)) or (nxt.group(1) in PARTICLES and re.match(r" \S+ [A-Z]", right))):
        return None
    words = left.split()
    if left.endswith(" ") and words:
        prev = words[-1]
        if prev.endswith(",") and _cap(prev):
            return None
        if _cap(prev) and len(words) > 1 and not prev.startswith("("):
            return None
        if prev in PARTICLES and len(words) > 1 and _cap(words[-2]):
            return None
    return b


def edit_ku_clean(sentences: list[str], forms: dict[str, str], relation: str,
                  ignore_case: bool = False) -> tuple[list[str], int] | None:
    """forms: old surface form → replacement. Every whole-word mention of every form is replaced (a place's
    ", Region" with it); None when any mention is not standalone (the KU is not cleanly editable)."""
    pat = _pattern(list(forms), re.IGNORECASE if ignore_case else 0)
    lookup = {(f.lower() if ignore_case else f): r for f, r in forms.items()}
    out, n = [], 0
    for s in sentences:
        pieces, pos = [], 0
        for m in pat.finditer(s):
            if m.start() < pos:
                continue
            end = standalone(s, m.start(), m.end(), relation)
            if end is None:
                return None
            pieces += [s[pos:m.start()], lookup[m.group(0).lower() if ignore_case else m.group(0)]]
            pos = end; n += 1
        out.append("".join(pieces) + s[pos:])
    return out, n
