"""String equality for answers (scoring and the measurement arms' slot labeling; never on the system's answer path).

Two answers match when their normalized strings are equal (mxn.expert.ops.normalize, the EM normalization), or the
country table maps both to the same key, or both parse as the same date (day, month, year; parts missing on both
sides are ignored, a part present on one side only does not match).
"""

from __future__ import annotations

import re

from mxn.expert.ops import normalize

_M = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
      "december"]
_MON = {m[:3]: i + 1 for i, m in enumerate(_M)}


def date_key(s: str):
    w = re.findall(r"[a-z]+|\d+", s.lower())
    y = [int(x) for x in w if x.isdigit() and len(x) in (3, 4)]
    d = [int(x) for x in w if x.isdigit() and len(x) <= 2]
    m = [_MON[x[:3]] for x in w if x[:3] in _MON and (len(x) == 3 or x in _M or x == "sept")]
    if len(y) != 1 or len(d) > 1 or len(m) > 1 or len(w) != len(y) + len(d) + len(m):
        return None
    return (y[0], m[0] if m else None, d[0] if d else None)


def same(a: str | None, b: str | None, equiv: dict | None = None) -> bool:
    if not a or not b:
        return False
    na, nb = normalize(a), normalize(b)
    if na == nb:
        return True
    if equiv and equiv.get(na, na) == equiv.get(nb, nb):
        return True
    da, db = date_key(a), date_key(b)
    return da is not None and da == db


def find_slot(value: str | None, slots, equiv: dict | None = None) -> int | None:
    """First slot whose label matches value (slot order = first occurrence in the article)."""
    if not value:
        return None
    return next((k for k, s in enumerate(slots) if same(s.label, value, equiv)), None)
