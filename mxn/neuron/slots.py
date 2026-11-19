"""Answer slots of one KU, built from its own article only (ruling N1: KU-owned state).

Surface rules, no relation semantics: full dates ("5 August 1898", "August 5, 1898"), bare years, and capitalized
name spans (runs of capitalized words, inner connectors allowed). A slot is one distinct normalized string; its
occurrences are (sentence index, start, end) in the KU's own sentences. Name slots can later carry a pointer
(another KU's id, wired by the KU's own RESOLVE broadcast).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from mxn.common.stopwords import _STOP

_MONTH = ("January|February|March|April|May|June|July|August|September|October|November|December|"
          "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec")
DATE = re.compile(rf"\b(?:\d{{1,2}} (?:{_MONTH})\.? \d{{3,4}}|(?:{_MONTH})\.? \d{{1,2}},? \d{{3,4}})\b")
YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
NAME = re.compile(r"\b[A-Z][\w'\-\.]*(?:\s+(?:(?:of|de|da|del|der|van|von|la|le|du|y|al|bin|ibn)\s+)?"
                  r"[A-Z][\w'\-\.]*)*")
_LEAD = {"He", "She", "It", "They", "His", "Her", "Its", "Their", "The", "A", "An", "In", "On", "At", "After",
         "Before", "During", "From", "This", "That", "These", "Those", "As", "When", "While", "Although", "But"}


@dataclass
class Slot:
    label: str                         # the answer string (literal) as written in the article
    kind: str                          # date | year | name
    occ: list = field(default_factory=list)     # (sentence index, start, end)
    pointer: int | None = None         # another KU's id once wired (name slots only)


def _norm(s: str) -> str:
    return " ".join(s.split()).strip(" .,;:'\"")


def build_slots(title: str, sentences: list[str], max_slots: int = 256) -> list[Slot]:
    """Slots in first-occurrence order; the KU's own title (and its parts) is not a slot. max_slots 256 (options
    round: the whole available article is covered; 64 cut 6 of 500 dev KUs)."""
    tl = _norm(re.sub(r"\s*\([^)]*\)\s*$", "", title)).lower()
    slots: dict[tuple, Slot] = {}

    def add(kind, label, i, a, b):
        label = _norm(label)
        if not label:
            return
        key = (kind, label.lower())
        if key not in slots:
            if len(slots) >= max_slots:
                return
            slots[key] = Slot(label, kind)
        slots[key].occ.append((i, a, b))

    for i, s in enumerate(sentences):
        taken = []
        for m in DATE.finditer(s):
            add("date", m.group(0), i, m.start(), m.end()); taken.append((m.start(), m.end()))
        for m in YEAR.finditer(s):
            if not any(a <= m.start() < b for a, b in taken):
                add("year", m.group(0), i, m.start(), m.end())
        for m in NAME.finditer(s):
            a, b, t = m.start(), m.end(), m.group(0)
            if a and s[a - 1] in "-'":                     # "pre-Code": part of a lower-case word
                continue
            if any(x < b and a < y for x, y in taken):
                continue
            words = t.split()
            while words and words[0] in _LEAD:              # sentence-initial function word
                a += len(words[0]) + 1; words = words[1:]
            t = " ".join(words)
            if not t or t.lower() in _STOP or t.lower() in tl or tl in t.lower():
                continue
            if re.fullmatch(_MONTH, t):
                continue
            add("name", t, i, a, a + len(t))
    return list(slots.values())
