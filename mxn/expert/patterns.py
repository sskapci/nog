"""Extractor v3 pattern rules (rules: reports/experiment_log.md, "extractor v3 rules"): deterministic triggers for
family relations and places, mined from fit sentences only, applied like the date rules.

A sentence is tokenized and every run of capitalized words (connectors allowed inside) becomes one "<N>" token
carrying its text. A trigger is a tuple of 1-4 abstract tokens; it fires where it is followed by an "<N>" token,
whose text is the value.
"""

from __future__ import annotations

import re
from collections import defaultdict

PATTERN_RELS = ("father", "mother", "spouse", "place of birth", "place of death")
CONNECT = {"de", "da", "del", "di", "van", "von", "der", "la", "le", "of", "bin", "ibn", "al"}
TOKEN = re.compile(r"[^\W\d_]\.|\w[\w'’.-]*\w|\w|[^\w\s]")     # an initial keeps its period ("T.")
MAX_K = 4
# amendment (fit stats, before any mdev/test result): leading titles are not part of a value, and a person's
# peerage suffix (", 2nd Earl of Suffolk", ", Duke of Saxe-Merseburg") is
TITLES = {"King", "Queen", "Prince", "Princess", "Emperor", "Empress", "Sultan", "Duke", "Duchess", "Count",
          "Countess", "Sir", "Lord", "Lady", "Saint", "St.", "Pope", "Tsar", "Shah", "Archduke", "Grand", "Earl",
          "Baron", "Marquess", "Infante", "Infanta", "Crown", "Emir", "Maharaja", "Raja", "Khan"}
PEER = re.compile(r",\s+(?:\d+(?:st|nd|rd|th)\s+)?(?:Earl|Duke|Count|Baron|Marquess|Viscount|Prince|Lord|Margrave|"
                  r"Landgrave|Elector|Baronet)\s+of\s+[A-Z][\w'’-]*(?:[\s-]+[A-Z][\w'’-]*)*")


def _cap(w: str) -> bool:
    return w[:1].isupper()


def abstract(sentence: str) -> list[tuple[str, int, int]]:
    """[(abstract token, char begin, char end)]: capitalized runs → ("<N>", run begin, run end); other words
    lowercased; punctuation kept."""
    toks = [(m.group(), m.start(), m.end()) for m in TOKEN.finditer(sentence)]
    out, i = [], 0
    while i < len(toks):
        w, a, b = toks[i]
        if _cap(w):
            j = i
            while j + 1 < len(toks):
                if _cap(toks[j + 1][0]):
                    j += 1
                elif toks[j + 1][0] in CONNECT and j + 2 < len(toks) and _cap(toks[j + 2][0]):
                    j += 2
                else:
                    break
            out.append(("<N>", a, toks[j][2])); i = j + 1
        else:
            out.append((w.lower(), a, b)); i += 1
    return out


def value(s: str, a: int, b: int, place: bool) -> str:
    """A run's value: leading titles dropped (if a word remains); for persons, a peerage suffix kept."""
    words = s[a:b].split()
    while len(words) > 1 and words[0] in TITLES:
        words.pop(0)
    v = " ".join(words)
    if not place:
        m = PEER.match(s, b)
        if m:
            v += m.group()
    return v


def occurrences(sentences: list[str], place: bool = False):
    """Every (trigger, value) in sentence order, first occurrence of each trigger only."""
    seen = {}
    for s in sentences:
        ab = abstract(s)
        for i, (t, a, b) in enumerate(ab):
            if t != "<N>":
                continue
            for k in range(1, MAX_K + 1):
                if i - k < 0:
                    break
                trig = tuple(x[0] for x in ab[i - k: i])
                if trig not in seen:
                    seen[trig] = value(s, a, b, place)
    return seen


def mine(rows, normalize, place: bool, min_fires: int = 30, min_precision: float = 0.50):
    """rows: (sentences, gold strings) of one relation. → ranked [(trigger, fires, correct, precision)]."""
    fires, right = defaultdict(int), defaultdict(int)
    for sentences, golds in rows:
        g = {normalize(x) for x in golds}
        for trig, val in occurrences(sentences, place).items():
            fires[trig] += 1; right[trig] += normalize(val) in g
    keep = [(t, fires[t], right[t], right[t] / fires[t]) for t in fires
            if fires[t] >= min_fires and right[t] / fires[t] >= min_precision]
    return sorted(keep, key=lambda x: (-x[3], -x[1], x[0]))


def pattern_rule(relation: str, sentences: list[str], triggers: dict) -> str | None:
    """The value of the highest-ranked trigger that fires (first occurrence in sentence order); None if none or the
    relation has no rules switched on. triggers: {relation: [trigger tuple, ...] ranked}."""
    ranked = triggers.get(relation)
    if not ranked:
        return None
    occ = occurrences(sentences, relation.startswith("place"))
    for t in ranked:
        if t in occ:
            return occ[t]
    return None
