"""Arm S training examples for one KU, from its own sentences only (ruling N2).

For every slot occurrence: the sentence with the slot masked; relation codes from cue words of that relation at most
WINDOW words before the slot in the same sentence (kinds must fit: date relations take date / year slots, the others
name slots); for each such relation the query frames about the KU's own title ("the director of <title>", ...).
Negatives: frames of relations with no cue anywhere in the article → none; a masked sentence with a relation code
that has no cue next to that slot → none. Target -1 = none.
"""

from __future__ import annotations

import random
import re

from mxn.neuron.signal import CUES, RELATIONS, fits, frames
from mxn.neuron.slots import Slot

WINDOW = 4
MASK = "what"
_W = re.compile(r"\S+")


def _fits(rel: str, slot: Slot) -> bool:
    return fits(rel, slot.kind)


def _cue_spans(sentence: str) -> list[tuple[str, int, int]]:
    low = sentence.lower(); out = []
    for rel, cues in CUES.items():
        for c in cues:
            for m in re.finditer(rf"\b{re.escape(c)}\b", low):
                out.append((rel, m.start(), m.end()))
    return out


def _word_index(sentence: str, pos: int) -> int:
    return sum(1 for m in _W.finditer(sentence) if m.end() <= pos)


def _mask(sentence: str, spans) -> str:
    for a, b in sorted(set(spans), reverse=True):
        sentence = sentence[:a] + MASK + sentence[b:]
    return sentence


def slot_relations(sentences: list[str], slots: list[Slot]) -> dict[tuple, set]:
    """(slot index, occurrence index) → relations cued next to it."""
    cues = [_cue_spans(s) for s in sentences]
    out = {}
    for k, sl in enumerate(slots):
        for j, (i, a, b) in enumerate(sl.occ):
            wa = _word_index(sentences[i], a)
            out[(k, j)] = {r for r, ca, cb in cues[i]           # the cue comes first ("directed by X", "father, X")
                           if _fits(r, sl) and cb <= a and wa - _word_index(sentences[i], ca) <= WINDOW}
    return out


def examples(title: str, sentences: list[str], slots: list[Slot], seed: int = 0, neg_rel: int = 4,
             extra: dict | None = None, cues: bool = True, dense: bool = False) -> list[tuple]:
    """[(text, relation or None, target slot index or -1)]
    extra: {slot index: relations} added to every occurrence of that slot (measurement arms L, R, O only);
    cues=False drops the cue labels (arms R, O label from their own source alone).
    dense (option 1, own sentences only): per cued (slot k, relation r) also the unmasked sentence → k; the sentence
    with k and every other slot of the same kind masked → k; for each other same-kind slot k2 in the sentence, the
    sentence with only k2 masked + r → k (the answer does not follow the blank: same-kind slots as negatives). Every
    slot occurrence is already a target (masked, no relation)."""
    rng = random.Random(seed)
    subject = re.sub(r"\s*\([^)]*\)\s*$", "", title)
    rels = slot_relations(sentences, slots) if cues else {(k, j): set() for k, sl in enumerate(slots)
                                                          for j in range(len(sl.occ))}
    for k, rs in (extra or {}).items():
        for j in range(len(slots[k].occ)):
            rels[(k, j)] = rels[(k, j)] | set(rs)
    cued = set().union(*rels.values()) if rels else set()
    ex = []
    for (k, j), rs in rels.items():
        i, a, b = slots[k].occ[j]
        masked = sentences[i][:a] + MASK + sentences[i][b:]
        if rs:
            for r in sorted(rs):
                ex.append((masked, r, k))
                ex += [(f, r, k) for f in frames(r, subject)]
                if dense:
                    same = [(k2, a2, b2) for k2, sl in enumerate(slots) if k2 != k and sl.kind == slots[k].kind
                            for i2, a2, b2 in sl.occ if i2 == i]
                    ex.append((sentences[i], r, k))
                    ex.append((_mask(sentences[i], [(a, b)] + [(a2, b2) for _, a2, b2 in same]), r, k))
                    ex += [(_mask(sentences[i], [(a2, b2)]), r, k) for _, a2, b2 in same]
            other = [r for r in RELATIONS if r not in rs]
            ex.append((masked, rng.choice(other), -1))
        else:
            ex.append((masked, None, k))
    uncued = [r for r in RELATIONS if r not in cued]
    for r in rng.sample(uncued, min(neg_rel, len(uncued))):
        ex += [(f, r, -1) for f in frames(r, subject)]
    return ex
