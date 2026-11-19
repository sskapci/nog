"""Our fact extractor (step 4 of 2026-10-24b): (KU text, relation) → the object as a verbatim span.

The step 2 encoder's trunk (our weights) with a per-piece BIO head. Input per sentence:
"<relation> [SEP] <title>. <sentence>"; only the sentence's pieces are tagged. One shared model; a KU
gets a read-only copy (ReadOnlyExtractor) and calls it on its own sentences only.
Date-rules version: dates_rule() replaces the model for date relations when it fires.
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn

from mxn.expert.encoder import BiEncoder, pad
from mxn.expert.planner import pieces

TAGS = ["O", "B", "I"]
DATE_RELS = {"date of birth", "date of death", "publication date", "inception"}


class Extractor(nn.Module):
    def __init__(self, trunk: dict | None = None):
        super().__init__()
        self.trunk = BiEncoder(**(trunk or {}))
        self.tag = nn.Linear(self.trunk.norm.normalized_shape[0], len(TAGS))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        t = self.trunk
        mask = ids != 0
        pos = torch.arange(ids.shape[1], device=ids.device)[None, :]
        x = t.tok(ids) + t.pos(pos) + t.typ.weight[1]
        return self.tag(t.norm(t.enc(x, src_key_padding_mask=~mask)))


def encode_input(sp, relation: str, title: str, sentence: str, max_len: int = 128):
    """Piece ids, their character offsets inside `sentence` (None for the prefix's pieces)."""
    prefix = f"{relation} [SEP] {title}. "
    ids, offs = pieces(sp, prefix + sentence, max_len)
    n = len(prefix)
    return ids, [(a - n, b - n) if a >= n else None for a, b in offs]


def bio(offsets, spans) -> list[int]:
    """Tag per piece (-100 for prefix pieces) from character spans in the sentence."""
    out = []
    for o in offsets:
        if o is None:
            out.append(-100); continue
        t = 0
        for a, b in spans:
            s, e = o
            if (e > a and s < b and e > s) or (e == s and a <= s < b):
                t = 1 if s <= a else 2
        out.append(t)
    # a span's first tagged piece is B even when it starts mid-piece
    for a, b in spans:
        first = [i for i, o in enumerate(offsets) if o is not None and out[i] in (1, 2) and o[1] > a and o[0] < b]
        if first:
            out[first[0]] = 1
    return out


def predict(net: Extractor, sp, relation: str, title: str, sentences: list[str]) -> tuple[str, float]:
    """The best span over the KU's sentences and its score (mean P(B or I) over its pieces)."""
    enc = [encode_input(sp, relation, title, s) for s in sentences]
    if not enc:
        return "", 0.0
    with torch.inference_mode():
        P = net(pad([e[0] for e in enc])).softmax(-1)
    best = ("", 0.0)
    for k, (ids, offs) in enumerate(enc):
        p = P[k, : len(ids)]
        tags = p.argmax(-1).tolist()
        i = 0
        while i < len(ids):
            if offs[i] is not None and tags[i] in (1, 2):
                j = i
                while j + 1 < len(ids) and offs[j + 1] is not None and tags[j + 1] == 2:
                    j += 1
                score = float(p[i: j + 1, 1:].sum(-1).mean())
                a, b = offs[i][0], offs[j][1]
                text = sentences[k][max(a, 0): b].strip()
                if text and score > best[1]:
                    best = (text, score)
                i = j + 1
            else:
                i += 1
    return best


def piece_scores(net: Extractor, sp, relation: str, title: str, sentences: list[str]):
    """Per sentence: (p = P(B) + P(I) per piece, offsets); the input of decode()."""
    enc = [encode_input(sp, relation, title, s) for s in sentences]
    if not enc:
        return []
    with torch.inference_mode():
        P = net(pad([e[0] for e in enc])).softmax(-1)
    return [(P[k, : len(ids), 1:].sum(-1).tolist(), offs) for k, (ids, offs) in enumerate(enc)]


def snap(text: str, a: int, b: int) -> tuple[int, int]:
    """Widen [a, b) to whole words."""
    while a > 0 and (text[a - 1].isalnum() or text[a - 1] == "_"):
        a -= 1
    while b < len(text) and (text[b].isalnum() or text[b] == "_"):
        b += 1
    return a, b


def decode(scores, sentences: list[str], t: float) -> tuple[str, float]:
    """v2 decoding: maximal runs of sentence pieces with p >= t, best by mean p, snapped to whole words."""
    best = ("", 0.0)
    for (p, offs), s in zip(scores, sentences):
        i = 0
        while i < len(p):
            if offs[i] is not None and p[i] >= t:
                j = i
                while j + 1 < len(p) and offs[j + 1] is not None and p[j + 1] >= t:
                    j += 1
                score = sum(p[i: j + 1]) / (j + 1 - i)
                a, b = snap(s, max(offs[i][0], 0), offs[j][1])
                text = s[a: b].strip()
                if text and score > best[1]:
                    best = (text, score)
                i = j + 1
            else:
                i += 1
    return best


class ReadOnlyExtractor:
    """A KU's copy: inference only, weights frozen, on that KU's own text."""

    def __init__(self, net: Extractor, sp):
        self._net, self._sp = net, sp
        for p in self._net.parameters():
            p.requires_grad_(False)
        self._net.eval()

    def __call__(self, relation: str, title: str, sentences: list[str]) -> tuple[str, float]:
        return predict(self._net, self._sp, relation, title, sentences)


MONTH = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
DATE = re.compile(rf"\b(?:\d{{1,2}} {MONTH},? \d{{3,4}}|{MONTH} \d{{1,2}},? \d{{3,4}}|{MONTH},? \d{{3,4}}|\d{{3,4}})\b")


def dates_rule(relation: str, sentences: list[str]) -> str | None:
    """Rule-based date for date relations (step 4 rules); None if it does not fire."""
    if relation not in DATE_RELS:
        return None
    text = " ".join(sentences)
    if relation in ("date of birth", "date of death"):
        m = re.search(r"\(([^()]*?[–-][^()]*?)\)", text)
        if m:
            ds = DATE.findall(m.group(1))
            if relation == "date of birth" and ds:
                return ds[0]
            if relation == "date of death" and len(ds) >= 2:
                return ds[1]
        m = re.search(r"\b" + ("born" if relation == "date of birth" else "died") + r"\b(.*)", text)
        if m:
            ds = DATE.findall(m.group(1))
            if ds:
                return ds[0]
        return None
    ds = DATE.findall(text)
    return ds[0] if ds else None


PLACE_RELS = ("place of birth", "place of death")
_CAPRUN = r"[A-Z][^\s,;()–]*(?:\s+(?:(?:de|of|del|la|upon|on|am|an|der)\s+)?[A-Z][^\s,;()–]*)*"
_PLACE_IN = re.compile(rf"\bin\s+({_CAPRUN}(?:,\s*{_CAPRUN})*)")


def place_rule(relation: str, sentences: list[str]) -> str | None:
    """Extractor v4 life-span rule: in the first "(…)" with a dash between dates, the left side is birth and the right
    side death; the place is the capitalized run after "in", else what is left of the side once dates and
    "born"/"died" are removed, if that is a capitalized run. None if it does not fire."""
    if relation not in PLACE_RELS:
        return None
    m = re.search(r"\(([^()]*?)\)", " ".join(sentences))
    while m and not (DATE.search(m.group(1)) and re.search(r"[–]|\s-\s", m.group(1))):
        m = re.search(r"\(([^()]*?)\)", m.string[m.end():]) if m.end() < len(m.string) else None
    if not m:
        return None
    sides = re.split(r"\s*[–]\s*|\s+-\s+", m.group(1), maxsplit=1)
    if len(sides) < 2:
        return None
    side = sides[0] if relation == "place of birth" else sides[1]
    p = _PLACE_IN.search(side)
    if p:
        return p.group(1).strip()
    rest = re.sub(r"\b(?:born|died|b\.|d\.)\s*", " ", DATE.sub(" ", side))
    rest = re.sub(r"^[\s,;]+|[\s,;]+$", "", re.sub(r"\s+", " ", rest))
    return rest if rest and re.fullmatch(rf"{_CAPRUN}(?:,\s*{_CAPRUN})*", rest) else None
