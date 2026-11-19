"""Entity resolution owned by the KUs (decisions 2026-10-28, central resolver fix).

Architecture check (roadmap "share-nothing knowledge units"; decision 2026-10-24b §6): each KU owns its title vector
(computed by the KU from its own title with the shared read-only encoder, one title at a time); a host keeps a title
index over its own KUs only, as a cache that proposes candidates and is rebuilt from those KUs; the KU scores itself;
no index spans hosts. The gateway holds no titles, vectors or placement: it encodes the span and the question (a
language expert's job), broadcasts RESOLVE, and merges the returned scores with `merge` (ties by KU id, the same rule
everywhere).

Same decision order as the step 6 resolver (mxn/expert/resolve.py), now split across hosts:
  1. exact word-key title match; 2. title without its trailing "(...)"; in either tier, one candidate over all hosts
  wins outright, several are scored by each KU (question vs its title vector) and the best wins;
  3. fallback: each host proposes its best KU by span vs title vector, the KU's own score is returned, best wins.
Ties (equal scores) go to the smallest KU id in the shared order (id_key: build ids, then host-named ids).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from mxn.common.textkeys import _DISAMBIG, _WORD


def id_key(k) -> tuple:
    """Total order on KU ids, the same everywhere: the build's integer ids first (by value), then host-named ids
    ("<host>:<n>", by text). Used for ties and for sorting."""
    return (0, k, "") if isinstance(k, int) else (1, 0, str(k))


def key(s: str) -> tuple[str, ...]:
    return tuple(_WORD.findall(s.lower()))


class KUTitle:
    """What a KU owns for resolution: its id, title and title vector; it scores a query vector itself."""

    def __init__(self, k: int, title: str, vec: np.ndarray):
        self.k, self.title, self.vec = k, title, vec

    def score(self, qv: np.ndarray) -> float:
        return float(self.vec @ qv)


class HostTitleIndex:
    """A host's cache over its own KUs: the two key tables and a matrix of its KUs' own title vectors (rows in
    ascending KU id) for proposing the fallback candidate. Rebuilt from the KUs (`from_units`); add / remove / detach /
    attach keep it in step with the host's KUs."""

    def __init__(self, kus: list[KUTitle]):
        self.kus: dict[int, KUTitle] = {}
        self.off: dict[int, KUTitle] = {}
        for u in kus:
            self.kus[u.k] = u
        self._rebuild()

    @classmethod
    def from_units(cls, units: dict, enc, progress=None) -> "HostTitleIndex":
        """units: {k: (title, sentences)}; each KU encodes its own title (one at a time, as a KU would).
        progress(done, total) is called every 500 KUs."""
        kus = []
        for n, (k, (t, _)) in enumerate(sorted(units.items())):
            kus.append(KUTitle(k, t, title_vector(enc, t)))
            if progress and (n + 1) % 500 == 0:
                progress(n + 1, len(units))
        return cls(kus)

    def _rebuild(self) -> None:
        self.exact, self.nodis = defaultdict(list), defaultdict(list)
        for k in sorted(self.kus, key=id_key):
            t = self.kus[k].title
            self.exact[key(t)].append(k)
            self.nodis[key(_DISAMBIG.sub("", t))].append(k)
        self.ids = sorted(self.kus, key=id_key)                     # rows in the shared id order
        dim = next(iter(self.kus.values())).vec.shape[0] if self.kus else 1
        self.TV = np.stack([self.kus[k].vec for k in self.ids]) if len(self.ids) else np.zeros((0, dim), np.float32)

    def add(self, u: KUTitle) -> None:
        self.kus[u.k] = u; self._rebuild()

    def remove(self, k: int) -> None:
        self.kus.pop(k, None); self.off.pop(k, None); self._rebuild()

    def detach(self, k: int) -> None:
        if k in self.kus:
            self.off[k] = self.kus.pop(k); self._rebuild()

    def attach(self, k: int) -> None:
        if k in self.off:
            self.kus[k] = self.off.pop(k); self._rebuild()

    def propose(self, span: str, qv_span: np.ndarray, qv_question: np.ndarray) -> dict:
        """Candidates from this host's KUs. exact / nodis: [[k, title, question score], ...] (the KU's own score);
        fallback: [k, title, span score] of the host's best KU (ties: smallest k), or None if the host has no KU."""
        out = {}
        for name, table in (("exact", self.exact), ("nodis", self.nodis)):
            out[name] = [[k, self.kus[k].title, self.kus[k].score(qv_question)] for k in table.get(key(span), [])]
        if len(self.ids):
            r = int(np.argmax(self.TV @ qv_span))                  # proposal from the cache; first max = smallest id
            u = self.kus[self.ids[r]]
            out["fallback"] = [u.k, u.title, u.score(qv_span)]     # the KU scores itself
        else:
            out["fallback"] = None
        return out


def title_vector(enc, title: str) -> np.ndarray:
    return enc.encode([f"{title}."], "passage")[0]


def merge(replies: list[dict]) -> dict | None:
    """Merge hosts' proposals (each reply: {"host", "exact", "nodis", "fallback"}) → {"k", "title", "host", "rule"}.
    Same order as the step 6 resolver; equal scores go to the smallest KU id."""
    for rule, tier in (("exact", "exact"), ("no_disambiguator", "nodis")):
        c = [(k, t, s, r["host"]) for r in replies for k, t, s in r[tier]]
        if c:
            if len(c) == 1:
                k, t, _, h = c[0]
                return {"k": k, "title": t, "host": h, "rule": rule}
            k, t, _, h = min(c, key=lambda x: (-x[2], id_key(x[0])))
            return {"k": k, "title": t, "host": h, "rule": rule + "_tie"}
    c = [(r["fallback"][0], r["fallback"][1], r["fallback"][2], r["host"]) for r in replies if r.get("fallback")]
    if not c:
        return None
    k, t, _, h = min(c, key=lambda x: (-x[2], id_key(x[0])))
    return {"k": k, "title": t, "host": h, "rule": "fallback"}
