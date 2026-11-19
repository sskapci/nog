"""Entity resolution (step 6 rules): span text → KU index. Same logic as experiments/expert/resolve_run.py:
exact word-key title match; then title without its trailing "(...)" (ties broken by our encoder, question vs
"title."); then our encoder, span vs every "title.", top-1."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from mxn.common.textkeys import _DISAMBIG, _WORD


def key(s: str) -> tuple[str, ...]:
    return tuple(_WORD.findall(s.lower()))


class Resolver:
    """The gateway's title index, per KU id k: titles[k]; exact[key(title)] and nodis[key(title without "(...)")] →
    [k, ...]; one title vector ("title.") per KU, in TV with ids[row] = k. No KU content. off: detached KUs (masked,
    kept). remove(k): permanent removal (title, both keys and the vector deleted; titles[k] = None keeps ids stable)."""

    def __init__(self, titles: list[str], enc):
        self.titles, self.enc = list(titles), enc
        self.exact, self.nodis = defaultdict(list), defaultdict(list)
        for k, t in enumerate(titles):
            self._keys(k, t, "append")
        self.TV = enc.encode([f"{t}." for t in titles], "passage")
        self.ids = np.arange(len(titles))              # TV row → KU id; ascending (adds append, removals delete)
        self.off: set[int] = set()                  # detached KUs (step 9): never returned

    def _keys(self, k: int, title: str, op: str) -> None:
        for table, kk in ((self.exact, key(title)), (self.nodis, key(_DISAMBIG.sub("", title)))):
            getattr(table[kk], op)(k)
            if not table[kk]:
                del table[kk]

    def rows(self, ks) -> np.ndarray:
        return np.searchsorted(self.ids, ks)

    def add(self, title: str) -> int:
        """A new KU (step 9 UPDATABLE): indexed like the others; returns its index."""
        k = len(self.titles)
        self.titles.append(title)
        self._keys(k, title, "append")
        self.TV = np.vstack([self.TV, self.enc.encode([f"{title}."], "passage")])
        self.ids = np.append(self.ids, k)
        return k

    def remove(self, k: int) -> None:
        """Permanent removal: the KU's title, both keys and its vector are deleted from the index."""
        self._keys(k, self.titles[k], "remove")
        r = int(self.rows([k])[0])
        assert self.ids[r] == k
        self.TV, self.ids = np.delete(self.TV, r, axis=0), np.delete(self.ids, r)
        self.titles[k] = None
        self.off.discard(k)

    def __call__(self, span: str, question: str) -> tuple[int, str]:
        """(KU index, rule)."""
        for rule, table in (("exact", self.exact), ("no_disambiguator", self.nodis)):
            c = [k for k in table.get(key(span), []) if k not in self.off]
            if c:
                if len(c) == 1:
                    return c[0], rule
                q = self.enc.encode([question], "query")[0]
                return c[int(np.argmax(self.TV[self.rows(c)] @ q))], rule + "_tie"
        q = self.enc.encode([span], "query")[0]
        s = self.TV @ q
        if self.off:
            s[self.rows(sorted(self.off))] = -np.inf
        return int(self.ids[int(np.argmax(s))]), "fallback"
