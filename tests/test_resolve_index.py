import zlib

import numpy as np

from mxn.expert.resolve import Resolver, key


class FakeEnc:
    """Deterministic unit vectors from the text's words (shared words → similar vectors)."""

    def encode(self, texts, kind):
        out = []
        for t in texts:
            v = np.zeros(64, np.float32)
            for w in key(t):
                v[zlib.crc32(w.encode()) % 64] += 1
            out.append(v / (np.linalg.norm(v) or 1))
        return np.array(out)


TITLES = ["Paris", "Paris (band)", "Lyon", "John Smith", "Graz"]


def test_remove_deletes_title_keys_and_vector():
    r = Resolver(TITLES, FakeEnc())
    assert r("Paris", "q") == (0, "exact")
    r.remove(0)
    assert r.titles[0] is None and len(r.TV) == 4 and 0 not in r.ids
    assert all(0 not in v for t in (r.exact, r.nodis) for v in t.values())
    assert key("Paris") not in r.exact                      # empty key deleted, not left as []
    assert r("Paris", "q") == (1, "no_disambiguator")        # the band is the only "Paris" left
    r.remove(1)
    k, rule = r("Paris", "q")
    assert rule == "fallback" and k not in (0, 1)            # never returns a removed KU
    assert r("Graz", "q") == (4, "exact")                    # ids stay stable after removals
    assert np.allclose(r.TV[r.rows([4])[0]], FakeEnc().encode(["Graz."], "passage")[0])


def test_index_after_remove_equals_index_built_without_it():
    r = Resolver(TITLES, FakeEnc()); r.remove(2)
    fresh = Resolver([t for k, t in enumerate(TITLES) if k != 2], FakeEnc())
    m = {k: n for n, k in enumerate(k for k in range(len(TITLES)) if k != 2)}
    assert {kk: [m[k] for k in v] for kk, v in r.exact.items()} == dict(fresh.exact)
    assert {kk: [m[k] for k in v] for kk, v in r.nodis.items()} == dict(fresh.nodis)
    assert np.array_equal(r.TV, fresh.TV)
    for span in ("Lyon", "Paris", "John", "Smith Graz"):
        k, rule = r(span, "q"); kf, rf = fresh(span, "q")
        assert (m[k], rule) == (kf, rf)


def test_add_after_remove_and_detach_mask():
    r = Resolver(TITLES, FakeEnc()); r.remove(3)
    k = r.add("Brno")
    assert k == 5 and r("Brno", "q") == (5, "exact") and list(r.ids) == [0, 1, 2, 4, 5]
    r.off.add(5)
    assert r("Brno", "q")[0] != 5                            # detach masks, remove deletes
    r.off.discard(5); r.remove(5)
    assert r.titles[5] is None and 5 not in r.ids and not r.off
