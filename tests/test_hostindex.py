"""KU-owned resolution (mxn/expert/hostindex.py): with the same title vectors, per-host proposals merged by the
gateway give the step 6 resolver's decision on every call; host index bookkeeping."""
import hashlib

import numpy as np

from mxn.expert.hostindex import HostTitleIndex, KUTitle, merge
from mxn.expert.resolve import Resolver


class FakeEnc:
    def encode(self, texts, kind="query", batch=256):
        out = []
        for t in texts:
            seed = int(hashlib.sha1(f"{kind}|{t}".encode()).hexdigest()[:8], 16)
            v = np.random.default_rng(seed).standard_normal(16).astype(np.float32)
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


TITLES = ["John Smith (actor)", "John Smith (footballer)", "Paris", "Paris (film)", "Anne of Cleves", "London Films",
          "London", "Mary Smith", "John Smith", "The Heck with Hollywood!", "Frank Sinatra", "Paris (2003 film)"]


def split(titles, enc, H=3, vecs=None):
    TV = enc.encode([f"{t}." for t in titles], "passage") if vecs is None else vecs
    hosts = [HostTitleIndex([KUTitle(k, titles[k], TV[k]) for k in range(len(titles)) if k % H == h]) for h in range(H)]
    return hosts, TV


def resolve_split(hosts, enc, span, question):
    qs, qq = enc.encode([span], "query")[0], enc.encode([question], "query")[0]
    return merge([{"host": f"h{i}", **h.propose(span, qs, qq)} for i, h in enumerate(hosts)])


def test_same_decisions_as_central_resolver():
    enc = FakeEnc()
    central = Resolver(TITLES, enc)
    hosts, _ = split(TITLES, enc, vecs=central.TV)
    spans = ["John Smith", "Paris", "paris", "London", "Anne of Cleves", "Hollywood", "Sinatra", "Mary", "nothing at all",
             "the heck with hollywood", "John  Smith", "Paris (film)"]
    for i, sp in enumerate(spans):
        q = f"question {i} about {sp}?"
        k, rule = central(sp, q)
        r = resolve_split(hosts, enc, sp, q)
        assert (r["k"], r["rule"]) == (k, rule), sp
        assert r["title"] == TITLES[k]


def test_ties_go_to_smallest_ku_id():
    v = np.ones(4, np.float32) / 2
    hosts = [HostTitleIndex([KUTitle(5, "Same", v)]), HostTitleIndex([KUTitle(2, "Same", v)])]
    r = merge([{"host": f"h{i}", **h.propose("Same", v, v)} for i, h in enumerate(hosts)])
    assert r["k"] == 2 and r["rule"] == "exact_tie"
    r = merge([{"host": f"h{i}", **h.propose("other", v, v)} for i, h in enumerate(hosts)])
    assert r["k"] == 2 and r["rule"] == "fallback"


def test_detach_attach_add_remove():
    enc = FakeEnc()
    hosts, TV = split(TITLES, enc)
    k = TITLES.index("Frank Sinatra"); h = hosts[k % 3]
    h.detach(k)
    assert resolve_split(hosts, enc, "Frank Sinatra", "q")["k"] != k or resolve_split(hosts, enc, "Frank Sinatra", "q")["rule"] == "fallback"
    assert k not in h.ids
    h.attach(k)
    assert resolve_split(hosts, enc, "Frank Sinatra", "q") == {"k": k, "title": "Frank Sinatra", "host": f"h{k % 3}", "rule": "exact"}
    h.add(KUTitle(99, "Pequo Quoquozar", enc.encode(["Pequo Quoquozar."], "passage")[0]))
    assert resolve_split(hosts, enc, "Pequo Quoquozar", "q")["k"] == 99
    h.remove(99)
    assert 99 not in h.ids and "Pequo Quoquozar" not in [u.title for u in h.kus.values()]


def test_rebuild_from_kus_is_identical():
    enc = FakeEnc()
    hosts, _ = split(TITLES, enc)
    again = [HostTitleIndex(list(h.kus.values())) for h in hosts]
    for a, b in zip(hosts, again):
        assert dict(a.exact) == dict(b.exact) and dict(a.nodis) == dict(b.nodis)
        assert a.ids == b.ids and np.array_equal(a.TV, b.TV)
