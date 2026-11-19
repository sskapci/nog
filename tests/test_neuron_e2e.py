"""Neuron system end to end: a KU is built from its own article only, its stored package holds no raw text, its
connections come from its own article, and the resultant combines findings by the plan's template."""

import numpy as np
import torch

from mxn.neuron.ngateway import Resultant
from mxn.neuron.nku import Settings, build, package, unpackage


class FakeEnc:
    def encode(self, texts, kind="query"):
        return np.stack([np.random.default_rng(abs(hash(t)) % 2**32).standard_normal(16).astype(np.float32)
                         for t in texts])


TITLE = "Tomorrow at Seven"
SENTS = ["Tomorrow at Seven is a 1933 American pre-Code film directed by Ray Enright.",
         "It was released on 5 August 1933 and stars Chester Morris and Vivienne Osborne."]


def test_build_package_no_text_and_own_connections():
    st = Settings(hidden=8, epochs=5)
    u = build(1, TITLE, SENTS, FakeEnc(), st)
    d = package(u)
    flat = repr({k: v for k, v in d.items() if k != "state"})
    assert all(s not in flat for s in SENTS)
    j = next(j for j, s in enumerate(u.slots) if s.label == "Ray Enright")
    assert u.conns[j]["rels"] == {"director": 1} and u.conns[j]["target"] is None
    v = unpackage(d)
    assert v.text is None and all(torch.equal(a, b) for a, b in zip(u.net.state_dict().values(),
                                                                       v.net.state_dict().values()))


def test_build_is_deterministic():
    st = Settings(hidden=8, epochs=5)
    a, b = build(7, TITLE, SENTS, FakeEnc(), st), build(7, TITLE, SENTS, FakeEnc(), st)
    assert all(torch.equal(x, y) for x, y in zip(a.net.state_dict().values(), b.net.state_dict().values()))


def test_resultant_templates():
    # findings carry the answering KU's id k, as every host finding does (nhost.step); the resultant sorts by it
    # since 17ba29c (reverse lookup build), so findings without k raised KeyError 'k' (test data out of date).
    R = Resultant(None, "title", tau=0.1)
    plan = {"template": "chain", "r1": "director", "r2": "date of birth", "op": None, "conf": 1.0}
    R.register("q1", plan, [0], {"title": ["X"], "span": ["X"]})
    R.add([{"qid": "q1", "chain": 0, "hop": 0, "k": 1, "value": "Ray Enright", "f": 0.9, "final": False},
           {"qid": "q1", "chain": 0, "hop": 1, "k": 2, "value": "1890", "f": 0.8, "final": True}])
    assert R.done["q1"]["answer"] == "1890" and abs(R.done["q1"]["conf"] - 0.72) < 1e-9
    plan = {"template": "compare", "r1": "date of birth", "r2": None, "op": "earlier", "conf": 1.0}
    R.register("q2", plan, [0, 1], {"title": ["A", "B"], "span": ["A", "B"]})
    R.add([{"qid": "q2", "chain": 1, "hop": 0, "k": 3, "value": "1900", "f": 1.0, "final": True}])
    assert "q2" not in R.done
    R.add([{"qid": "q2", "chain": 0, "hop": 0, "k": 4, "value": "1890", "f": 1.0, "final": True}])
    assert R.done["q2"]["answer"] == "A"
    R.register("q3", plan, [], {"title": [], "span": []})
    assert R.done["q3"]["answer"] is None


def test_merge_once_counts_a_moving_ku_once():
    """Rebalancing: a KU held on two hosts for a moment is one candidate (exact, not exact_tie), at the first host;
    with no KU on two hosts merge_once = merge."""
    from mxn.expert.hostindex import merge
    from mxn.neuron.nhost import merge_once
    a = {"exact": [[7, "Same", 0.5]], "nodis": [], "fallback": [7, "Same", 0.5]}
    b = {"exact": [[8, "Same", 0.4]], "nodis": [], "fallback": [8, "Same", 0.4]}
    assert merge_once([{"host": "h3", **a}, {"host": "h1", **a}]) == {"k": 7, "title": "Same", "host": "h1",
                                                                       "rule": "exact"}
    assert merge([{"host": "h3", **a}, {"host": "h1", **a}])["rule"] == "exact_tie"
    two = [{"host": "h3", **a}, {"host": "h2", **b}]
    assert merge_once(two) == merge(two) and merge_once(two)["rule"] == "exact_tie"
    c = {"exact": [], "nodis": [], "fallback": None}
    assert merge_once([{"host": "h0", **c}]) is None


def test_value_kind_restriction_is_the_kus_own(monkeypatch):
    """A date relation is answered only from the KU's date / year slots; an entity relation only from name slots;
    the restriction travels with the package (design_value_kind.md)."""
    u = build(1, TITLE, SENTS, FakeEnc(), Settings(hidden=8, epochs=5, kind=True))
    name = next(j for j, s in enumerate(u.slots) if s.kind == "name")
    date = next(j for j, s in enumerate(u.slots) if s.kind == "date")
    with torch.no_grad():                         # the net prefers the name slot for every query
        u.net.net[-1].weight.zero_(); u.net.net[-1].bias.zero_(); u.net.net[-1].bias[name] = 5.0
        u.net.net[-1].bias[date] = 1.0
    q = np.zeros(16, np.float32)
    assert u.answer(q, "publication date", f_min=0.0)[1] == date
    assert u.answer(q, "director", f_min=0.0)[1] == name
    v = unpackage(package(u))
    assert v.kind_restrict and v.answer(q, "publication date", f_min=0.0)[1] == date
    monkeypatch.setenv("NKU_NO_KIND", "1")         # known-bad: restriction off → the name slot answers a date
    assert u.answer(q, "publication date", f_min=0.0)[1] == name
    monkeypatch.delenv("NKU_NO_KIND")
    u.kind_restrict = False
    assert u.answer(q, "publication date", f_min=0.0)[1] == name


def test_set_mode_chain_answer_is_last_hop():
    """Set mode: a one-hop chain answers hop 0's values (MetaQA, unchanged); a chain with r2 answers hop 1's values
    (demo corpus A parent -> X, reports/design_demo_rules.md §2). Fictional data."""
    R = Resultant(None, "title", tau=0.1, sets={"theta_f": 0.3})
    one = {"template": "chain", "r1": "has_tags", "r2": None, "op": None, "conf": 1.0}
    R.register("a", one, [0], {"title": ["M"], "span": ["M"]})
    R.add([{"qid": "a", "chain": 0, "hop": 0, "k": 1, "value": "x", "values": ["x", "y"], "f": 0.9, "final": True}])
    assert R.done["a"]["answer_set"] == ["x", "y"]
    two = {"template": "chain", "r1": "parent", "r2": "ceo", "op": None, "conf": 1.0}
    R.register("b", two, [0], {"title": ["Acme"], "span": ["Acme"]})
    R.add([{"qid": "b", "chain": 0, "hop": 0, "k": 1, "value": "Parentco", "f": 0.9, "final": False},
           {"qid": "b", "chain": 0, "hop": 1, "k": 2, "value": "Jo Doe", "values": ["Jo Doe"], "f": 0.8, "final": True}])
    assert R.done["b"]["answer_set"] == ["Jo Doe"]
