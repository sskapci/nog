"""FORGET and its audit FAUDIT on a stand-in host (reports/design_forget_edit.md §1): known-good (everything of the
forgotten KU gone, audit 0) and known-bad (cold copy kept, pointers kept, a parameter reference kept: each caught).
Fictional data."""
import threading
import types

import torch

from mxn.neuron.nhost import NeuronHost


class _Index:
    def __init__(self, kus):
        self.kus = {k: types.SimpleNamespace(k=k, title=u.title) for k, u in kus.items()}

    def remove(self, k):
        del self.kus[k]


def _ku(k, title, labels, target=None):
    torch.manual_seed(k)
    u = types.SimpleNamespace(k=k, title=title, slots=[types.SimpleNamespace(label=x) for x in labels],
                              conns={0: {"target": target}} if target is not None else {}, net=torch.nn.Linear(3, 2))
    u.labels = lambda u=u: [s.label for s in u.slots]
    return u


def _host(tmp_path):
    h = NeuronHost.__new__(NeuronHost)
    h.name = "h0"
    h.kus = {1: _ku(1, "Acme Labs", ["Jo Vexa", "Zorburg"]), 2: _ku(2, "Bolt Group", ["Acme Labs", "Zorburg"], target=1),
             3: _ku(3, "Cyra Works", ["Ann Pell"])}
    h.index = _Index(h.kus)
    h.incoming, h.moved, h.plan, h.pcache = {}, {}, {}, {(1, "ceo"): [0.5], (2, "ceo"): [0.5]}
    h._cache, h._cache_lab, h._cache_ids, h._cache_lock = {}, {}, set(), threading.Lock()
    for k in h.kus:
        h._cache_add(k)
    h._cache_ids = set(h.kus)
    h.cold = tmp_path / "cold"; h.cold.mkdir()
    torch.save([{"k": 1, "title": "Acme Labs"}, {"k": 3, "title": "Cyra Works"}], h.cold / "g1_h1_00000.pt")
    h.cold_ids = {1: {1, 3}}
    return h


def _audit(h, ph):
    return h.faudit({"mode": "audit", "ids": [1], "titles": ["Acme Labs"], "values": ["Jo Vexa"], "sha": [],
                     "phash": [ph]})


def test_forget_known_good(tmp_path):
    h = _host(tmp_path)
    ph = NeuronHost.phash(h.kus[1])
    before = _audit(h, ph)
    assert before["held"] == 1 and before["cold"] == 1 and before["pointers"] == 1 and before["unique_labels"] == 1
    r = h.forget([1])
    assert r == {"host": "h0", "deleted": 1, "cold_deleted": 1, "pointers_cleared": 1}
    a = _audit(h, ph)
    assert a["total"] == 0, a
    assert 1 not in h.kus and 1 not in h.index.kus and (1, "ceo") not in h.pcache and (2, "ceo") in h.pcache
    assert all(1 not in v for v in h._cache.values()) and "Jo Vexa" not in h._cache
    assert h.kus[3].net is not None and 3 in h._cache["Ann Pell"]          # other KUs untouched


def test_forget_known_bad_cold(tmp_path, monkeypatch):
    h = _host(tmp_path)
    ph = NeuronHost.phash(h.kus[1])
    monkeypatch.setenv("NHOST_FORGET_KEEP_COLD", "1")
    h.forget([1])
    assert _audit(h, ph)["cold"] == 1


def test_forget_known_bad_scrub(tmp_path, monkeypatch):
    h = _host(tmp_path)
    ph = NeuronHost.phash(h.kus[1])
    monkeypatch.setenv("NHOST_FORGET_NO_SCRUB", "1")
    h.forget([1])
    assert _audit(h, ph)["pointers"] == 1


def test_forget_known_bad_parameters_kept(tmp_path):
    h = _host(tmp_path)
    ph = NeuronHost.phash(h.kus[1])
    kept = h.kus[1].net                                    # a stray reference to the forgotten KU's parameters
    h.forget([1])
    assert _audit(h, ph)["parameter_modules"] == 1
    del kept
