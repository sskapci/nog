"""The host's slot-value cache uses the same label normalization as the KU's own slot check (nku.holds), on every
label of the test pool (owner requirement 2026-11-12; reports/design_slot_cache.md).

Builds each pool KU's slots from its own article (as TRAIN does; no training needed for slots) and checks, for every
KU and every slot label: the KU is in the cache under exactly the key its own holds() uses; a KU whose holds() says
yes for (relation, label) is always a cache candidate for that label; every cache entry names only KUs whose own
labels contain the key. Pool: artifacts/expert_mqs/pool_mqstest.pkl (skipped if absent)."""
import pickle
import types
from pathlib import Path

import pytest

from mxn.expert.ops import normalize
from mxn.neuron.nhost import NeuronHost
from mxn.neuron.nku import NeuronKU
from mxn.neuron.signal import KIND
from mxn.neuron.slots import build_slots

POOL = Path(__file__).resolve().parents[1] / "artifacts" / "expert_mqs" / "pool_mqstest.pkl"


def _kus(limit=None):
    M = pickle.load(open(POOL, "rb"))
    sents = {k: [] for k in range(len(M["titles"]))}
    for s, o in zip(M["sentences"], M["owner"]):
        sents[o].append(s)
    kus = {}
    for k, t in enumerate(M["titles"][:limit]):
        u = NeuronKU.__new__(NeuronKU)          # slots only: what holds() and labels() read
        u.slots = build_slots(t, sents[k])
        kus[k] = u
    return kus


@pytest.mark.skipif(not POOL.exists(), reason="test pool not built here")
def test_cache_normalization_equals_holds():
    kus = _kus()
    cache = NeuronHost.cache_rebuild(types.SimpleNamespace(kus=kus))
    rels = sorted(KIND)
    n = 0
    for k, u in kus.items():
        for s in u.slots:
            key = normalize(s.label)
            assert key in u.labels(), (k, s.label)
            if key:
                assert k in cache[key], (k, s.label, key)
            for r in rels:
                if u.holds(r, s.label):
                    assert key and k in cache[key], (k, r, s.label)
            n += 1
    for key, ids in cache.items():
        for k in ids:
            assert key in kus[k].labels(), (key, k)
    assert n > 0
