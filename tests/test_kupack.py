"""Security fix 2026-11-20: KU packages and data files cannot execute code. Known-good: a real KU survives the round
trip with identical parameters and answers. Known-bad: pickle payloads (torch.save blob, raw pickle, checkpoint file)
are refused by every entry point and the payload never runs; malformed packages are refused."""
import base64
import io
import json
import os
import pickle
import struct

import numpy as np
import pytest
import torch

from mxn.common import safeload
from mxn.neuron import kupack
from mxn.neuron.nku import Settings, build, package, unpackage
from tests.test_neuron_e2e import SENTS, TITLE, FakeEnc


class _Evil:
    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        return (os.system, (f"touch {self.marker}",))


def _ku():
    return build(1, TITLE, SENTS, FakeEnc(), Settings(hidden=8, epochs=5))


def test_round_trip_identical_package_and_answers():
    u = _ku()
    d = package(u); d["key"] = 7; d["conns"][next(iter(d["conns"]))]["tkey"] = 3
    (e,), extra = kupack.loads(kupack.dumps([d], {"tag": {"settings": {"hidden": 8}, "arm": "neuron"}}))
    assert extra == {"tag": {"settings": {"hidden": 8}, "arm": "neuron"}}
    assert {k: v for k, v in e.items() if k != "state"} == {k: v for k, v in d.items() if k != "state"}
    assert all(torch.equal(d["state"][n], e["state"][n]) for n in d["state"])
    v = unpackage(e)
    q = np.ones(16, dtype=np.float32)
    assert v.answer(q, "director") == u.answer(q, "director")


@pytest.mark.parametrize("make", ["torch_save", "pickle"])
def test_pickle_blob_refused_and_not_run(tmp_path, make):
    marker = tmp_path / "pwned"
    b = io.BytesIO()
    torch.save([_Evil(marker)], b) if make == "torch_save" else pickle.dump([_Evil(marker)], b)
    with pytest.raises(kupack.PackageError):
        kupack.loads(b.getvalue())
    with pytest.raises(pickle.UnpicklingError):
        safeload.pickle_loads(pickle.dumps(_Evil(marker)))
    f = tmp_path / "evil.pt"; torch.save({"x": _Evil(marker)}, f)
    with pytest.raises(Exception):
        safeload.torch_load(f)
    assert not marker.exists()


def test_host_entry_points_refuse_malicious_blob(tmp_path):
    from mxn.neuron.nhost import NeuronHost
    marker = tmp_path / "pwned"
    b = io.BytesIO(); torch.save([_Evil(marker)], b)
    blob = base64.b64encode(b.getvalue()).decode()
    h = NeuronHost.__new__(NeuronHost); h.cold = tmp_path / "cold"; h.cold_ids = {}
    with pytest.raises(kupack.PackageError):
        h.move_in(blob)
    with pytest.raises(kupack.PackageError):
        h.cold_put({"blob": blob, "gen": 1, "from": "h1", "i": 0})
    assert not marker.exists() and not any(h.cold.glob("*")) if h.cold.exists() else not marker.exists()


def _blob(head: dict, body: bytes = b"") -> bytes:
    hb = json.dumps(head).encode()
    return kupack.MAGIC + struct.pack("<Q", len(hb)) + hb + body


@pytest.mark.parametrize("head", [
    {"packages": [{"k": "1"}], "extra": {}},                                   # wrong type
    {"packages": [{"k": 1, "exec": "x"}], "extra": {}},                        # unknown field
    {"packages": [{"title": "x"}], "extra": {}},                               # no k
    {"packages": [{"k": 1, "state": ["w"]}], "extra": {}},                     # tensor not present
    {"packages": [], "extra": {}, "more": 1},                                  # bad header
])
def test_malformed_packages_refused(head):
    with pytest.raises(kupack.PackageError):
        kupack.loads(_blob(head))


def test_undeclared_tensor_refused():
    from safetensors.torch import save
    with pytest.raises(kupack.PackageError):
        kupack.loads(_blob({"packages": [{"k": 1}], "extra": {}}, save({"0/w": torch.zeros(2)})))
