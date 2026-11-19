"""Add / drop / detach / attach / move on the compliant design (no placement map): owners are found by broadcast,
new KUs are named by their host, and resolution follows every change."""
import asyncio
import threading

import numpy as np

from mxn.expert.fhost import ADD, ATTACH, DETACH, DROP, MOVE, FindHost
from mxn.expert.hostindex import HostTitleIndex, KUTitle, id_key
from mxn.expert.rgateway import HostsClient, ResolveGateway
from mxn.common.frames import serve

from tests.helpers import free_addr
from tests.test_hostindex import FakeEnc

TITLES = ["Paris", "Paris (film)", "London", "Anne of Cleves", "Frank Sinatra", "The Heck with Hollywood!"]
NAMES = ("h0", "h1", "h2")


def hosts():
    enc = FakeEnc()
    out, addrs = {}, {}
    for i, name in enumerate(NAMES):
        h = FindHost.__new__(FindHost)
        h.units = {k: (t, [f"{t} value {k}."]) for k, t in enumerate(TITLES) if k % 3 == i}
        h.name, h.detached, h.expert_dir, h.enc, h.connections = name, {}, None, enc, {}
        h.index = HostTitleIndex([KUTitle(k, t, enc.encode([f"{t}."], "passage")[0]) for k, (t, _) in h.units.items()])
        h.find = lambda k, rel, h=h: (h.units[k][1][0], 1.0)                 # stand-in extractor: first sentence
        addrs[name] = free_addr()
        loop = asyncio.new_event_loop()
        threading.Thread(target=loop.run_forever, daemon=True).start()
        asyncio.run_coroutine_threadsafe(serve(addrs[name], h.on_conn), loop).result()
        out[name] = h
    client = HostsClient(addrs)
    return out, addrs, client, ResolveGateway(None, None, enc, None, client=client)


def answer(G, span):
    e = G.resolve(span, f"Who is {span}?")
    return (e["k"], e["rule"], G.find(e, "x")["value"]), e["host"]


def test_add_then_resolves_drop_then_never():
    H, addrs, c, G = hosts()
    r = c.request("h1", ADD, title="Pequo Quoquozar", sentences=["Pequo Quoquozar was born in Graz."])
    assert r["applied"] and r["k"] == "h1:1"
    (k, rule, v), host = answer(G, "Pequo Quoquozar")
    assert (k, rule, host) == ("h1:1", "exact", "h1") and v == "Pequo Quoquozar was born in Graz."
    owner = c.manage(DROP, k="h1:1")
    assert owner["host"] == "h1"
    for span in ("Pequo Quoquozar", "pequo quoquozar", "Pequo", "Quoquozar Pequo", "something else"):
        assert answer(G, span)[0][0] != "h1:1"
    assert c.manage(DROP, k="h1:1") is None                               # nobody holds it any more
    assert "h1:1" not in H["h1"].units and "h1:1" not in H["h1"].index.ids
    c.close()


def test_detach_attach_round_trip():
    H, addrs, c, G = hosts()
    before = answer(G, "Frank Sinatra")
    k = before[0][0]
    assert c.manage(DETACH, k=k)["host"] == before[1]
    assert answer(G, "Frank Sinatra")[0][0] != k
    assert c.manage(ATTACH, k=k)["host"] == before[1]
    assert answer(G, "Frank Sinatra") == before
    c.close()


def test_move_gives_identical_answers():
    H, addrs, c, G = hosts()
    spans = TITLES + ["Paris", "Hollywood", "Sinatra", "nothing at all"]
    before = [answer(G, s) for s in spans]
    k = TITLES.index("Frank Sinatra"); src = before[TITLES.index("Frank Sinatra")][1]
    dst = "h2" if src != "h2" else "h0"
    vec = H[src].index.kus[k].vec.copy()
    owner = c.manage(MOVE, k=k, to=addrs[dst], to_name=dst)
    assert owner["host"] == src and owner["moved_to"] == addrs[dst]
    assert k not in H[src].units and k in H[dst].units
    assert np.array_equal(H[dst].index.kus[k].vec, vec)                 # the KU's own vector travelled with it
    after = [answer(G, s) for s in spans]
    assert [a[0] for a in after] == [b[0] for b in before]               # same KU, rule and value for every span
    assert after[TITLES.index("Frank Sinatra")][1] == dst
    c.close()


def test_host_named_ids_unique_and_ordered():
    H, addrs, c, G = hosts()
    a = c.request("h0", ADD, title="X one", sentences=["X one."])["k"]
    b = c.request("h2", ADD, title="X two", sentences=["X two."])["k"]
    a2 = c.request("h0", ADD, title="X three", sentences=["X three."])["k"]
    assert len({a, b, a2}) == 3 and a == "h0:1" and a2 == "h0:2" and b == "h2:1"
    assert sorted([b, 3, a, 0], key=id_key) == [0, 3, "h0:1", "h2:1"]
    c.close()


def test_ids_unique_across_restart(tmp_path):
    """A host continues from the highest host-named id it holds and from its persisted counter."""
    from mxn.expert.fhost import FindHost

    def host(units, state):
        h = FindHost.__new__(FindHost)
        h.units, h.detached, h.name, h.index, h.state_dir = dict(units), {}, "h7", None, state
        return h

    h = host({0: ("A", ["A."])}, str(tmp_path))
    ids = [h.new_id() for _ in range(3)]                                 # h7:1, h7:2, h7:3
    for k in ids:
        h.units[k] = ("t", ["s."])
    del h.units["h7:3"]                                                 # the highest one is dropped before the restart
    h2 = host(h.units, str(tmp_path))                                   # restart: same KUs, same state directory
    more = [h2.new_id() for _ in range(2)]
    assert ids == ["h7:1", "h7:2", "h7:3"] and more == ["h7:4", "h7:5"]  # h7:3 is not reused
    h3 = host({"h7:9": ("t", ["s."])}, None)                            # no state: continue from the highest held
    assert h3.new_id() == "h7:10"


def test_move_interrupted_after_confirm_is_detected_and_repaired():
    """The source dies after the destination confirmed the IMPORT but before deleting its copy: both hold the KU.
    manage() refuses (duplicate owner); repair_duplicate keeps the destination's copy and drops the source's."""
    import json as _json

    import pytest

    from mxn.expert.fhost import DETACH, IMPORT
    from mxn.common.frames import frame, open_conn, read_frame

    H, addrs, c, G = hosts()
    k = TITLES.index("Frank Sinatra")
    before = answer(G, "Frank Sinatra")
    src = before[1]; dst = "h2" if src != "h2" else "h0"

    async def crashing_move_out(kk, to, h=H[src]):
        reader, writer = await open_conn(to)
        writer.write(frame(IMPORT, _json.dumps({"rid": 0, "package": h.package(kk)}).encode())); await writer.drain()
        await read_frame(reader)                                         # destination confirmed
        writer.close()
        raise RuntimeError("source died before deleting its copy")       # the handler dies: no reply, no delete
    H[src].move_out = crashing_move_out
    assert c.manage(MOVE, k=k, to=addrs[dst], to_name=dst) is None                   # the owner never answered
    assert k in H[src].units and k in H[dst].units                      # duplicate
    e = G.resolve("Frank Sinatra", "Who is Frank Sinatra?")
    assert e["k"] == k and e["rule"] == "exact_tie"                      # visible in resolution too (same KU twice)
    with pytest.raises(RuntimeError, match="several hosts"):
        c.manage(DETACH, k=k)                                            # the duplicate-owner check
    rep = c.repair_duplicate(k)
    assert rep["resolved"] and rep["kept"] == dst and rep["action"] == f"dropped the source copy on {src}"
    assert k not in H[src].units and k in H[dst].units
    after = answer(G, "Frank Sinatra")
    assert after[0] == before[0] and after[1] == dst
    c.close()
