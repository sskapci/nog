import asyncio
import threading

from mxn.expert.fhost import ADD, DROP, EDIT, FindClient, FindHost
from mxn.common.frames import serve

from tests.helpers import free_addr


def _host(units, addr):
    h = FindHost.__new__(FindHost)                                  # no extractor needed for these messages
    h.units, h.name, h.detached, h.expert_dir = units, "h0", {}, None
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    asyncio.run_coroutine_threadsafe(serve(addr, h.on_conn), loop).result()
    return h


def test_announce_names_only_and_follows_add_edit_drop():
    addr = free_addr()
    _host({0: ("Paris", ["Paris is a city."]), 2: ("Lyon", ["Lyon is a city."])}, addr)
    c = FindClient({"h0": addr}, {0: "h0", 2: "h0", 5: "h0"})
    m = c.announce("h0")
    assert m["host"] == "h0" and m["kus"] == [[0, "Paris"], [2, "Lyon"]]
    assert set(m) == {"rid", "host", "kus"}                         # no sentences, values or confidences
    c.send(5, ADD, title="Graz", sentences=["Graz is a city."])
    c.send(0, EDIT, sentences=["Paris is the capital."])
    c.send(2, DROP)
    assert c.announce("h0")["kus"] == [[0, "Paris"], [5, "Graz"]]
    assert "capital" not in str(c.announce("h0"))
    c.close()


def test_editv_applied_by_owning_host_matches_kuedit():
    import hashlib

    from mxn.expert.fhost import ATTACH, DETACH, EDITV
    from mxn.expert.kuedit import edit_ku_clean

    addr = free_addr()
    sents = ["Leo Mittler was an Austrian playwright.", "He left Austria in 1938."]
    h = _host({0: ("Leo Mittler", list(sents))}, addr)
    c = FindClient({"h0": addr}, {0: "h0"})
    forms = {"Austria": "Argentina", "Austrian": "Argentine"}
    r = c.send(0, EDITV, relation="country of citizenship", old="Austria", new="Argentina", mode="clean", forms=forms)
    want, n = edit_ku_clean(sents, forms, "country of citizenship")
    assert r["replaced"] == n == 2 and h.units[0][1] == want
    assert r["sha1"] == hashlib.sha1("\n".join(want).encode()).hexdigest()
    bad = c.send(0, EDITV, relation="place of birth", old="Argentina", new="X", mode="clean",
                 forms={"Argentina": "X"})                          # standalone, so this one applies
    assert "error" not in bad
    c.send(0, DETACH)
    assert 0 not in h.units and c.announce("h0")["kus"] == []
    c.send(0, ATTACH)
    assert h.units[0][0] == "Leo Mittler" and c.announce("h0")["kus"] == [[0, "Leo Mittler"]]
    c.close()


def test_resolve_and_via_messages():
    """RESOLVE: the host's own title index proposes and each KU scores itself; VIA: no connections yet → null."""
    import numpy as np

    from mxn.expert.fhost import RESOLVE, VIA
    from mxn.expert.hostindex import HostTitleIndex, KUTitle
    from mxn.expert.rgateway import HostsClient

    addr = free_addr()
    v = {0: np.array([1, 0, 0], np.float32), 2: np.array([0, 1, 0], np.float32)}
    h = _host({0: ("Paris", ["Paris is a city."]), 2: ("Paris (film)", ["A film."])}, addr)
    h.index = HostTitleIndex([KUTitle(k, h.units[k][0], v[k]) for k in (0, 2)])
    h.connections = {}
    c = HostsClient({"h0": addr})
    r = c.broadcast(RESOLVE, span="Paris", qv_span=[0, 1, 0], qv_question=[1, 0, 0])[0]
    assert r["exact"] == [[0, "Paris", 1.0]] and r["nodis"] == [[0, "Paris", 1.0], [2, "Paris (film)", 0.0]]
    assert r["fallback"] == [2, "Paris (film)", 1.0] and r["host"] == "h0"
    assert c.request("h0", VIA, k=0, relation="director", value="X")["target"] is None
    c.close()
