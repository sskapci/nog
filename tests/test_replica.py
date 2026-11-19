"""Primary-backup replication (chapter 3 step 1, design 69a315d): in-process hosts on free ports, stand-in extractor
(a KU's first sentence) and FakeEnc vectors."""
import asyncio
import subprocess
import sys
import threading
import time

import pytest

from mxn.expert.fhost import ADD, EDIT, EDITV
from mxn.expert.replica import (RepClient, RepGateway, RepHost, check_r, order, placement, replicas)
from mxn.common.frames import serve

from tests.helpers import free_addr
from tests.test_hostindex import FakeEnc

TITLES = ["Paris", "Paris (film)", "London", "Anne of Cleves", "Frank Sinatra", "The Heck with Hollywood!",
          "John Smith", "John Smith (actor)", "Mary Smith", "London Films", "Graz", "Porto"]
SPANS = ["Paris", "London", "Frank Sinatra", "Anne of Cleves", "paris film", "something else", "Hollywood", "John Smith",
         "Smith", "Graz", "film"]


def cluster(n=4, R=2, machines=None):
    enc = FakeEnc()
    names = [f"h{i}" for i in range(n)]
    machines = machines or {h: f"m{i}" for i, h in enumerate(names)}
    addrs = {h: free_addr() for h in names}
    hostlist = {h: {"addr": addrs[h], "machine": machines[h]} for h in names}
    place = placement(range(len(TITLES)), machines, R)
    vec = {k: enc.encode([f"{t}."], "passage")[0] for k, t in enumerate(TITLES)}
    H = {}
    for h in names:
        x = RepHost.__new__(RepHost)
        x.units = {k: (TITLES[k], [f"{TITLES[k]} value {k} v0."]) for k in place[h]}
        x.name, x.detached, x.expert_dir, x.enc, x.connections = h, {}, None, enc, {}
        x.find = lambda k, rel, x=x: (x.units[k][1][0], 1.0)
        x.setup(hostlist, R, {k: dict(m) for k, m in place[h].items()}, {k: vec[k] for k in place[h]})
        loop = asyncio.new_event_loop()
        threading.Thread(target=loop.run_forever, daemon=True).start()
        asyncio.run_coroutine_threadsafe(serve(addrs[h], x.on_conn), loop).result()
        H[h] = x
    c = RepClient(addrs, timeout=3.0)
    return H, c, RepGateway(None, None, enc, None, client=c, R=R)


def answer(G, span):
    e = G.resolve(span, f"Who is {span}?")
    if e is None:
        return None
    f = G.client.request(e["host"], 1, k=e["k"], relation="x")          # FIND
    return e["k"], e["rule"], f["value"], f.get("version")


def answers(G):
    return {s: answer(G, s) for s in SPANS}


def wait(cond, t=5.0):
    t0 = time.time()
    while time.time() - t0 < t:
        if cond():
            return True
        time.sleep(0.02)
    return False


def copies(H, k):
    return {h: x.meta[k]["role"] for h, x in H.items() if k in x.meta and x.alive}


# ---- placement
def test_placement_same_everywhere_from_the_id_alone():
    machines = {f"h{i}": f"m{i % 3}" for i in range(8)}
    code = ("import json,sys; from mxn.expert.replica import replicas; "
            "m=json.loads(sys.argv[1]); print(json.dumps({k: replicas(k, m, 2) for k in range(50)}))")
    import json
    a = json.loads(subprocess.run([sys.executable, "-c", code, json.dumps(machines)], capture_output=True, text=True).stdout)
    b = json.loads(subprocess.run([sys.executable, "-c", code, json.dumps(dict(reversed(list(machines.items()))))],
                                  capture_output=True, text=True).stdout)
    assert a == b == {str(k): replicas(k, machines, 2) for k in range(50)}
    for k in range(50):
        reps = replicas(k, machines, 3)
        assert len({machines[h] for h in reps}) == 3                    # different machines when enough exist
        assert reps[0] == order(k, list(machines))[0]                    # primary = first in the KU's order
    one = {f"h{i}": "m0" for i in range(4)}                               # one machine: labels repeat (reported)
    assert len(set(replicas(7, one, 2))) == 2


def test_refuses_fewer_than_two_copies():
    with pytest.raises(ValueError):
        check_r(1)
    with pytest.raises(ValueError):
        cluster(R=1)


# ---- normal operation
def test_exactly_one_copy_answers_and_resolve_work_equals_single_copy():
    H, c, G = cluster()
    for k in range(len(TITLES)):
        assert sorted(copies(H, k).values()) == ["backup", "primary"]
    a = answers(G)
    assert G.stats["dedupe_errors"] == 0 and G.stats["failover_rounds"] == 0
    assert sum(len(x.index.kus) for x in H.values()) == len(TITLES)       # indexed once: the single-copy work
    assert all(v is not None for v in a.values())
    c.close()


# ---- failover
def test_kill_one_host_every_answer_identical():
    H, c, G = cluster()
    before = answers(G)
    H["h1"].alive = False
    after = answers(G)
    assert after == before
    assert G.stats["failover_rounds"] == 1 and G.stats["promoted"] > 0 and G.stats["dedupe_errors"] == 0
    for k in range(len(TITLES)):                                          # re-replicated: R live copies, one primary
        assert wait(lambda: sorted(copies(H, k).values()) == ["backup", "primary"]), (k, copies(H, k))
    c.close()


def test_r2_both_copies_dead_degraded_mode_r3_still_answered():
    H, c, G = cluster(n=4, R=2)
    k = 0                                                                 # "Paris"
    for h in replicas(k, {h: x.hostlist[h]["machine"] for h, x in H.items()}, 2):
        H[h].alive = False
    assert answer(G, "Paris") is None or answer(G, "Paris")[0] != k
    assert G.resolve("something else", "?") is None                       # ≥ R hosts missing: degraded mode
    c.close()
    H, c, G = cluster(n=5, R=3)
    before = answers(G)
    for h in replicas(0, {h: x.hostlist[h]["machine"] for h, x in H.items()}, 3)[:2]:
        H[h].alive = False
    assert answer(G, "Paris") == before["Paris"]
    c.close()


def test_slow_primary_returning_after_failover_does_not_answer():
    H, c, G = cluster()
    k = 0
    p = next(h for h, x in H.items() if k in x.meta and x.meta[k]["role"] == "primary")
    before = answer(G, "Paris")
    H[p].alive = False                                                    # slow, not dead
    assert answer(G, "Paris")[:3] == before[:3]
    newp = next(h for h, x in H.items() if x.alive and k in x.meta and x.meta[k]["role"] == "primary")
    assert H[newp].meta[k]["epoch"] == 1
    H[p].alive = True                                                     # comes back with its old state
    assert answer(G, "Paris")[:3] == before[:3]                           # REJOIN first: it demoted or dropped its copy
    assert not (k in H[p].meta and H[p].meta[k]["role"] == "primary")
    roles = [x.meta[k]["role"] for x in H.values() if k in x.meta]
    assert roles.count("primary") == 1 and G.stats["dedupe_errors"] == 0
    c.close()


# ---- edits
def test_edits_no_request_sees_two_versions():
    H, c, G = cluster()
    k = 4                                                                 # "Frank Sinatra"
    seen = []
    for n in range(1, 6):
        r = c.manage(EDIT, k=k, sentences=[f"Frank Sinatra value {k} v{n}."])
        assert r["version"] == n
        for _ in range(3):
            replies = c.broadcast(1, k=k, relation="x")                    # FIND to every host: only the primary answers
            vals = {x["value"] for x in replies if not x.get("not_primary")}
            assert len(vals) == 1
            seen.append(vals.pop())
    assert seen[-1].endswith("v5.")
    assert wait(lambda: all(x.meta[k]["version"] == 5 for x in H.values() if k in x.meta))
    c.close()


def test_edit_with_backup_down_caught_up_on_return_and_served_after_failover():
    H, c, G = cluster()
    k = 4
    p = next(h for h, x in H.items() if k in x.meta and x.meta[k]["role"] == "primary")
    b = next(h for h, x in H.items() if k in x.meta and x.meta[k]["role"] == "backup")
    H[b].alive = False
    assert c.manage(EDIT, k=k, sentences=["Frank Sinatra value 4 new."])["version"] == 1
    time.sleep(0.2)
    assert H[b].meta[k]["version"] == 0                                   # missed it
    H[b].alive = True
    c.down.add(b)                                                         # the gateway saw it fail (as after a timeout)
    c.request(b, 18, k=k)                                                 # any request: REJOIN first
    assert H[b].meta[k]["version"] == 1 and H[b].units[k][1] == ["Frank Sinatra value 4 new."]
    H[p].alive = False
    assert answer(G, "Frank Sinatra")[2] == "Frank Sinatra value 4 new."
    c.close()


def test_lost_edit_when_primary_dies_before_forwarding_is_reported():
    H, c, G = cluster()
    k = 4
    p = next(h for h, x in H.items() if k in x.meta and x.meta[k]["role"] == "primary")
    H[p].hold_forward = True                                              # dies after applying, before forwarding
    r = c.manage(EDIT, k=k, sentences=["Frank Sinatra value 4 lost."])
    assert r["applied"] and r["version"] == 1
    H[p].alive = False
    assert answer(G, "Frank Sinatra")[2] == "Frank Sinatra value 4 v0."    # the backup's older version answers
    st = c.edit_status(k, r["version"])
    assert st["lost"] and st["current_version"] == 0 and st["primary"] != p
    c.close()


def test_add_places_copies_by_rendezvous_and_resolves():
    H, c, G = cluster()
    r = c.request("h2", ADD, title="Pequo Quoquozar", sentences=["Pequo Quoquozar was born in Graz."])
    k = r["k"]
    reps = replicas(k, {h: x.hostlist[h]["machine"] for h, x in H.items()}, 2)
    assert r["copies"] == reps and copies(H, k) == {reps[0]: "primary", reps[1]: "backup"}
    assert answer(G, "Pequo Quoquozar")[:3] == (k, "exact", "Pequo Quoquozar was born in Graz.")
    c.close()


def test_rereplication_restores_r_copies_with_one_primary():
    H, c, G = cluster(n=5)
    answers(G)
    H["h3"].alive = False
    answers(G)
    for k in range(len(TITLES)):
        assert wait(lambda: sorted(copies(H, k).values()) == ["backup", "primary"]), (k, copies(H, k))
        live = [h for h in copies(H, k)]
        assert "h3" not in live
    c.close()
