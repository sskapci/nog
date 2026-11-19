"""KU-owned connections with scores (chapter 3 step 2, design reports/design_connections.md): in-process replicated
hosts (tests/test_replica.py's cluster), stand-in extractor (a KU's first sentence is its value for every relation)."""
import time

from mxn.expert.fhost import DROP, EDIT, EDITV
from mxn.expert.replica import FEEDBACK, conn_score, value_sha1

from tests.test_replica import cluster, wait

SRC_FALLBACK, SRC_EXACT = 10, 11                     # "Graz" → "something else" (no title); "Porto" → "London" (exact)
TARGET = 5                                           # "The Heck with Hollywood!"


def setup(tau=0.6, half_life=None, n=4):
    H, c, G = cluster(n=n)
    for x in H.values():
        x.tau_c, x.half_life = tau, half_life
        for k, s in ((SRC_FALLBACK, "something else"), (SRC_EXACT, "London")):
            if k in x.units:
                x.units[k] = (x.units[k][0], [s])
    return H, c, G


def src(G, k):
    e = G.resolve({SRC_FALLBACK: "Graz", SRC_EXACT: "Porto"}[k], "?")
    assert e["k"] == k
    return G.find(e, "r1")


def feedback(c, k, value, target, outcome, t=0.0):
    return c.manage(FEEDBACK, k=k, relation="r1", value_sha1=value_sha1(value), target=target, outcome=outcome, t=t,
                    source="gold")


def holders(H, k):
    return {h: x for h, x in H.items() if k in x.meta and x.alive}


def test_success_creates_connection_on_primary_then_replicated_and_used():
    H, c, G = setup()
    before = G.hop_find(src(G, SRC_FALLBACK), "r1", "r2", "?")
    assert before["resolved_by"] == "fallback" and before["ku"] != TARGET
    r = feedback(c, SRC_FALLBACK, "something else", TARGET, "success")
    assert r["applied"] and r["version"] == 1
    p = next(x for x in holders(H, SRC_FALLBACK).values() if x.is_primary(SRC_FALLBACK))
    assert [x["target"] for x in p.connections[SRC_FALLBACK]] == [TARGET]
    assert wait(lambda: all(x.connections.get(SRC_FALLBACK) == p.connections[SRC_FALLBACK]
                            and x.meta[SRC_FALLBACK]["version"] == 1 for x in holders(H, SRC_FALLBACK).values()))
    f = G.hop_find(src(G, SRC_FALLBACK), "r1", "r2", "?")
    assert f["ku"] == TARGET and f["resolved_by"] == "connection b0" and f["conn_used"]
    assert f["resolve_k"] == before["ku"] and f["src"] == SRC_FALLBACK            # RESOLVE's choice still recorded
    assert G.stats["conn_accepted"] == 1
    c.close()


def test_without_tau_or_feedback_hops_are_unchanged():
    H, c, G = setup(tau=None)
    a = G.hop_find(src(G, SRC_FALLBACK), "r1", "r2", "?")
    feedback(c, SRC_FALLBACK, "something else", TARGET, "success")
    b = G.hop_find(src(G, SRC_FALLBACK), "r1", "r2", "?")                       # tau None: connections never used
    assert a["ku"] == b["ku"] and b["resolved_by"] == "fallback" and G.stats["conn_proposed"] == 0
    c.close()


def test_failure_lowers_score_below_tau_and_decay_lowers_old_scores():
    H, c, G = setup(tau=0.6, half_life=10.0)
    feedback(c, SRC_FALLBACK, "something else", TARGET, "success", t=0)
    G.clock = lambda: 0.0
    assert G.hop_find(src(G, SRC_FALLBACK), "r1", "r2", "?")["ku"] == TARGET   # score 2/3 at t = 0
    G.clock = lambda: 1000.0                                                     # 100 half-lives later: ≈ 1/2 < τ
    assert G.hop_find(src(G, SRC_FALLBACK), "r1", "r2", "?")["resolved_by"] == "fallback"
    G.clock = lambda: 0.0
    feedback(c, SRC_FALLBACK, "something else", TARGET, "failure", t=0)
    assert G.hop_find(src(G, SRC_FALLBACK), "r1", "r2", "?")["resolved_by"] == "fallback"   # 2/4 < τ
    cc = {"succ": 1.0, "fail": 0.0, "t": 0.0}
    assert conn_score(cc, 0.0, 10.0) == 2 / 3 and abs(conn_score(cc, 10.0, 10.0) - 1.5 / 2.5) < 1e-12
    c.close()


def test_unverified_or_misaddressed_feedback_changes_nothing():
    H, c, G = setup()
    assert feedback(c, SRC_FALLBACK, "something else", TARGET, "failure") is None
    assert all(not x.connections.get(SRC_FALLBACK) for x in H.values())          # a failure never creates one
    b = next(h for h, x in holders(H, SRC_FALLBACK).items() if not x.is_primary(SRC_FALLBACK))
    r = c.request(b, FEEDBACK, k=SRC_FALLBACK, relation="r1", value_sha1=value_sha1("something else"), target=TARGET,
                  outcome="success", t=0.0, source="gold")
    assert r["applied"] is False and r["not_primary"]
    assert all(not x.connections.get(SRC_FALLBACK) for x in H.values())
    a = G.hop_find(src(G, SRC_FALLBACK), "r1", "r2", "?")                          # answering alone never learns
    G.hop_find(src(G, SRC_FALLBACK), "r1", "r2", "?")
    assert all(not x.connections.get(SRC_FALLBACK) for x in H.values()) and a["resolved_by"] == "fallback"
    c.close()


def test_unique_exact_title_match_wins_and_conflict_is_logged():
    H, c, G = setup()
    feedback(c, SRC_EXACT, "London", TARGET, "success")
    f = G.hop_find(src(G, SRC_EXACT), "r1", "r2", "?")
    assert f["ku"] == 2 and f["resolved_by"] == "exact" and f["conn"][0] == TARGET and f["conn_conflict"]
    assert G.stats["conn_conflicts"] == 1 and G.stats["conn_accepted"] == 0
    c.close()


def test_guard_off_connection_used_and_difference_recorded():
    H, c, G = setup()
    G.exact_guard = False
    feedback(c, SRC_EXACT, "London", TARGET, "success")
    f = G.hop_find(src(G, SRC_EXACT), "r1", "r2", "?")
    assert f["ku"] == TARGET and f["conn_used"] and f["conn_exact_differs"] and f["resolve_k"] == 2
    assert not f.get("conn_conflict") and G.stats["conn_conflicts"] == 0 and G.stats["conn_accepted"] == 1
    c.close()


def test_dropped_or_dead_target_falls_back_to_resolve():
    H, c, G = setup()
    base = G.hop_find(src(G, SRC_FALLBACK), "r1", "r2", "?")
    feedback(c, SRC_FALLBACK, "something else", TARGET, "success")
    assert c.manage(DROP, k=TARGET)["applied"]
    f = G.hop_find(src(G, SRC_FALLBACK), "r1", "r2", "?")
    assert f["ku"] == base["ku"] and f["conn_check_failed"] and G.stats["conn_check_failed"] == 1
    c.close()


def test_edit_of_the_relation_resets_and_changed_value_is_not_proposed():
    H, c, G = setup()
    feedback(c, SRC_FALLBACK, "something else", TARGET, "success")
    p = next(x for x in holders(H, SRC_FALLBACK).values() if x.is_primary(SRC_FALLBACK))
    p.connections[SRC_FALLBACK].append({**p.connections[SRC_FALLBACK][0], "relation": "other"})
    for x in H.values():                                                           # stand-in for the value rewrite
        x.edit_value = lambda m: {"replaced": 1}
    assert c.manage(EDITV, k=SRC_FALLBACK, relation="r1", old="else", new="other", mode="v2") is not None
    assert [x["relation"] for x in p.connections[SRC_FALLBACK]] == ["other"]     # r1 reset, other relation kept
    assert wait(lambda: all(x.connections.get(SRC_FALLBACK) == p.connections[SRC_FALLBACK]
                            for x in holders(H, SRC_FALLBACK).values()))
    assert c.manage(EDIT, k=SRC_FALLBACK, sentences=["something else"])["applied"]  # all facts replaced: all reset
    assert not p.connections.get(SRC_FALLBACK)
    c.close()
    H, c, G = setup()
    feedback(c, SRC_FALLBACK, "a different value", TARGET, "success")             # learned for another value
    f = G.hop_find(src(G, SRC_FALLBACK), "r1", "r2", "?")
    assert f["resolved_by"] == "fallback" and G.stats["conn_proposed"] == 0
    c.close()


def test_failover_keeps_connections_and_stale_primary_never_serves_them():
    H, c, G = setup()
    feedback(c, SRC_FALLBACK, "something else", TARGET, "success")
    p = next(h for h, x in holders(H, SRC_FALLBACK).items() if x.is_primary(SRC_FALLBACK))
    assert wait(lambda: all(x.meta[SRC_FALLBACK]["version"] == 1 for x in holders(H, SRC_FALLBACK).values()))
    H[p].alive = False
    G.resolve("London", "?")                                                       # failover round
    f = G.hop_find(src(G, SRC_FALLBACK), "r1", "r2", "?")
    assert f["ku"] == TARGET and f["resolved_by"] == "connection b0"
    newp = next(h for h, x in holders(H, SRC_FALLBACK).items() if x.is_primary(SRC_FALLBACK))
    assert H[newp].meta[SRC_FALLBACK]["version"] == 1 and H[newp].meta[SRC_FALLBACK]["epoch"] == 1
    G.clock = lambda: 0.0
    feedback(c, SRC_FALLBACK, "something else", TARGET, "failure")                # learned on the new primary only
    H[p].alive = True                                                              # old primary back with old state
    f = G.hop_find(src(G, SRC_FALLBACK), "r1", "r2", "?")                          # REJOIN first: demoted, caught up
    assert not H[p].is_primary(SRC_FALLBACK) and f["resolved_by"] == "fallback"   # score 2/4 < τ from the new primary
    if SRC_FALLBACK in H[p].meta:
        assert wait(lambda: H[p].connections[SRC_FALLBACK] == H[newp].connections[SRC_FALLBACK])
    c.close()


def test_rereplicated_copy_carries_connections():
    H, c, G = setup(n=5)
    feedback(c, SRC_FALLBACK, "something else", TARGET, "success")
    assert wait(lambda: all(x.meta[SRC_FALLBACK]["version"] == 1 for x in holders(H, SRC_FALLBACK).values()))
    b = next(h for h, x in holders(H, SRC_FALLBACK).items() if not x.is_primary(SRC_FALLBACK))
    H[b].alive = False
    G.resolve("London", "?")
    assert wait(lambda: len(holders(H, SRC_FALLBACK)) == 2)
    new = next(h for h in holders(H, SRC_FALLBACK) if h != b and not H[h].is_primary(SRC_FALLBACK))
    assert wait(lambda: [x["target"] for x in H[new].connections.get(SRC_FALLBACK, [])] == [TARGET])
    c.close()


def test_gateway_holds_no_connection_state():
    H, c, G = setup()
    feedback(c, SRC_FALLBACK, "something else", TARGET, "success")
    G.hop_find(src(G, SRC_FALLBACK), "r1", "r2", "?")
    banned = {"connections", "conn", "scores", "gold", "targets"}
    assert not banned & set(vars(G)) and not banned & set(vars(G.client))
    c.close()


def test_only_the_primary_answers_via():
    from mxn.expert.fhost import VIA
    H, c, G = setup()
    feedback(c, SRC_FALLBACK, "something else", TARGET, "success")
    assert wait(lambda: all(x.connections.get(SRC_FALLBACK) for x in holders(H, SRC_FALLBACK).values()))
    for h, x in holders(H, SRC_FALLBACK).items():
        r = c.request(h, VIA, k=SRC_FALLBACK, relation="r1", value="something else", t=0.0)
        assert (r["target"] is not None) == x.is_primary(SRC_FALLBACK)
    c.close()


def test_correction_creates_the_verified_target_and_is_proposed():
    from mxn.expert.fhost import VIA
    H, c, G = setup()
    before = G.hop_find(src(G, SRC_FALLBACK), "r1", "r2", "?")
    assert before["ku"] != TARGET
    feedback(c, SRC_FALLBACK, "something else", before["ku"], "failure")       # the wrong hop: no connection
    assert all(not x.connections.get(SRC_FALLBACK) for x in H.values())
    r = feedback(c, SRC_FALLBACK, "something else", TARGET, "correction")
    assert r["applied"]
    p = next(h for h, x in holders(H, SRC_FALLBACK).items() if x.is_primary(SRC_FALLBACK))
    cc = H[p].connections[SRC_FALLBACK][0]
    assert (cc["target"], cc["succ"], cc["fail"], cc["corr"]) == (TARGET, 1.0, 0.0, 1.0)
    assert wait(lambda: all(x.connections.get(SRC_FALLBACK) == H[p].connections[SRC_FALLBACK]
                            for x in holders(H, SRC_FALLBACK).values()))
    v = c.request(p, VIA, k=SRC_FALLBACK, relation="r1", value="something else", t=0.0)
    assert v["target"][0] == TARGET and v["target"][3] == 1.0
    f = G.hop_find(src(G, SRC_FALLBACK), "r1", "r2", "?")
    assert f["ku"] == TARGET and f["conn"] == [TARGET, 2 / 3, 1.0] and f["resolve_s"] > 0 and "check_s" in f
    feedback(c, SRC_FALLBACK, "something else", TARGET, "success")             # a later success: corr unchanged
    assert (cc["succ"], cc["corr"]) == (2.0, 1.0)
    c.close()
