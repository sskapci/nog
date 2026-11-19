"""Degraded mode (decision 2026-10-28 Part A2): with a host missing from a RESOLVE broadcast, only a unique exact-title
match among the replies is accepted; ties and weaker rules abstain. With every host answering, nothing changes."""
from mxn.expert.fhost import ADD

from tests.helpers import free_addr
from tests.test_kuops import hosts

SPANS = ("Paris", "London", "Frank Sinatra", "Anne of Cleves", "paris film", "something else", "Hollywood")


def kill(c, name):
    """Point the client at a port nobody listens on: every request to `name` is refused."""
    c.hosts[name] = free_addr(); c.conns.pop(name, None)


def resolved(G, span):
    e = G.resolve(span, f"Who is {span}?")
    return None if e is None else (e["k"], e["rule"], e["host"])


def test_all_hosts_up_degraded_changes_nothing():
    _, _, c, G = hosts()
    G2 = type(G)(None, None, G.enc, None, client=c, degraded=False)
    for span in SPANS:
        assert resolved(G, span) == resolved(G2, span)
    assert G.stats["resolve_missing_reply"] == 0 and G.stats["degraded_abstained"] == {}
    c.close()


def test_host_down_unique_exact_kept_weaker_rules_abstain():
    _, _, c, G = hosts()
    before = {s: resolved(G, s) for s in SPANS}
    assert before["Paris"] == (0, "exact", "h0")
    kill(c, "h0")
    assert resolved(G, "London") == before["London"]                     # unique exact on a live host: kept
    assert resolved(G, "Frank Sinatra") == before["Frank Sinatra"]
    assert resolved(G, "Paris") is None                                  # would be "Paris (film)" by no_disambiguator
    assert resolved(G, "something else") is None                         # vector fallback: abstain
    G2 = type(G)(None, None, G.enc, None, client=c, degraded=False)      # known-bad: without degraded mode
    assert resolved(G2, "Paris") == (1, "no_disambiguator", "h1")        # confidently the wrong KU
    assert G.stats["degraded_abstained"] == {"no_disambiguator": 1, "fallback": 1}
    c.close()


def test_host_down_exact_tie_abstains():
    _, _, c, G = hosts()
    assert c.request("h1", ADD, title="London", sentences=["London is in Ontario."])["applied"]
    assert resolved(G, "London")[1] == "exact_tie"
    kill(c, "h0")
    assert resolved(G, "London") is None                                 # ambiguous name: h0 might hold another
    assert G.stats["degraded_abstained"] == {"exact_tie": 1}
    c.close()


def test_host_down_does_not_crash_find():
    _, _, c, G = hosts()
    e = G.resolve("Anne of Cleves", "Who is Anne of Cleves?")
    kill(c, e["host"])
    assert G.find(e, "x")["value"] == ""                                 # owner gone: empty value, no exception
    c.close()


def test_gateway_record_gives_c2_the_entity_rules():
    """c2 looks an entity's resolution rule up by title (mxn/expert/confidence.kinds), as the pipeline records it.
    9931 found the compliant gateway recording KU ids there: every lookup missed and defaulted to "exact"."""
    from mxn.expert.confidence import kinds
    _, _, c, G = hosts()
    plan = {"template": "chain", "r1": "x", "r2": "y", "op": None, "spans": [[0, 14]], "conf": 1.0}
    rec = G.answer("something else director?", plan)
    assert rec["entities"] == [[rec["finds"][0]["title"], "fallback"]]
    assert next(kinds(rec))[1] == "entity fallback"
    c.close()
