"""The verifier's message rule for conn_run.py (experiments only; holds gold, never KU state). Design
reports/design_connections.md §5, §8 and the 2026-10-29 amendment (correction feedback)."""


def feedback_messages(rows: list[dict], gold: dict, b: int) -> list[dict]:
    """FEEDBACK messages for one traffic batch: rows = the driver's traffic rows of that batch, gold = qid → [(s, r, o)].
    A hop leaving a gold source KU is verifiable: success if it reached a gold target of that source, else failure,
    followed by a correction to the verified target when the gold names exactly one for that source and relation
    (that source alone if no gold triple has the hop's relation); otherwise a "correction_ambiguous" record (not sent)."""
    msgs = []
    for r in rows:
        g = {}
        for s, rel, o in gold[r["id"]]:
            g.setdefault(s, set()).add(o)
        for h in r["hops"]:
            if h["src"] not in g:
                continue
            m = {"k": h["src"], "relation": h["src_relation"], "value_sha1": h["value_sha1"], "target": h["ku"],
                 "outcome": "success" if h["ku"] in g[h["src"]] else "failure", "t": r["t"], "source": "gold",
                 "batch": b, "qid": r["id"]}
            msgs.append(m)
            if m["outcome"] == "failure":
                cand = {o for s, rel, o in gold[r["id"]] if s == h["src"] and rel == h["src_relation"]} or g[h["src"]]
                if len(cand) == 1:
                    msgs.append({**m, "target": next(iter(cand)), "outcome": "correction"})
                else:
                    msgs.append({**m, "target": None, "outcome": "correction_ambiguous"})
    return msgs
