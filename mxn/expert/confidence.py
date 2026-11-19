"""Confidence c2 (abstention fix, decision 2026-10-27 §2; failed its own test rule: 87.0% vs 90%):
c2 = plan confidence × Π over FINDs p_ext × Π over FINDs a(kind, resolution rule); p_KU is not used."""


def kinds(rec):
    """(FIND, "entity <rule>" | "hop <rule>") for each FIND of a pipeline record."""
    er = {t: rule for t, rule in rec["entities"]}
    for f in rec["finds"]:
        rule = f.get("resolved_by") or er.get(f["title"], "exact")
        yield f, ("hop " if "resolved_by" in f else "entity ") + rule


def c2(rec, tab: dict) -> float:
    if rec["answer"] is None:
        return 0.0
    c = rec["plan"]["conf"]
    for f, k in kinds(rec):
        c *= f["p_ext"] * tab.get(k, tab["other"])
    return c
