"""Step 3 planner targets (rules in reports/experiment_log.md, "step 3 rules"): each 2Wiki question
converted automatically from its evidence triples into a plan in the fixed plan language, or marked
not clean with a reason. Writes plans_{fit,mdev,test}.jsonl and prints the clean counts.

A plan: {"template": chain|compare|bridge, "r1", "r2" (None for compare), "op" (None for chain),
"spans": [[start, end], ...] (character offsets of E1, E2 in the question; E1 first)}.

Version 2 (--v2, post hoc, decisions 2026-10-25): the entity span is the mention as it appears in
the question: when the title is not found, the question span with the highest token F1 against it
(F1 >= 0.5, one shared token of 3+ characters); every span is extended over a directly following
"(...)". Bridge root detection is unchanged. Writes plans2_{split}.jsonl.

usage: plans.py DATA_DIR EXPERT_DIR [--v2]
"""
import json, re, sys, unicodedata
from collections import Counter
from pathlib import Path

DATES = {"date of birth", "date of death", "publication date", "inception"}
OPS = ["earlier", "later", "larger", "smaller", "same", "different"]
MONTHS = {m: i + 1 for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                          "september", "october", "november", "december"])}


def fold(s):
    """Lowercase, accents and punctuation removed, spaces kept as single spaces (span comparison)."""
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c if c.isalnum() else " " for c in s if not unicodedata.combining(c))
    return " ".join(s.split())


def span_key(s):
    return fold(s).replace(" ", "")


def find_span(question, entity):
    """Character span of the entity in the question: exact (case-insensitive), then without a
    trailing "(...)", then after folding; None if not found."""
    q = question.lower()
    for e in (entity, re.sub(r"\s*\([^)]*\)\s*$", "", entity)):
        i = q.find(e.lower())
        if e and i >= 0:
            return [i, i + len(e)]
    # folded match: map folded characters back to question offsets
    chars, pos = [], []
    for i, c in enumerate(question):
        for d in unicodedata.normalize("NFKD", c.lower()):
            if unicodedata.combining(d):
                continue
            chars.append(d if d.isalnum() else " "); pos.append(i)
    fq = "".join(chars)
    for e in (entity, re.sub(r"\s*\([^)]*\)\s*$", "", entity)):
        fe = fold(e)
        m = re.search(r"(?<![a-z0-9])" + re.escape(fe).replace(r"\ ", r"\s+") + r"(?![a-z0-9])", fq) if fe else None
        if m:
            return [pos[m.start()], pos[m.end() - 1] + 1]
    return None


V2 = False


def fuzzy_span(question, entity):
    """v2: the question span (1-12 words) with the highest token F1 against the entity (with and
    without its trailing "(...)"); ties -> longer; None unless F1 >= 0.5 with a shared 3+ char token."""
    words = list(re.finditer(r"\S+", question))
    targets = [set(fold(e).split()) for e in (entity, re.sub(r"\s*\([^)]*\)\s*$", "", entity)) if fold(e)]
    best = (0.0, 0, None)
    for i in range(len(words)):
        for j in range(i, min(i + 12, len(words))):
            a, b = words[i].start(), words[j].end()
            while b > a and not (question[b - 1].isalnum() or question[b - 1] == ")"):
                b -= 1
            if re.search(r"['\u2019]s$", question[a:b]):
                b -= 2
            toks = fold(question[a:b]).split()
            if not toks:
                continue
            for t in targets:
                shared = set(toks) & t
                if not any(len(x) >= 3 for x in shared):
                    continue
                pr, rc = len(shared) / len(toks), len(shared) / len(t)
                f = 2 * pr * rc / (pr + rc)
                if (f, b - a) > best[:2]:
                    best = (f, b - a, [a, b])
    return best[2] if best[0] >= 0.5 else None


def mention(question, entity):
    """The entity's span: v1 rule; in v2, the fuzzy fallback and the "(...)" extension."""
    sp = find_span(question, entity)
    if not V2:
        return sp
    if sp is None:
        sp = fuzzy_span(question, entity)
    if sp is not None:
        m = re.match(r"\s*\([^)]*\)", question[sp[1]:])
        if m:
            sp = [sp[0], sp[1] + m.end()]
    return sp


def parse_date(s):
    """(year, month, day) from the formats 2Wiki uses; None if it does not parse."""
    s = s.strip().lower().replace(",", "")
    m = re.fullmatch(r"(\d{1,2}) ([a-z]+) (\d{1,4})", s) or None
    if m and m.group(2) in MONTHS:
        return (int(m.group(3)), MONTHS[m.group(2)], int(m.group(1)))
    m = re.fullmatch(r"([a-z]+) (\d{1,2}) (\d{1,4})", s)
    if m and m.group(1) in MONTHS:
        return (int(m.group(3)), MONTHS[m.group(1)], int(m.group(2)))
    m = re.fullmatch(r"([a-z]+) (\d{1,4})", s)
    if m and m.group(1) in MONTHS:
        return (int(m.group(2)), MONTHS[m.group(1)], 0)
    m = re.fullmatch(r"(\d{1,4})", s)
    return (int(m.group(1)), 0, 0) if m else None


def same(a, b):
    return a.lower() == b.lower()


def compare_op(r, question, answer, roots, values):
    """op for a comparison over roots (E1, E2) with values (lists, one per root); (op, reason)."""
    if answer.lower() in ("yes", "no"):
        return ("different" if "different" in question.lower() else "same"), None
    if r not in DATES:
        return None, "non-date relation with a non-yes/no answer"
    side = [k for k, e in enumerate(roots) if same(answer, e)]
    if len(side) != 1:
        return None, "answer is not one root"
    d = [[parse_date(v) for v in vs] for vs in values]
    if any(len(x) != 1 or x[0] is None for x in d):
        return None, "date does not parse"
    a, b = d[0][0], d[1][0]
    if a == b:
        return None, "equal dates"
    win_earlier = 0 if a < b else 1
    return ("earlier" if side[0] == win_earlier else "later"), None


def convert(r):
    """(plan, None) or (None, reason)."""
    q, ev, t = r["question"], r["evidences"], r["type"]
    if t in ("compositional", "inference"):
        if len(ev) != 2 or not same(ev[0][2], ev[1][0]):
            return None, "chain triples do not link"
        sp = mention(q, ev[0][0])
        if sp is None:
            return None, "entity not in question"
        return {"template": "chain", "r1": ev[0][1], "r2": ev[1][1], "op": None, "spans": [sp]}, None
    if t == "comparison":
        rels = {e[1] for e in ev}
        subs = list(dict.fromkeys(e[0] for e in ev))
        if len(rels) != 1 or len(subs) != 2:
            return None, "comparison is not one relation over two entities"
        rel = rels.pop()
        sps = [mention(q, s) for s in subs]
        if None in sps:
            return None, "entity not in question"
        if V2 and sps[0][0] < sps[1][1] and sps[1][0] < sps[0][1]:
            return None, "entity spans overlap"
        order = sorted(range(2), key=lambda k: sps[k][0])
        roots, sps = [subs[k] for k in order], [sps[k] for k in order]
        vals = [[e[2] for e in ev if e[0] == s] for s in roots]
        op, why = compare_op(rel, q, r["answer"], roots, vals)
        if op is None:
            return None, why
        return {"template": "compare", "r1": rel, "r2": None, "op": op, "spans": sps}, None
    if t == "bridge_comparison":
        first = [e for e in ev if find_span(q, e[0]) is not None]
        roots = list(dict.fromkeys(e[0] for e in first))
        if len(roots) != 2:
            return None, "bridge roots not found in question"
        r1s = {e[1] for e in ev if e[0] in roots}
        if len(r1s) != 1:
            return None, "bridge first hop is not one relation"
        hop1 = {s: [e[2] for e in ev if e[0] == s] for s in roots}
        rest = [e for e in ev if e[0] not in roots]
        r2s = {e[1] for e in rest}
        if len(r2s) != 1:
            return None, "bridge second hop is not one relation"
        sps = [mention(q, s) for s in roots]
        order = sorted(range(2), key=lambda k: sps[k][0])
        roots, sps = [roots[k] for k in order], [sps[k] for k in order]
        # second-hop values per root: triples whose subject is that root's first-hop value
        # (subjects can differ in spelling from the value, e.g. "George Blair (director)": fall back to
        # the evidence order, first-hop i ↔ second-hop i)
        vals = []
        for s in roots:
            vs = [e[2] for e in rest if any(same(e[0], v) or same(re.sub(r"\s*\([^)]*\)$", "", e[0]), v) for v in hop1[s])]
            vals.append(vs)
        if any(not v for v in vals):
            h1 = [e for e in ev if e[0] in roots]
            if len(h1) == 2 and len(rest) == 2:
                vals = [[rest[[e[0] for e in h1].index(s)][2]] for s in roots]
            else:
                return None, "second hop does not link"
        op, why = compare_op(next(iter(r2s)), q, r["answer"], roots, vals)
        if op is None:
            return None, why
        return {"template": "bridge", "r1": r1s.pop(), "r2": r2s.pop(), "op": op, "spans": sps}, None
    return None, f"type {t}"


def plan_text(q, p):
    """Readable plan (the canonical form, spans as their question text)."""
    E = [q[a:b] for a, b in p["spans"]]
    if p["template"] == "chain":
        return f"FIND({E[0]}, {p['r1']})→v1; FIND(v1, {p['r2']})→v2; ANSWER(v2)"
    if p["template"] == "compare":
        return f"FIND({E[0]}, {p['r1']})→v1; FIND({E[1]}, {p['r1']})→v2; COMPARE(v1, v2, {p['op']}); ANSWER"
    return (f"FIND({E[0]}, {p['r1']})→v1; FIND({E[1]}, {p['r1']})→v2; FIND(v1, {p['r2']})→v3; "
            f"FIND(v2, {p['r2']})→v4; COMPARE(v3, v4, {p['op']}); ANSWER")


if __name__ == "__main__":
    V2 = "--v2" in sys.argv
    data, exp = Path(sys.argv[1]), Path(sys.argv[2])
    prefix = "plans2" if V2 else "plans"
    train = {r["_id"]: r for r in json.load(open(data / "train.json"))}
    dev = {r["_id"]: r for r in json.load(open(data / "dev.json"))}
    splits = {"fit": [train[i] for i in open(exp / "fit_ids.txt").read().split()],
              "mdev": [train[i] for i in open(exp / "mdev_ids.txt").read().split()]}
    pool = __import__("pickle").load(open(exp / "pool_test.pkl", "rb"))
    splits["test"] = [dev[x["id"]] for x in pool["questions"]]
    summary = {}
    for name, recs in splits.items():
        ok, why, by = 0, Counter(), Counter()
        with open(exp / f"{prefix}_{name}.jsonl", "w") as f:
            for r in recs:
                p, reason = convert(r)
                by[(r["type"], p is not None)] += 1
                if p:
                    ok += 1
                else:
                    why[reason] += 1
                f.write(json.dumps({"id": r["_id"], "type": r["type"], "question": r["question"], "plan": p,
                                    "reason": reason}) + "\n")
        types = sorted({t for t, _ in by})
        summary[name] = {"n": len(recs), "clean": ok,
                         "by_type": {t: [by[(t, True)], by[(t, True)] + by[(t, False)]] for t in types},
                         "not_clean": dict(why.most_common())}
        print(name, json.dumps(summary[name]), flush=True)
    json.dump(summary, open(exp / f"{prefix}_summary.json", "w"), indent=1)
