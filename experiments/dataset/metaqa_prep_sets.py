"""MetaQA set answers and person heads (reports/design_metaqa_sets_persons.md): sets, pools, gold plans, planner targets.

Template paths and categories as experiments/dataset/metaqa_scope.py (scope.json, train only): now / sets (movie head
naming one document, one / several gold answers, at most one reverse step), person / person_sets (value head, one
reverse step). A cell = hop count x category (10 cells; 3-hop has no person heads). Dev: the dev split's questions of a
cell, at most N_DEV in seeded order, halves dev-fit / dev-pick by seed. Test: the test split's questions of a cell whose
head heads no dev question of that hop count and that no earlier MetaQA test used (9985b mq_test_ids, 9987b
mq2 / mq3_test_ids: same line numbers), at most N_TEST in seeded order (fresh test sets).
Records (DATA_OUT/dev.json, 2Wiki record format plus "answers" = the gold set and "cat"): type =
"<cat>:<n>hop:<path>" with path steps "f:r" / "r:r". Evidences (oracle labels only): the KB triples along the path
(movies with one document), at most MAX_EV per question.
Plans: chain (1-hop movie head: r1), rev (movie head 2- / 3-hop: r1, r2[, r3]), prev (value head: r1 reverse[, r2
forward]); spans [the head]. Planner targets: every train question of the four categories (any answer count), seeded
10% hold-out. gold_plans.json: dry runs only.
usage: python metaqa_prep_sets.py WIKIMOVIES_DIR METAQA_DIR SCOPE.json EXPERT_DIR DATA_OUT EX_OUT IDS_DIR
"""
import collections, json, os, pickle, re, sys
from pathlib import Path

import numpy as np

wm, mq, scopef, ex, dout, xout, ids_dir = (Path(a) for a in sys.argv[1:8])
SEED, SPLIT_SEED, HOLD_SEED, N_DEV, N_TEST, MAX_EV = 20261112, 20261113, 20261114, 2000, 2000, 2000
CATS = ("now", "sets", "person", "person_sets")
norm = lambda s: re.sub(r"\s+", " ", s.lower()).strip()
strip = lambda t: re.sub(r"\s*\([^)]*\)$", "", t)

titles, sents, cur = [], [], None
for line in open(wm / "knowledge_source" / "wiki.txt", encoding="utf-8"):
    line = line.rstrip("\n")
    if not line.strip():
        cur = None; continue
    n, _, text = line.partition(" ")
    if n == "1":
        titles.append(text); sents.append([]); cur = sents[-1]
    elif cur is not None and text.strip():
        cur.append(text.strip())
by_name = collections.defaultdict(list)
for k, t in enumerate(titles):
    by_name[norm(strip(t))].append(k)
fwd = collections.defaultdict(lambda: collections.defaultdict(set)); rev = collections.defaultdict(lambda: collections.defaultdict(set))
raw = {}
for l in open(mq / "kb" / "kb.txt", encoding="utf-8"):
    p = l.rstrip("\n").split("|")
    if len(p) == 3:
        h, r, v = norm(p[0]), p[1], norm(p[2]); raw.setdefault(v, p[2]); raw.setdefault(h, p[0])
        fwd[h][r].add(v); rev[v][r].add(h)
TP = json.load(open(scopef))["template_paths"]
used = {i.split("-", 1)[0][-1] + "-" + i.split("-", 2)[2] for f in ("mq_test_ids", "mq2_test_ids", "mq3_test_ids")
        for i in json.load(open(ids_dir / f"{f}.json"))}           # "<n>-<line>" of earlier test questions


def read(hop, split):
    for k, line in enumerate(open(mq / hop / "vanilla" / f"qa_{split}.txt", encoding="utf-8")):
        q, _, a = line.rstrip("\n").partition("\t")
        m = re.search(r"\[([^\]]+)\]", q)
        yield {"k": k, "text": q[:m.start()] + m.group(1) + q[m.end():], "span": [m.start(), m.start() + len(m.group(1))],
               "head": m.group(1), "tpl": q[:m.start()] + "X" + q[m.end():], "answers": [x for x in a.split("|") if x]}


def category(hop, q):
    p = TP[hop].get(q["tpl"])
    if p is None or sum(s[0] == "r" for s in p) >= 2:
        return None, p
    if p[0][0] == "f":
        if len(by_name.get(norm(q["head"]), [])) != 1:
            return None, p
        return ("now" if len(q["answers"]) == 1 else "sets"), p
    return ("person" if len(q["answers"]) == 1 else "person_sets"), p


def plan(p, span):
    r = [s.split(":")[1] for s in p]
    if p[0][0] == "r":
        return {"template": "prev", "r1": r[0], "r2": r[1] if len(r) > 1 else None, "r3": None, "op": None,
                "spans": [span]}
    if len(p) == 1:
        return {"template": "chain", "r1": r[0], "r2": None, "r3": None, "op": None, "spans": [span]}
    return {"template": "rev", "r1": r[0], "r2": r[1], "r3": r[2] if len(r) > 2 else None, "op": None, "spans": [span]}


def evid(head, p):
    """KB triples (movie, relation, value) along the path; movies with one document only."""
    ev, S = [], {head}
    for s in p:
        d, r = s.split(":")
        if d == "f":
            T = set().union(*[fwd[m].get(r, set()) for m in S]) - {head}
            ev += [(m, r, v) for m in S for v in fwd[m].get(r, set()) if v in T]
        else:
            T = set().union(*[rev[v].get(r, set()) for v in S]) - {head}
            ev += [(m, r, v) for v in S for m in rev[v].get(r, set()) if m in T]
        S = T
        if len(ev) > MAX_EV:
            break
    t = lambda name: titles[by_name[name][0]] if len(by_name.get(name, [])) == 1 else None
    return [[t(m), r, raw.get(v, v)] for m, r, v in ev[:MAX_EV] if t(m)]


plans, info, gold_plans, recs = [], {"seeds": [SEED, SPLIT_SEED, HOLD_SEED], "n_dev": N_DEV, "n_test": N_TEST}, {}, []
pools = {"dev": [], "test": []}
ids = collections.defaultdict(list)
for hop in ("1-hop", "2-hop", "3-hop"):
    n = int(hop[0])
    for q in read(hop, "train"):
        c, p = category(hop, q)
        if c:
            plans.append({"id": f"mqs{n}-train-{q['k']}", "type": f"{c}:{n}hop:" + ">".join(p), "question": q["text"],
                          "plan": plan(p, q["span"]), "reason": None})
    cells = {s: collections.defaultdict(list) for s in ("dev", "test")}
    for split in ("dev", "test"):
        for q in read(hop, split):
            c, p = category(hop, q)
            if c:
                cells[split][c].append({**q, "cat": c, "path": p, "split": split, "n": n})
    dev_heads = {norm(q["head"]) for c in CATS for q in cells["dev"][c]}
    for ci, c in enumerate(CATS):
        dv = cells["dev"][c]
        ts = [q for q in cells["test"][c] if norm(q["head"]) not in dev_heads and f"{n}-{q['k']}" not in used]
        if not dv and not ts:
            continue
        dv = [dv[i] for i in sorted(np.random.default_rng(SEED + 10 * n + ci).permutation(len(dv))[:N_DEV])]
        ts_order = np.random.default_rng(SEED + 100 + 10 * n + ci).permutation(len(ts))
        ts = [ts[i] for i in ts_order[:N_TEST]]
        half = np.random.default_rng(SPLIT_SEED + 10 * n + ci).permutation(len(dv))
        qid = lambda q: f"mqs{q['n']}-{q['split']}-{q['k']}"
        cell = f"{n}h_{c}"
        ids[f"{cell}_devfit"] = [qid(dv[i]) for i in sorted(half[:len(dv) // 2])]
        ids[f"{cell}_devpick"] = [qid(dv[i]) for i in sorted(half[len(dv) // 2:])]
        ids[f"{cell}_test"] = [qid(q) for q in ts]
        for q in dv + ts:
            pl = plan(q["path"], q["span"])
            gold_plans[qid(q)] = {**pl, "conf": 1.0}
            doc = by_name[norm(q["head"])][0] if q["path"][0][0] == "f" else None
            recs.append({"_id": qid(q), "question": q["text"], "answer": " | ".join(q["answers"]), "answers": q["answers"],
                         "answer_id": None, "cat": c, "type": f"{c}:{n}hop:" + ">".join(q["path"]),
                         "context": [[titles[doc], sents[doc]]] if doc is not None else [],
                         "supporting_facts": [[titles[doc], 0]] if doc is not None else [],
                         "evidences": evid(norm(q["head"]), q["path"])})
            pools[q["split"]].append({"id": qid(q), "question": q["text"], "type": f"{c}:{n}hop:" + ">".join(q["path"]),
                                      "gold": [doc] if doc is not None else []})
        info[cell] = {"dev_eligible": len(cells["dev"][c]), "test_eligible": len(cells["test"][c]),
                      "test_head_in_dev": len(cells["test"][c]) - len([q for q in cells["test"][c]
                                                                       if norm(q["head"]) not in dev_heads]),
                      "test_used_before": sum(f"{n}-{q['k']}" in used for q in cells["test"][c]
                                              if norm(q["head"]) not in dev_heads),
                      "devfit": len(ids[f"{cell}_devfit"]), "devpick": len(ids[f"{cell}_devpick"]),
                      "test": len(ids[f"{cell}_test"])}
    print(hop, {k: v for k, v in info.items() if k.startswith(f"{n}h_")}, flush=True)

for part in ("devfit", "devpick", "test"):
    ids[part] = [i for k, v in sorted(ids.items()) if k.endswith("_" + part) for i in v]
perm = np.random.default_rng(HOLD_SEED).permutation(len(plans)); cut = len(plans) // 10
hold = set(perm[:cut].tolist())
xout.mkdir(parents=True, exist_ok=True)
with open(xout / "mqs_fit.jsonl", "w") as f, open(xout / "mqs_mdev.jsonl", "w") as g:
    for i, p in enumerate(plans):
        (g if i in hold else f).write(json.dumps(p) + "\n")
info["planner_targets"] = {"fit": len(plans) - cut, "mdev": cut,
                           "templates": collections.Counter(p["plan"]["template"] for p in plans)}
dout.mkdir(parents=True, exist_ok=True)
json.dump(recs, open(dout / "dev.json", "w"))
open(dout / "id_aliases.json", "w").close()
flat = [s for ss in sents for s in ss]
owner = np.array([k for k, ss in enumerate(sents) for _ in ss], np.int32)
for name, qs in pools.items():
    pickle.dump({"titles": titles, "sentences": flat, "owner": owner, "questions": qs},
                open(xout / f"pool_mqs{name}.pkl", "wb"))
for f in ("enc", "tok16k.model", "country_table.json", "step7_dev.json"):
    if not (xout / f).exists():
        os.symlink((ex / f).resolve(), xout / f)
json.dump(gold_plans, open(xout / "gold_plans.json", "w"))      # dry runs only (mechanics without the planner)
ids_dir.mkdir(parents=True, exist_ok=True)
for k, v in ids.items():
    json.dump(v, open(ids_dir / f"mqs_{k}_ids.json", "w"))
json.dump(info, open(ids_dir / "mqs_sets.json", "w"), indent=1)
print(json.dumps(info, indent=1))
