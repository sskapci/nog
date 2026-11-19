"""MetaQA 2- and 3-hop (reverse lookup, reports/design_reverse_lookup.md): sets, pools and planner targets.

Template paths: experiments/dataset/metaqa_paths.py on train (paths_train.json). KUs and documents as metaqa_prep.py
(one KU per WikiMovies document, 18,127). A question is eligible when its head names exactly one document, it has one
gold answer and its template has a path. Test (per hop count): the test split's eligible questions whose head heads no
dev question of that hop count, first N_TEST in seeded order; dev = all eligible dev-split questions, halves dev-fit /
dev-pick by seed. Type of a record = "<n>hop:r1>r2[>r3]".
Evidences (oracle labels and scoring only): the KB triples along the path: (head, r1, v) for v in V; (m, r2, v) for the
movies m reached; (m, r3, a) for the answers a; titles = the documents of those movies (unique names only).
gold_plans.json (dev / test ids → their template-path plan): dry runs only, to test the mechanics apart from the planner.
Pools: pool_mq{2,3}{dev,test}.pkl and pool_mqmh{dev,test}.pkl (2- and 3-hop together; one run per arm).
Planner targets (train only): every train 2- / 3-hop movie-head question whose template has a path: {template rev, r1,
r2, r3 (None for 2-hop), op None, spans [head]}; seeded 10% hold-out.

usage: python metaqa_prep_mh.py WIKIMOVIES_DIR METAQA_DIR PATHS.json EXPERT_DIR DATA_OUT EX_OUT IDS_DIR
"""
import collections, json, os, pickle, re, sys
from pathlib import Path

import numpy as np

wm, mq, pathf, ex, dout, xout, ids_dir = (Path(a) for a in sys.argv[1:8])
SEED, SPLIT_SEED, HOLD_SEED, N_TEST = 20261108, 20261109, 20261110, 2000
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
        h, r, v = norm(p[0]), p[1], norm(p[2]); raw.setdefault(v, p[2])
        fwd[h][r].add(v); rev[v][r].add(h)
movies = set(fwd)
P = json.load(open(pathf))
tpl_path = {hop: {t: json.loads(v) for t, v in P[hop]["template_path"].items()} for hop in ("2-hop", "3-hop")}


def read(hop, split):
    out = []
    for k, line in enumerate(open(mq / hop / "vanilla" / f"qa_{split}.txt", encoding="utf-8")):
        q, _, a = line.rstrip("\n").partition("\t")
        m = re.search(r"\[([^\]]+)\]", q)
        out.append({"k": k, "text": q[:m.start()] + m.group(1) + q[m.end():], "span": [m.start(), m.start() + len(m.group(1))],
                    "head": m.group(1), "tpl": q[:m.start()] + "X" + q[m.end():], "answers": [x for x in a.split("|") if x]})
    return out


def evid(head, path, answer):
    ev = []
    V = fwd[head].get(path[0], set())
    ev += [(head, path[0], v) for v in V]
    M = set().union(*[rev[v].get(path[1], set()) for v in V]) - {head} if V else set()
    ev += [(m, path[1], v) for m in M for v in V if m in rev[v].get(path[1], set())]
    if len(path) == 3:
        ev += [(m, path[2], norm(answer)) for m in M if norm(answer) in fwd[m].get(path[2], set())]
    t = lambda name: titles[by_name[name][0]] if len(by_name.get(name, [])) == 1 else None
    return [[t(s), r, raw.get(v, v)] for s, r, v in ev if t(s)]


plans, info, gold_plans = [], {"seeds": [SEED, SPLIT_SEED, HOLD_SEED]}, {}
recs, pools = [], {}
for hop in ("2-hop", "3-hop"):
    n = int(hop[0]); tp = tpl_path[hop]
    for q in read(hop, "train"):
        if norm(q["head"]) in movies and q["tpl"] in tp:
            pth = tp[q["tpl"]]
            plans.append({"id": f"mq{n}-train-{q['k']}", "type": f"{n}hop:" + ">".join(pth), "question": q["text"],
                          "plan": {"template": "rev", "r1": pth[0], "r2": pth[1], "r3": pth[2] if n == 3 else None,
                                   "op": None, "spans": [q["span"]]}, "reason": None})

    def eligible(split):
        keep, why = [], collections.Counter()
        for q in read(hop, split):
            docs = by_name.get(norm(q["head"]), [])
            if norm(q["head"]) not in movies or not docs:
                why["head_not_movie_doc"] += 1
            elif len(q["answers"]) != 1:
                why["several_answers"] += 1
            elif len(docs) > 1:
                why["head_names_several_documents"] += 1
            elif q["tpl"] not in tp:
                why["template_without_path"] += 1
            else:
                keep.append({**q, "doc": docs[0], "split": split}); why["kept"] += 1
        return keep, dict(why)

    dev, why_dev = eligible("dev")
    test_all, why_test = eligible("test")
    dev_heads = {q["doc"] for q in dev}
    test_pool = [q for q in test_all if q["doc"] not in dev_heads]
    order = np.random.default_rng(SEED + n).permutation(len(test_pool))
    test = [test_pool[i] for i in order[:N_TEST]]
    half = np.random.default_rng(SPLIT_SEED + n).permutation(len(dev))
    qid = lambda q: f"mq{n}-{q['split']}-{q['k']}"
    fit = [qid(dev[i]) for i in sorted(half[:len(dev) // 2])]
    pick = [qid(dev[i]) for i in sorted(half[len(dev) // 2:])]
    for q in dev + test:
        pth = tp[q["tpl"]]
        gold_plans[qid(q)] = {"template": "rev", "r1": pth[0], "r2": pth[1], "r3": pth[2] if n == 3 else None,
                              "op": None, "spans": [q["span"]], "conf": 1.0}
        recs.append({"_id": qid(q), "question": q["text"], "answer": q["answers"][0], "answer_id": None,
                     "type": f"{n}hop:" + ">".join(pth), "context": [[titles[q["doc"]], sents[q["doc"]]]],
                     "supporting_facts": [[titles[q["doc"]], 0]], "evidences": evid(norm(q["head"]), pth, q["answers"][0])})
    pools[f"mq{n}dev"], pools[f"mq{n}test"] = dev, test
    ids_dir.mkdir(parents=True, exist_ok=True)
    for name, v in ((f"mq{n}_devfit_ids", fit), (f"mq{n}_devpick_ids", pick), (f"mq{n}_test_ids", [qid(q) for q in test])):
        json.dump(v, open(ids_dir / f"{name}.json", "w"))
    info[hop] = {"dev": why_dev, "test": why_test, "test_head_in_dev": len(test_all) - len(test_pool),
                 "sets": {"devfit": len(fit), "devpick": len(pick), "test": len(test)},
                 "types_test": collections.Counter(tp[q["tpl"]][-1] for q in test).most_common()}

perm = np.random.default_rng(HOLD_SEED).permutation(len(plans)); cut = len(plans) // 10
hold = set(perm[:cut].tolist())
xout.mkdir(parents=True, exist_ok=True)
with open(xout / "mqmh_fit.jsonl", "w") as f, open(xout / "mqmh_mdev.jsonl", "w") as g:
    for i, p in enumerate(plans):
        (g if i in hold else f).write(json.dumps(p) + "\n")
info["planner_targets"] = {"fit": len(plans) - cut, "mdev": cut}
dout.mkdir(parents=True, exist_ok=True)
json.dump(recs, open(dout / "dev.json", "w"))
open(dout / "id_aliases.json", "w").close()
flat = [s for ss in sents for s in ss]
owner = np.array([k for k, ss in enumerate(sents) for _ in ss], np.int32)
for name, qs in pools.items():
    n = name[2]
    pickle.dump({"titles": titles, "sentences": flat, "owner": owner,
                 "questions": [{"id": f"mq{n}-{q['split']}-{q['k']}", "question": q["text"],
                                "type": f"{n}hop:" + ">".join(tpl_path[f'{n}-hop'][q['tpl']]), "gold": [q["doc"]]}
                               for q in qs]}, open(xout / f"pool_{name}.pkl", "wb"))
for name in ("dev", "test"):                    # one run per arm serves both hop counts (scored apart by ids)
    qs = [q for n in "23" for q in pickle.load(open(xout / f"pool_mq{n}{name}.pkl", "rb"))["questions"]]
    pickle.dump({"titles": titles, "sentences": flat, "owner": owner, "questions": qs},
                open(xout / f"pool_mqmh{name}.pkl", "wb"))
for f in ("enc", "tok16k.model", "country_table.json", "step7_dev.json"):
    if not (xout / f).exists():
        os.symlink((ex / f).resolve(), xout / f)
json.dump(gold_plans, open(xout / "gold_plans.json", "w"))      # dry runs only (mechanics without the planner)
json.dump(info, open(ids_dir / "mqmh_sets.json", "w"), indent=1)
print(json.dumps(info, indent=1))
