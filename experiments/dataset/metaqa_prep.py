"""MetaQA 1-hop first version: sets, pools and planner targets (reports/design_metaqa.md; user decision 2026-11-04 §3).

KUs: one per WikiMovies document (knowledge_source/wiki.txt; 18,127 documents, each title unique). A KB head (movie
name) names a document when exactly one document has that title with a trailing "(...)" stripped (case-folded);
1,430 names belong to several documents (remakes etc.): questions about them are ambiguous and left out of dev / test.

Relation of a question template (template = question with the bracketed head replaced by "X"): over MetaQA train
1-hop questions whose head is a KB movie, the question's relation is the single KB relation r with (head, r, answer)
for every answer (else unlabelled); a template's relation = the majority label if it covers >= 90% of its labelled
train questions, else the template is dropped (counted).

Planner targets (train only): every train 1-hop movie-head question with a template relation: plan {template chain,
r1 = relation, r2 None, op None, spans [[a, b]] = the head in the question without brackets}; a seeded 10% hold-out
("mdev") for the planner check. Dev / test: the fitting subset (head is a movie named by exactly one document, one
gold answer) with a template relation; test = the MetaQA test split's such questions whose head heads no dev
question, first 2,000 in seeded order; dev = all such dev-split questions, halves dev-fit / dev-pick by seed.

Writes (2Wiki record format, so the neuron pipeline runs unchanged):
  DATA_OUT/dev.json (dev + test records: _id, question, answer, answer_id None, type = relation, context = the head's
  document, supporting_facts, evidences [[title, relation, answer]]), DATA_OUT/id_aliases.json (empty);
  EX_OUT/pool_mqdev.pkl, pool_mqtest.pkl (all documents; the set's questions), mq_fit.jsonl, mq_mdev.jsonl, and links to
  the shared language expert files of EXPERT_DIR; IDS_DIR/mq_{devfit,devpick,test}_ids.json, mq_sets.json.

usage: python metaqa_prep.py WIKIMOVIES_DIR METAQA_DIR EXPERT_DIR DATA_OUT EX_OUT IDS_DIR
"""
import collections, json, os, pickle, re, sys
from pathlib import Path

import numpy as np

wm, mq, ex, dout, xout, ids_dir = (Path(a) for a in sys.argv[1:7])
SEED, SPLIT_SEED, HOLD_SEED, N_TEST, MAJ = 20261105, 20261106, 20261107, 2000, 0.90
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
assert len(set(titles)) == len(titles)
by_name = collections.defaultdict(list)
for k, t in enumerate(titles):
    by_name[norm(strip(t))].append(k)
kb = [l.rstrip("\n").split("|") for l in open(mq / "kb" / "kb.txt", encoding="utf-8") if l.count("|") == 2]
movies = {norm(h) for h, r, v in kb}
rel_of = collections.defaultdict(set)
for h, r, v in kb:
    rel_of[(norm(h), norm(v))].add(r)


def read(split):
    out = []
    for k, line in enumerate(open(mq / "1-hop" / "vanilla" / f"qa_{split}.txt", encoding="utf-8")):
        q, _, a = line.rstrip("\n").partition("\t")
        m = re.search(r"\[([^\]]+)\]", q)
        text = q[:m.start()] + m.group(1) + q[m.end():]
        out.append({"k": k, "text": text, "span": [m.start(), m.start() + len(m.group(1))], "head": m.group(1),
                    "tpl": q[:m.start()] + "X" + q[m.end():], "answers": [x for x in a.split("|") if x]})
    return out


train = read("train")
lab = collections.defaultdict(collections.Counter)
for q in train:
    if norm(q["head"]) not in movies:
        continue
    rs = set.intersection(*[rel_of.get((norm(q["head"]), norm(a)), set()) for a in q["answers"]])
    lab[q["tpl"]][next(iter(rs)) if len(rs) == 1 else None] += 1
tpl_rel, dropped = {}, {}
for t, c in lab.items():
    lc = {r: n for r, n in c.items() if r}
    best = max(lc.items(), key=lambda x: x[1]) if lc else (None, 0)
    if best[0] and best[1] >= MAJ * sum(lc.values()):
        tpl_rel[t] = best[0]
    else:
        dropped[t] = dict(c)

plans = [{"id": f"mq1-train-{q['k']}", "type": tpl_rel[q["tpl"]], "question": q["text"],
          "plan": {"template": "chain", "r1": tpl_rel[q["tpl"]], "r2": None, "op": None, "spans": [q["span"]]},
          "reason": None}
         for q in train if norm(q["head"]) in movies and q["tpl"] in tpl_rel]
perm = np.random.default_rng(HOLD_SEED).permutation(len(plans)); cut = len(plans) // 10
hold = set(perm[:cut].tolist())
xout.mkdir(parents=True, exist_ok=True)
with open(xout / "mq_fit.jsonl", "w") as f:
    for i, p in enumerate(plans):
        if i not in hold:
            f.write(json.dumps(p) + "\n")
with open(xout / "mq_mdev.jsonl", "w") as f:
    for i, p in enumerate(plans):
        if i in hold:
            f.write(json.dumps(p) + "\n")
del train


def eligible(split):
    keep, why = [], collections.Counter()
    for q in read(split):
        docs = by_name.get(norm(q["head"]), [])
        if norm(q["head"]) not in movies or not docs:
            why["head_not_movie_doc"] += 1
        elif len(q["answers"]) != 1:
            why["several_answers"] += 1
        elif len(docs) > 1:
            why["head_names_several_documents"] += 1
        elif q["tpl"] not in tpl_rel:
            why["template_without_relation"] += 1
        else:
            keep.append({**q, "doc": docs[0], "split": split}); why["kept"] += 1
    return keep, dict(why)


dev, why_dev = eligible("dev")
test_all, why_test = eligible("test")
dev_heads = {q["doc"] for q in dev}
test_pool = [q for q in test_all if q["doc"] not in dev_heads]
order = np.random.default_rng(SEED).permutation(len(test_pool))
test = [test_pool[i] for i in order[:N_TEST]]
half = np.random.default_rng(SPLIT_SEED).permutation(len(dev))
qid = lambda q: f"mq1-{q['split']}-{q['k']}"
fit = [qid(dev[i]) for i in sorted(half[:len(dev) // 2])]
pick = [qid(dev[i]) for i in sorted(half[len(dev) // 2:])]

recs = [{"_id": qid(q), "question": q["text"], "answer": q["answers"][0], "answer_id": None, "type": tpl_rel[q["tpl"]],
         "context": [[titles[q["doc"]], sents[q["doc"]]]], "supporting_facts": [[titles[q["doc"]], 0]],
         "evidences": [[titles[q["doc"]], tpl_rel[q["tpl"]], q["answers"][0]]]} for q in dev + test]
dout.mkdir(parents=True, exist_ok=True)
json.dump(recs, open(dout / "dev.json", "w"))
open(dout / "id_aliases.json", "w").close()

flat = [s for ss in sents for s in ss]
owner = np.array([k for k, ss in enumerate(sents) for _ in ss], np.int32)
for name, qs in (("mqdev", dev), ("mqtest", test)):
    P = {"titles": titles, "sentences": flat, "owner": owner,
         "questions": [{"id": qid(q), "question": q["text"], "type": tpl_rel[q["tpl"]], "gold": [q["doc"]]} for q in qs]}
    pickle.dump(P, open(xout / f"pool_{name}.pkl", "wb"))
for f in ("enc", "tok16k.model", "country_table.json", "step7_dev.json"):
    if not (xout / f).exists():
        os.symlink((ex / f).resolve(), xout / f)
ids_dir.mkdir(parents=True, exist_ok=True)
for name, v in (("mq_devfit_ids", fit), ("mq_devpick_ids", pick), ("mq_test_ids", [qid(q) for q in test])):
    json.dump(v, open(ids_dir / f"{name}.json", "w"))
info = {"seeds": [SEED, SPLIT_SEED, HOLD_SEED], "documents": len(titles), "sentences": len(flat),
        "names_with_several_documents": sum(len(v) > 1 for v in by_name.values()),
        "templates": len(lab), "templates_with_relation": len(tpl_rel), "templates_dropped": dropped,
        "template_relations": collections.Counter(tpl_rel.values()),
        "planner_targets": {"fit": len(plans) - cut, "mdev": cut},
        "dev": why_dev, "test": why_test, "test_head_in_dev": len(test_all) - len(test_pool),
        "sets": {"devfit": len(fit), "devpick": len(pick), "test": len(test)},
        "types": {n: collections.Counter(tpl_rel[q["tpl"]] for q in qs) for n, qs in (("dev", dev), ("test", test))}}
json.dump(info, open(ids_dir / "mq_sets.json", "w"), indent=1)
print(json.dumps(info, indent=1)[:3000])
