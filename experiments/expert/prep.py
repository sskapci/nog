"""Step 2 data: retrieval pools (test = the Phase 3 build's KUs, rebuilt from the same contexts in the
same order; mdev = the mdev questions' own contexts) and training pairs from the fit split.

Pool: titles, sentences (stripped, deduplicated per title, as p3_build.py), owner (title index),
questions (id, text, type, gold = title index per supporting fact). Pairs (fit only), each with its
hard-negative candidates (the question's other context sentences; BM25 picks one at training time):
  q   question ↔ each supporting sentence
  t   "<relation> of <subject>" ↔ the supporting sentence in the subject's paragraph
  ti  title ↔ up to 3 of its sentences (sampled, rng 20261202; each title once)

usage: prep.py DATA_DIR OUT_DIR
"""
import json, pickle, sys
from pathlib import Path

import numpy as np

data, out = Path(sys.argv[1]), Path(sys.argv[2])


def pool(records):
    sents: dict[str, list[str]] = {}
    for r in records:
        for title, ss in r["context"]:
            have = sents.setdefault(title, [])
            for s in ss:
                s = s.strip()
                if s and s not in have:
                    have.append(s)
    titles = [t for t in sents if sents[t]]
    tid = {t: i for i, t in enumerate(titles)}
    flat = [s for t in titles for s in sents[t]]
    owner = np.array([tid[t] for t in titles for _ in sents[t]], np.int32)
    qs = [{"id": r["_id"], "question": r["question"], "type": r["type"],
           "gold": [tid[t] for t, _ in r["supporting_facts"] if t in tid]} for r in records]
    return {"titles": titles, "sentences": flat, "owner": owner, "questions": qs}


dev = json.load(open(data / "dev.json"))
test = [dev[i] for i in np.random.default_rng(20261103).permutation(len(dev))[:2000]]
del dev
train = json.load(open(data / "train.json"))
tune = [train[i] for i in np.random.default_rng(20261104).permutation(len(train))[:500]]
P = pool(test + tune)
P["questions"] = P["questions"][:2000]
print(f"test pool: {len(P['titles'])} KUs, {len(P['sentences'])} sentences (Phase 3: 14,128 / 50,587)")
pickle.dump(P, open(out / "pool_test.pkl", "wb"))
ids = {n: set((out / f"{n}_ids.txt").read_text().split()) for n in ("mdev", "fit")}
M = pool([r for r in train if r["_id"] in ids["mdev"]])
print(f"mdev pool: {len(M['titles'])} KUs, {len(M['sentences'])} sentences, {len(M['questions'])} questions")
pickle.dump(M, open(out / "pool_mdev.pkl", "wb"))

rng = np.random.default_rng(20261202)
pairs, kinds, seen_titles = [], {"q": 0, "t": 0, "ti": 0}, set()
for r in train:
    if r["_id"] not in ids["fit"]:
        continue
    ctx = {t: [s.strip() for s in ss] for t, ss in r["context"]}
    sup = {(t, i) for t, i in r["supporting_facts"] if t in ctx and i < len(ctx[t])}
    negs = [(t, s) for t, ss in ctx.items() for i, s in enumerate(ss) if (t, i) not in sup and s]
    for t, i in sorted(sup):
        pairs.append(("q", r["question"], t, ctx[t][i], negs)); kinds["q"] += 1
    for subj, rel, _obj in r["evidences"]:
        hit = [(t, i) for t, i in sup if t.lower() == subj.lower()]
        if hit:
            t, i = hit[0]
            pairs.append(("t", f"{rel} of {subj}", t, ctx[t][i], negs)); kinds["t"] += 1
    for t, ss in ctx.items():
        if t in seen_titles:                    # each title once across the fit split
            continue
        seen_titles.add(t)
        for i in rng.permutation(len(ss))[:3]:
            if ss[i]:
                pairs.append(("ti", t, t, ss[i], [])); kinds["ti"] += 1
print("fit pairs:", kinds, len(pairs))
pickle.dump(pairs, open(out / "pairs_fit.pkl", "wb"))
