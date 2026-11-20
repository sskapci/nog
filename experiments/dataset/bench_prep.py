"""Forgetting and updating benchmarks on demo corpus A: plans and question sets (reports/design_forget_edit.md).

Fictional companies (synthetic data generated for this demo). Any resemblance to real companies or people is
unintended.

usage: python bench_prep.py A3_DIR DATA_DIR EX_DIR IDS_DIR OUT_DIR
Reads a3 (pool.pkl, kb.json), the adapter's dev.json and gold_plans.json, the biz ids. Writes OUT_DIR:
- pool_all.pkl: the 3,000 company documents + every dev and test question + the synthetic edit questions
  (gold = head document for company heads, as demo_prep.py).
- data/dev.json: dev.json records + synthetic records (same schema); data/id_aliases.json (empty).
- gold_plans.json: gold plans of every question (adapter's + synthetic).
- forget_plan.json: nested forget sets (1 / 5 / 10%: 30 / 150 / 300 companies, seed FORGET_SEED) with, per forgotten
  KU: title, values stated only in forgotten documents (audit "unique labels"), sentence sha1 (audit); question
  classes per set (forget: a path company is forgotten; retain: none is).
- edit_plan.json: 500 edits (seed EDIT_SEED) on distinct companies {k, title, rel, old, new, sentences}, and per question
  of the edit run: gold before, gold after (from the edited KB), class (efficacy, generalization, specificity_own,
  specificity_neighbour, multihop, other).
- questions_all.json (forget runs), questions_edit.json (edit runs): question ids for e2e_run --questions.
Plans: gold plans (main arm, labelled "gold plans").
"""
from __future__ import annotations

import copy
import hashlib
import json
import pickle
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mxn.common.safeload import pickle_load  # no code execution from data files (2026-11-20)
from experiments.dataset import demo_gen as G                      # noqa: E402

FORGET_SEED, EDIT_SEED = 20261121, 20261122
FORGET_SIZES = (30, 150, 300)
N_EDITS = 500
EDIT_RELS = ("ceo", "headquartered_in", "industry", "product", "founded_by", "founded_year")


def answers(kb: dict, typ: str, head: str) -> list[str]:
    """Gold set of a question type ("<cat>:<n>hop:<path>", path steps joined by ">") from a KB."""
    path = typ.split(":", 2)[2].split(">")
    st = [(s[0], s[2:]) for s in path]
    if st[0][0] == "f":
        if st == [("f", "subsidiary"), ("f", "product")]:
            cs = [c for c, f in kb.items() if head in f["parent"]]
            return sorted({v for c in cs for v in kb[c]["product"]})
        vals = list(kb[head][st[0][1]])
        if len(st) == 2:
            vals = [v for m in vals for v in kb[m][st[1][1]]]
        return sorted(set(vals))
    cs = [c for c, f in kb.items() if head in f[st[0][1]]]
    if len(st) == 2:
        return sorted({v for c in cs for v in kb[c][st[1][1]]})
    return sorted(set(cs))


def main(a3: Path, data: Path, ex: Path, ids: Path, out: Path) -> None:
    P = pickle_load(a3 / "pool.pkl")
    kb = json.load(open(a3 / "kb.json"))["kb"]
    D = {r["_id"]: r for r in json.load(open(data / "dev.json"))}
    GP = json.load(open(ex / "gold_plans.json"))
    titles = P["titles"]; tix = {t: i for i, t in enumerate(titles)}
    docs = {t: [] for t in titles}
    for s, o in zip(P["sentences"], P["owner"]):
        docs[titles[o]].append(s)
    qids = sorted(D)
    out.mkdir(parents=True, exist_ok=True); (out / "data").mkdir(exist_ok=True)

    # ---- forget plan
    comps = sorted(kb); random.Random(FORGET_SEED).shuffle(comps)
    path_cos = {i: {x for tr in D[i]["evidences"] for x in (tr[0], tr[2]) if x in kb} | ({D[i]["context"][0][0]}
                if D[i]["context"] else set()) for i in qids}
    value_src: dict[str, set] = {}
    for c, f in kb.items():
        for r, vs in f.items():
            for v in vs:
                if v not in kb:
                    value_src.setdefault(v, set()).add(c)
    sets = {}
    for n in FORGET_SIZES:
        fs = set(comps[:n])
        sets[str(n)] = {"companies": sorted(fs), "ids": sorted(tix[c] for c in fs),
                        "forget_questions": [i for i in qids if path_cos[i] & fs],
                        "retain_questions": [i for i in qids if not path_cos[i] & fs]}
    ku = {}
    for c in comps[:max(FORGET_SIZES)]:
        ku[str(tix[c])] = {"title": c, "values": sorted({v for vs in kb[c].values() for v in vs
                                                         if v not in kb and value_src.get(v, {c}) <= {c}}),
                           "sha": [hashlib.sha1(s.encode()).hexdigest() for s in docs[c]]}
    json.dump({"notice": G.NOTICE, "seed": FORGET_SEED, "sets": sets, "ku": ku, "value_sources": {v: sorted(s) for v, s in
              value_src.items()}}, open(out / "forget_plan.json", "w"))

    # ---- edits
    rng = random.Random(EDIT_SEED)
    pools = {r: sorted({v for f in kb.values() for v in f[r]}) for r in EDIT_RELS}
    cand = sorted(c for c in kb if any(kb[c][r] for r in EDIT_RELS)); rng.shuffle(cand)
    edits, kb2 = [], copy.deepcopy(kb)
    for c in cand:
        if len(edits) == N_EDITS:
            break
        r = rng.choice([r for r in EDIT_RELS if kb[c][r]])
        old = rng.choice(kb[c][r])
        if r == "founded_year":
            new = str(rng.choice([y for y in range(1890, 2023) if str(y) != old]))
        else:
            choices = [v for v in pools[r] if v not in kb[c][r]]
            new = rng.choice(choices)
        if not any(re.search(r"\b" + re.escape(old) + r"\b", s) for s in docs[c]):
            continue                                            # old value not stated in its own article: skip
        sents = [re.sub(r"\b" + re.escape(old) + r"\b", new, s) for s in docs[c]]
        kb2[c][r] = [new if v == old else v for v in kb2[c][r]]
        edits.append({"i": len(edits), "k": tix[c], "title": c, "rel": r, "old": old, "new": new, "sentences": sents})

    # ---- edit questions: dataset questions + generalization wordings (synthetic)
    syn = {}
    for e in edits:
        asked = {D[i]["question"] for i in qids if D[i]["type"] == f"{'year' if e['rel'] == 'founded_year' else 'company'}:1hop:f:{e['rel']}"
                 and (D[i]["context"] and D[i]["context"][0][0] == e["title"])}
        for j, w in enumerate(G.VOCAB["ask"][e["rel"]]):
            q = w.replace("{D}", e["title"])
            if q in asked:
                continue
            qid = f"bench-gen-{e['i']}-{j}"
            cat = "year" if e["rel"] == "founded_year" else "company"
            typ = f"{cat}:1hop:f:{e['rel']}"
            i0 = q.find(e["title"])
            syn[qid] = {"_id": qid, "question": q, "answer": "", "answers": answers(kb, typ, e["title"]), "answer_id": None,
                        "cat": cat, "type": typ, "context": [[e["title"], docs[e["title"]]]],
                        "supporting_facts": [[e["title"], 0]], "evidences": [[e["title"], e["rel"], v]
                                                                             for v in kb[e["title"]][e["rel"]]]}
            GP[qid] = {"template": "chain", "r1": e["rel"], "r2": None, "r3": None, "op": None,
                       "spans": [[i0, i0 + len(e["title"])]], "conf": 1.0}
    allrec = dict(D, **syn)
    first: dict = {}
    for qid in syn:
        ei, j = (int(x) for x in qid.split("-")[2:4]); first.setdefault(ei, []).append(j)
    has_ds = {e["i"] for e in edits if any(D[i]["context"] and D[i]["context"][0][0] == e["title"]
              and D[i]["type"].endswith(f"1hop:f:{e['rel']}") for i in qids)}
    edited = {e["title"]: e for e in edits}
    cls, gold_b, gold_a = {}, {}, {}
    for i, r in allrec.items():
        head = r["context"][0][0] if r["context"] else None
        if head is None:                                        # value heads: the head is the value in the question
            path = r["type"].split(":", 2)[2].split(">")
            rel = path[0][2:]
            head = next(v for v in pool_values(kb, rel) if v in r["question"]) if True else None
        gold_b[i] = answers(kb, r["type"], head); gold_a[i] = answers(kb2, r["type"], head)
        one = r["type"].split(":")[1] == "1hop" and r["type"].split(":", 2)[2].startswith("f:")
        rel = r["type"].split(":", 2)[2].split(">")[0][2:]
        if i in syn:                                            # efficacy: the first wording when the dataset
            ei, j = (int(x) for x in i.split("-")[2:4])        # has no question for this (company, relation)
            cls[i] = "efficacy" if (ei not in has_ds and j == min(first[ei])) else "generalization"
        elif one and head in edited and rel == edited[head]["rel"]:
            cls[i] = "efficacy"
        elif one and head in edited:
            cls[i] = "specificity_own"
        elif one and any(e["old"] in kb[head].get(rel, []) and e["rel"] == rel for e in edits if head != e["title"]):
            cls[i] = "specificity_neighbour"
        elif gold_b[i] != gold_a[i]:
            cls[i] = "multihop"
        else:
            cls[i] = "other"
    for e in edits:                                             # the efficacy question of each edit (timing)
        e["efficacy_qid"] = next(i for i, c in cls.items() if c == "efficacy" and (
            (i in syn and int(i.split("-")[2]) == e["i"]) or
            (i not in syn and allrec[i]["context"] and allrec[i]["context"][0][0] == e["title"])))
    json.dump({"notice": G.NOTICE, "seed": EDIT_SEED, "edits": edits, "class": cls, "gold_before": gold_b,
               "gold_after": gold_a}, open(out / "edit_plan.json", "w"))
    json.dump(list(allrec.values()), open(out / "data" / "dev.json", "w"))
    (out / "data" / "id_aliases.json").write_text("")
    json.dump(GP, open(out / "gold_plans.json", "w"))
    qs = [{"id": i, "question": r["question"], "type": r["type"],
           "gold": [tix[r["context"][0][0]]] if r["context"] else []} for i, r in allrec.items()]
    pickle.dump({"titles": titles, "sentences": P["sentences"], "owner": P["owner"], "questions": qs},
                open(out / "pool_all.pkl", "wb"))
    json.dump(qids, open(out / "questions_all.json", "w"))
    json.dump(sorted(allrec), open(out / "questions_edit.json", "w"))
    (out / "NOTICE.txt").write_text(G.NOTICE + "\n")
    summ = {"forget": {n: {"companies": len(s["companies"]), "forget_q": len(s["forget_questions"]),
                           "retain_q": len(s["retain_questions"])} for n, s in sets.items()},
            "edits": len(edits), "edit_rels": {r: sum(e["rel"] == r for e in edits) for r in EDIT_RELS},
            "edit_classes": {c: sum(v == c for v in cls.values()) for c in sorted(set(cls.values()))},
            "questions_all": len(qids), "questions_edit": len(allrec)}
    json.dump(summ, open(out / "summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


_VCACHE: dict = {}


def pool_values(kb: dict, rel: str) -> list[str]:
    if rel not in _VCACHE:                                      # longest first: a value that contains another wins
        _VCACHE[rel] = sorted({v for f in kb.values() for v in f[rel]}, key=len, reverse=True)
    return _VCACHE[rel]


if __name__ == "__main__":
    main(*[Path(a) for a in sys.argv[1:6]])
