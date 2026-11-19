"""Demo corpus A adapter: a3 → the files the MetaQA sets pipeline reads (reports/design_demo_rules.md §1, approved
2026-11-13).

Fictional companies (synthetic data generated for this demo). Any resemblance to real companies or people is
unintended.

Input: demo_gen.py output (OUT_DIR with pool.pkl, kb.json, gen.json). Nothing else is read except the biz vocabulary
(mxn/neuron/vocab/biz.json) and the wording lists of demo_gen.py (held-out check).

Outputs (same schemas as experiments/dataset/metaqa_prep_sets.py; prefix biz instead of mqs):
- EX_OUT/biz_fit.jsonl, biz_mdev.jsonl: planner targets, every train-split question; a seeded 10% (HOLD_SEED) to mdev.
  Line: {id, type, question, plan, reason: null}.
- EX_OUT/pool_bizdev.pkl, pool_biztest.pkl: {titles, sentences, owner, questions [{id, question, type, gold}]}; the
  same 3,000 company documents (one KU each); gold = [head doc] for company heads, [] for value heads.
- EX_OUT/gold_plans.json: qid -> plan + conf 1.0 for every dev and test question (dry runs only).
- EX_OUT/{enc, tok16k.model, country_table.json, step7_dev.json}: symlinks to EXPERT_DIR (corpus-free files).
- DATA_OUT/dev.json: dev + test records {_id, question, answer, answers, answer_id, cat, type, context,
  supporting_facts, evidences}; DATA_OUT/id_aliases.json (empty).
- IDS_DIR/biz_<cell>_{devfit,devpick,test}_ids.json, biz_{devfit,devpick,test}_ids.json, biz_sets.json (summary).
- SMOKE_DIR/{pool.pkl, ids.json}: dry-run pool (§5): 12 dev-fit questions per cell (60), their evidence companies plus
  seeded companies up to 300, closed under parent / subsidiary links.
- NOTICE.txt in every output directory; the notice also heads biz_sets.json.

Cells (type "<cat>:<n>hop:<path>", path steps f:<rel> / r:<rel>; cell = "<n>h_<cat>" as set_score.py derives it):
1h_company (forward, all relations but founded_year), 1h_year (founded_year), 1h_value (reverse), 2h_company
(parent -> ceo / headquartered_in; subsidiary -> product), 2h_value (founder -> headquarters; city -> chief execs).
Plans: company 1-hop chain r1; parent -> X chain r1 parent, r2 X; subsidiary -> product prev r1 parent, r2 product
(head words; rules §1 exception); value 1-hop prev r1; value 2-hop prev r1, r2. Spans: the head's character span.
Dev: dev-split questions, at most 2,000 per cell in seeded order (SPLIT_SEED), halves dev-fit / dev-pick. Test:
test-split questions, at most 2,000 per cell in seeded order (TEST_SEED); test questions must use held-out wordings
only and planner targets must use none (checked; the run fails otherwise).

usage: python demo_prep.py A3_DIR EXPERT_DIR DATA_OUT EX_OUT IDS_DIR SMOKE_DIR
"""
from __future__ import annotations

import json
import pickle
import random
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.dataset import demo_gen as G                      # noqa: E402  (wording lists, NOTICE)

HOLD_SEED, SPLIT_SEED, TEST_SEED, SMOKE_SEED = 20261117, 20261118, 20261119, 20261120
N_DEV = N_TEST = MAX_EV = 2000
N_SMOKE_Q, N_SMOKE_KU = 12, 300
CELLS = ("1h_company", "1h_year", "1h_value", "2h_company", "2h_value")


def held_out_patterns() -> tuple[list[re.Pattern], list[re.Pattern]]:
    """(held-out wordings, other wordings) as regexes over the question text; slots {D} / {V} match any text."""
    lists = list(G.VOCAB["ask"].values()) + list(G.REV.values()) + list(G.TWO.values())
    rx = lambda w: re.compile("^" + re.escape(w).replace(re.escape("{D}"), ".+").replace(re.escape("{V}"), ".+") + "$")
    return [rx(ws[-1]) for ws in lists], [rx(w) for ws in lists for w in ws[:-1]]


def convert(q: dict) -> tuple[str, str, list[str], dict]:
    """a3 question -> (cat, type, path, plan)."""
    kind, hop, *steps = q["type"].split(":")
    path = [f"{steps[i]}:{steps[i + 1]}" for i in range(0, len(steps), 2)]
    rels = [s[2:] for s in path]
    n = int(hop[0])
    cat = "year" if kind == "company" and path == ["f:founded_year"] else kind
    i = q["question"].find(q["head"])
    assert i >= 0 and q["question"].find(q["head"], i + 1) < 0, ("head not once in question", q["id"])
    sp = [[i, i + len(q["head"])]]
    if kind == "company" and path == ["f:subsidiary", "f:product"]:
        plan = {"template": "prev", "r1": "parent", "r2": "product"}
    elif kind == "company":
        plan = {"template": "chain", "r1": rels[0], "r2": rels[1] if n == 2 else None}
    else:
        plan = {"template": "prev", "r1": rels[0], "r2": rels[1] if n == 2 else None}
    plan.update(r3=None, op=None, spans=sp)
    return cat, f"{cat}:{n}hop:" + ">".join(path), path, plan


def evidences(q: dict, path: list[str], kb: dict) -> list[list[str]]:
    """KB triples on the question's path (oracle arm only)."""
    h, ev = q["head"], []
    if path[0].startswith("f:"):
        r1 = path[0][2:]
        if path == ["f:subsidiary", "f:product"]:          # as planned: children whose parent is the head
            kids = [c for c, f in kb.items() if h in f["parent"]]
            ev = [[c, "parent", h] for c in kids] + [[c, "product", v] for c in kids for v in kb[c]["product"]]
        else:
            ev = [[h, r1, v] for v in kb[h][r1]]
            if len(path) == 2:
                ev += [[m, path[1][2:], v] for m in kb[h][r1] for v in kb[m][path[1][2:]]]
    else:
        r1 = path[0][2:]
        cs = [c for c, f in kb.items() if h in f[r1]]
        ev = [[c, r1, h] for c in cs]
        if len(path) == 2:
            ev += [[c, path[1][2:], v] for c in cs for v in kb[c][path[1][2:]]]
    return ev[:MAX_EV]


def main(a3: Path, expert: Path, data_out: Path, ex_out: Path, ids_dir: Path, smoke: Path) -> None:
    P = pickle.load(open(a3 / "pool.pkl", "rb"))
    kb = json.load(open(a3 / "kb.json"))["kb"]
    titles, sents, owner = P["titles"], P["sentences"], np.asarray(P["owner"], dtype=np.int32)
    tix = {t: i for i, t in enumerate(titles)}
    held, other = held_out_patterns()
    rows, recs, plans, bad = [], {}, {}, []
    by_cell = {c: {"train": [], "dev": [], "test": []} for c in CELLS}
    for q in P["questions"]:
        cat, typ, path, plan = convert(q)
        cell = f"{typ.split(':')[1][0]}h_{cat}"
        by_cell[cell][q["split"]].append(q["id"])
        is_held = any(r.match(q["question"]) for r in held)
        if q["split"] == "test" and not is_held:
            bad.append(("test question not in a held-out wording", q["id"], q["question"]))
        if q["split"] != "test" and is_held and not any(r.match(q["question"]) for r in other):
            bad.append(("held-out wording outside test", q["id"], q["question"]))
        if q["split"] == "train":
            rows.append({"id": q["id"], "type": typ, "question": q["question"], "plan": plan, "reason": None})
            continue
        plans[q["id"]] = {**plan, "conf": 1.0}
        own = cat in ("company", "year")                       # company head: its own document
        recs[q["id"]] = {"_id": q["id"], "question": q["question"], "answer": " | ".join(q["answers"]),
                         "answers": q["answers"], "answer_id": None, "cat": cat, "type": typ,
                         "context": [[q["head"], [s for s, o in zip(sents, owner) if titles[o] == q["head"]]]]
                         if own else [],
                         "supporting_facts": [[q["head"], 0]] if own else [],
                         "evidences": evidences(q, path, kb),
                         "_gold": [tix[q["head"]]] if own else []}
    if bad:
        raise SystemExit(f"wording check failed ({len(bad)}): {bad[:5]}")
    # planner targets: train only; seeded 10% hold-out
    perm = list(range(len(rows)))
    random.Random(HOLD_SEED).shuffle(perm)
    hold = set(perm[: len(rows) // 10])
    ex_out.mkdir(parents=True, exist_ok=True)
    with open(ex_out / "biz_fit.jsonl", "w") as f, open(ex_out / "biz_mdev.jsonl", "w") as g:
        for i, r in enumerate(rows):
            (g if i in hold else f).write(json.dumps(r) + "\n")
    # ids per cell
    ids, summary = {}, {}
    for c in CELLS:
        dv, ts = sorted(by_cell[c]["dev"]), sorted(by_cell[c]["test"])
        random.Random(f"{SPLIT_SEED}:{c}").shuffle(dv)
        random.Random(f"{TEST_SEED}:{c}").shuffle(ts)
        dv, ts = dv[:N_DEV], ts[:N_TEST]
        ids[f"{c}_devfit"], ids[f"{c}_devpick"], ids[f"{c}_test"] = dv[: len(dv) // 2], dv[len(dv) // 2:], ts
        summary[c] = {"train": len(by_cell[c]["train"]), "dev_eligible": len(by_cell[c]["dev"]),
                      "test_eligible": len(by_cell[c]["test"]), "devfit": len(ids[f"{c}_devfit"]),
                      "devpick": len(ids[f"{c}_devpick"]), "test": len(ts)}
    for part in ("devfit", "devpick", "test"):
        ids[part] = [q for c in sorted(CELLS) for q in ids[f"{c}_{part}"]]
    ids_dir.mkdir(parents=True, exist_ok=True)
    for k, v in ids.items():
        json.dump(v, open(ids_dir / f"biz_{k}_ids.json", "w"))
    # dev.json (dev + test records), pools
    data_out.mkdir(parents=True, exist_ok=True)
    gold = {k: r.pop("_gold") for k, r in recs.items()}
    json.dump(list(recs.values()), open(data_out / "dev.json", "w"))
    (data_out / "id_aliases.json").write_text("")
    for split in ("dev", "test"):
        qs = [{"id": k, "question": r["question"], "type": r["type"], "gold": gold[k]}
              for k, r in recs.items() if k.startswith(f"demo-{split}-")]
        pickle.dump({"titles": titles, "sentences": sents, "owner": owner, "questions": qs},
                    open(ex_out / f"pool_biz{split}.pkl", "wb"))
    json.dump(plans, open(ex_out / "gold_plans.json", "w"))
    for f in ("enc", "tok16k.model", "country_table.json", "step7_dev.json"):
        l = ex_out / f
        if l.is_symlink() or l.exists():
            l.unlink()
        l.symlink_to((expert / f).resolve())
    # smoke pool for the dry runs (§5)
    rng = random.Random(SMOKE_SEED)
    sq = [q for c in CELLS for q in ids[f"{c}_devfit"][:N_SMOKE_Q]]
    need = {s for q in sq for s, _, _ in recs[q]["evidences"]} | {titles[g] for q in sq for g in gold[q]}
    rest = [t for t in titles if t not in need]
    rng.shuffle(rest)
    keep = set(need)
    for t in rest:
        if len(keep) >= N_SMOKE_KU:
            break
        keep.add(t)
    while True:                                                  # close under parent / subsidiary links
        add = {m for t in keep for r in ("parent", "subsidiary") for m in kb[t][r]} - keep
        if not add:
            break
        keep |= add
    kt = [t for t in titles if t in keep]
    ki = {t: i for i, t in enumerate(kt)}
    ss, so = [], []
    for s, o in zip(sents, owner):
        if titles[o] in ki:
            ss.append(s); so.append(ki[titles[o]])
    smoke.mkdir(parents=True, exist_ok=True)
    pickle.dump({"titles": kt, "sentences": ss, "owner": np.asarray(so, dtype=np.int32),
                 "questions": [{"id": q, "question": recs[q]["question"], "type": recs[q]["type"],
                                "gold": [ki[titles[g]] for g in gold[q] if titles[g] in ki]} for q in sq]},
                open(smoke / "pool.pkl", "wb"))
    json.dump(sq, open(smoke / "ids.json", "w"))
    out = {"notice": G.NOTICE, "seeds": {"hold": HOLD_SEED, "split": SPLIT_SEED, "test": TEST_SEED,
                                          "smoke": SMOKE_SEED},
           "n_dev": N_DEV, "n_test": N_TEST, "cells": summary,
           "planner_targets": {"fit": len(rows) - len(hold), "mdev": len(hold),
                               "templates": {t: sum(1 for r in rows if r["plan"]["template"] == t)
                                             for t in ("chain", "prev")}},
           "dev_test_records": len(recs), "smoke": {"questions": len(sq), "kus": len(kt), "sentences": len(ss)},
           "wording_check": "pass (test: held-out wordings only; planner targets: none)"}
    json.dump(out, open(ids_dir / "biz_sets.json", "w"), indent=1)
    for d in (data_out, ex_out, smoke):
        (d / "NOTICE.txt").write_text(G.NOTICE + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*[Path(a) for a in sys.argv[1:7]])
