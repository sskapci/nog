"""Step 3: train our planner and measure it (rules: reports/experiment_log.md, "step 3 rules").

--stage=train: fit on plans_fit (clean targets only), trunk initialized from the step 2 encoder;
  cross-entropy summed over template, r1, r2, op (heads that do not apply to the target's template are
  trained toward "none") and the BIO tags; AdamW lr 1e-4 (warmup 500 steps, then constant), weight
  decay 0.01, batch 64; wall-clock budget; mdev exact-plan accuracy every 2,000 steps and at the end;
  the best checkpoint is kept as planner.pt.
--stage=test: the kept planner, once, on the 2,000 test questions.

Exact plan = template, r1, r2, op and the entity spans all equal the target (spans compared as their
question text after lowercasing, accent folding and removing punctuation and spaces).

--plans=plans2: version 2 targets (post hoc, decisions 2026-10-25), otherwise v1 ("plans").

usage: planner_run.py EXPERT_DIR ENCODER_DIR OUT_DIR --stage=train|test [--hours=2] [--plans=plans2]
"""
import json, math, random, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F

from experiments.expert.plans import span_key
from mxn.expert.encoder import n_params, pad
from mxn.expert.para_grammar import paraphrase
from mxn.expert.planner import OPS, TAGS, TEMPLATES, Planner, load_planner, pieces, predict, tag_targets
from mxn.common.safeload import torch_load  # no code execution from data files (2026-11-20)

opts = dict(a[2:].split("=", 1) for a in sys.argv if a.startswith("--"))
ex, enc_dir, out = [Path(a) for a in sys.argv[1:] if not a.startswith("--")][:3]
out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(20261204); rng = np.random.default_rng(20261204)
torch.set_num_threads(8)
sp = spm.SentencePieceProcessor(model_file=str(ex / "tok16k.model"))
log = open(out / "planner.log", "a")


def say(m):
    print(m, flush=True); log.write(m + "\n"); log.flush()


def load(name):
    return [json.loads(l) for l in open(ex / f"{opts.get('plans', 'plans')}_{name}.jsonl")]


def components(q, p, g):
    """Per-component agreement of a predicted plan p with the target g."""
    def keys(plan):
        return [span_key(q[a:b]) if s else None for s in plan["spans"] for a, b in [s or (0, 0)]]
    c = {"template": p["template"] == g["template"], "r1": p["r1"] == g["r1"], "r2": p["r2"] == g["r2"],
         "op": p["op"] == g["op"], "spans": keys(p) == keys(g)}
    if "r3" in g:                                       # reverse-lookup targets (design_reverse_lookup.md)
        c["r3"] = p.get("r3") == g["r3"]
    return c


def wilson(k, n, z=1.96):
    if not n:
        return [0.0, 0.0]
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def evaluate(net, rows):
    """Exact plan accuracy over rows with a clean target, per type and per component; plus the
    accuracy over all rows counting unclean ones as wrong."""
    clean = [r for r in rows if r["plan"]]
    preds = predict(net, sp, [r["question"] for r in clean])
    by, comp = defaultdict(lambda: [0, 0]), defaultdict(int)
    detail = []
    for r, p in zip(clean, preds):
        c = components(r["question"], p, r["plan"])
        ok = all(c.values())
        for t in ("all", r["type"]):
            by[t][0] += ok; by[t][1] += 1
        for k, v in c.items():
            comp[k] += v
        detail.append({"id": r["id"], "ok": ok, "pred": p, **{f"ok_{k}": v for k, v in c.items()}})
    res = {t: {"n": n, "exact": round(100 * k / n, 2), "ci": wilson(k, n)} for t, (k, n) in sorted(by.items())}
    res["components"] = {k: round(100 * v / len(clean), 2) for k, v in comp.items()}
    res["all_rows_unclean_wrong"] = {"n": len(rows), "exact": round(100 * by["all"][0] / len(rows), 2),
                                     "ci": wilson(by["all"][0], len(rows))}
    return res, detail


if opts["stage"] == "train":
    fit, mdev = [r for r in load("fit") if r["plan"]], load("mdev")
    relations = sorted({r["plan"]["r1"] for r in fit} | {r["plan"]["r2"] for r in fit if r["plan"]["r2"]}
                       | {r["plan"]["r3"] for r in fit if r["plan"].get("r3")})
    templates = TEMPLATES + sorted({r["plan"]["template"] for r in fit} - set(TEMPLATES))   # 2Wiki: unchanged
    R3 = any("r3" in r["plan"] for r in fit)
    ck = torch.load(enc_dir / "model.pt", map_location="cpu", weights_only=True)
    net = Planner(relations, ck["config"], templates, R3); net.trunk.load_state_dict(ck["state"])
    say(f"{len(fit)} clean fit plans, {len(relations)} relations; parameters: {n_params(net)} (trunk {n_params(net.trunk)})")
    def example(q, p):
        ids, offs = pieces(sp, q, net.trunk.max_len)
        return (ids, tag_targets(offs, p["spans"]), net.templates.index(p["template"]), net.relations.index(p["r1"]),
                net.relations.index(p["r2"] or "none"), OPS.index(p["op"] or "none"),
                net.relations.index(p.get("r3") or "none"))

    def grammar_row(r, g):
        """The grammar's paraphrase of a row (same plan, new text and spans), or None if it has none."""
        p = r["plan"]
        ents = [r["question"][a:b] for a, b in [s for s in p["spans"] if s]]
        out_ = paraphrase(p, ents, g) if len(ents) >= (1 if p["template"] == "chain" else 2) else None
        if out_ is None:
            return None
        q, spans = out_
        return {**r, "question": q, "plan": {**p, "spans": spans + p["spans"][len(spans):]}}

    # --grammar=P (planner v3 rules): each example is, with probability P, a fresh grammar draw for its plan
    G = float(opts.get("grammar", 0)); grng = random.Random(20261211)
    X = [example(r["question"], r["plan"]) for r in fit]
    if G:
        g2 = random.Random(20261212)
        mdev_g = [x for x in (grammar_row(r, g2) if r["plan"] else None for r in mdev) if x]
        say(f"grammar mixing p={G}; mdev grammar paraphrases: {len(mdev_g)}")
    HOURS, BATCH, EVAL = float(opts.get("hours", 2)), 64, 2000
    decay = [p for n, p in net.named_parameters() if p.dim() > 1]
    rest = [p for n, p in net.named_parameters() if p.dim() <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": rest, "weight_decay": 0.0}], lr=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 500))
    best, step, order, pos_i, used = (-1, None), 0, rng.permutation(len(X)), 0, 0.0
    if "resume" in opts and (out / "resume.pt").exists():
        rs = torch_load(out / "resume.pt")
        net.load_state_dict(rs["net"]); opt.load_state_dict(rs["opt"]); sched.load_state_dict(rs["sched"])
        best, step, order, pos_i, used = tuple(rs["best"]), rs["step"], rs["order"], rs["pos_i"], rs["used"]
        rng.bit_generator.state = rs["rng"]; torch.set_rng_state(rs["torch_rng"]); grng.setstate(rs["grng"])
        say(f"resumed at step {step}, {used / 3600:.2f} h used, best {best}")
    t0 = time.time() - used; net.train()
    while True:
        if pos_i + BATCH > len(order):
            order, pos_i = rng.permutation(len(X)), 0
        b = []
        for j in order[pos_i: pos_i + BATCH]:
            x = X[j]
            if G and grng.random() < G:
                gr = grammar_row(fit[j], grng)
                x = example(gr["question"], gr["plan"]) if gr else x
            b.append(x)
        pos_i += BATCH
        ids = pad([x[0] for x in b])
        tags = torch.full(ids.shape, -100, dtype=torch.long)
        for i, x in enumerate(b):
            tags[i, : len(x[1])] = torch.tensor(x[1])
        o = net(ids)
        y = [torch.tensor([x[k] for x in b]) for k in (2, 3, 4, 5)]
        loss = (F.cross_entropy(o["template"], y[0]) + F.cross_entropy(o["r1"], y[1]) + F.cross_entropy(o["r2"], y[2])
                + F.cross_entropy(o["op"], y[3]) + F.cross_entropy(o["tag"].reshape(-1, len(TAGS)), tags.reshape(-1)))
        if R3:
            loss = loss + F.cross_entropy(o["r3"], torch.tensor([x[6] for x in b]))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        step += 1
        el = time.time() - t0
        if step % 200 == 0 or step in (10, 50):
            say(f"step {step} loss {loss.item():.3f} {el / 60:.1f} min")
        done = el > HOURS * 3600
        if step % EVAL == 0 or done:
            res, _ = evaluate(net, mdev)
            say(f"step {step} mdev exact {res['all']['exact']} components {json.dumps(res['components'])}")
            score = res["all"]["exact"]
            if G:                                   # selection: mean of mdev originals and mdev grammar paraphrases
                res_g, _ = evaluate(net, mdev_g); res["grammar"] = res_g
                score = round((res["all"]["exact"] + res_g["all"]["exact"]) / 2, 3)
                say(f"step {step} mdev grammar exact {res_g['all']['exact']}; selection score {score}")
            net.train()
            if score > best[0]:
                best = (score, step)
                torch.save({"relations": relations, "trunk": ck["config"], "state": net.state_dict(), "step": step,
                            "mdev": res, **({"templates": templates, "r3": True} if R3 else {})}, out / "planner.pt")
                say(f"  best so far: step {step}, {best[0]}")
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "best": best,
                        "step": step, "order": order, "pos_i": pos_i, "used": time.time() - t0,
                        "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(), "grng": grng.getstate()},
                       out / "resume.pt")
        if done:
            break
    say(f"done: {step} steps in {(time.time() - t0) / 3600:.2f} h; best step {best[1]} mdev exact {best[0]}")
else:
    ck = torch.load(out / "planner.pt", map_location="cpu", weights_only=True)
    net = load_planner(ck)
    rows = load(opts["stage"] if opts["stage"] != "test" else "test")
    res, detail = evaluate(net, rows)
    res["parameters"] = n_params(net); res["step"] = ck["step"]
    json.dump(res, open(out / f"planner_{opts['stage']}.json", "w"), indent=1)
    with open(out / f"planner_{opts['stage']}_detail.jsonl", "w") as f:
        for d in detail:
            f.write(json.dumps(d) + "\n")
    print(json.dumps(res, indent=1))
    if opts["stage"] == "test":
        print("PASS" if res["all"]["exact"] >= 80 else "FAIL", "(rule: exact plan accuracy >= 80% on clean test targets)")
