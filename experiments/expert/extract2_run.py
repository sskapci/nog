"""Extractor v2 (post hoc follow-up of step 4; rules: reports/experiment_log.md, "extractor v2 rules").

--stage=train: 1 epoch (12,952 steps) of oversampled fit sentences (weight ∝ n_r^-1/2), checkpoint choice on
  the mdev subset at t = 0.5 (v2 decoding); extractor.pt + sampling_weights.json.
--stage=threshold: no-answer threshold t on the full mdev by date-rules EM; threshold.json.
--stage=test: once on the test triples, both versions; extract_test.json + rows; pass rule.

usage: extract2_run.py EXPERT_DIR ENCODER_DIR OUT_DIR --stage=train|threshold|test [--hours=24]
"""
import json, math, pickle, random, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from scipy.stats import binomtest

from mxn.expert.encoder import n_params, pad
from mxn.expert.extractor import DATE_RELS, Extractor, bio, dates_rule, decode, encode_input, piece_scores
from mxn.expert.ops import normalize

opts = dict(a[2:].split("=", 1) for a in sys.argv if a.startswith("--"))
ex, enc_dir, out = [Path(a) for a in sys.argv[1:] if not a.startswith("--")][:3]
out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(20261206); rng = np.random.default_rng(20261206)
torch.set_num_threads(8)
sp = spm.SentencePieceProcessor(model_file=str(ex / "tok16k.model"))
log = open(out / "extract.log", "a")
GRID = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02]


def say(m):
    print(m, flush=True); log.write(m + "\n"); log.flush()


def em(pred, golds):
    return any(normalize(pred) == normalize(g) for g in golds)


def wilson(k, n, z=1.96):
    if not n:
        return [0.0, 0.0]
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def cut_word(pred, sentences):
    return bool(pred) and not re.search(r"(?<!\w)" + re.escape(pred) + r"(?!\w)", " ".join(sentences))


def evaluate(net, rows, ts, rules):
    """{t: result rows}; piece scores computed once per triple."""
    net.eval()
    res = {t: [] for t in ts}
    for r in rows:
        sc = piece_scores(net, sp, r["relation"], r["subject"], r["sentences"])
        rule = dates_rule(r["relation"], r["sentences"]) if rules else None
        for t in ts:
            ans, score = (rule, 1.0) if rule is not None else decode(sc, r["sentences"], t)
            res[t].append({"subject": r["subject"], "relation": r["relation"], "pred": ans, "score": score,
                           "ok": em(ans, r["gold"]), "source": "rule" if rule is not None else "model",
                           "cut_word": cut_word(ans, r["sentences"])})
    return res


def summary(res):
    by = defaultdict(list)
    for x in res:
        by["all"].append(x["ok"]); by[x["relation"]].append(x["ok"])
    s = {k: {"n": len(v), "em": round(100 * float(np.mean(v)), 2), "ci": wilson(sum(v), len(v))} for k, v in by.items()}
    s["empty_rate"] = round(100 * float(np.mean([not x["pred"] for x in res])), 2)
    s["cut_word"] = sum(x["cut_word"] for x in res)
    return s


stage = opts["stage"]
if stage == "train":
    X = pickle.load(open(ex / "extract_fit.pkl", "rb"))
    mdev = pickle.load(open(ex / "extract_eval.pkl", "rb"))["mdev"]
    random.Random(20261205).shuffle(mdev); mdev_sel = mdev[:3000]          # the v1 subset
    ck = torch.load(enc_dir / "model.pt", map_location="cpu")
    net = Extractor(ck["config"]); net.trunk.load_state_dict(ck["state"])
    S = [(rel, t, s, sp_) for rel, t, sents, spans in X for s, sp_ in zip(sents, spans)]
    n_r = Counter(x[0] for x in S)
    w = np.array([n_r[x[0]] ** -0.5 for x in S]); w /= w.sum()
    Z = sum(math.sqrt(v) for v in n_r.values())
    json.dump({r: {"sentences": v, "share": round(100 * v / len(S), 3), "sampled_share": round(100 * math.sqrt(v) / Z, 3),
                   "multiplier": round(math.sqrt(v) / Z * len(S) / v, 3)} for r, v in n_r.most_common()},
              open(out / "sampling_weights.json", "w"), indent=1)
    BATCH, EVAL, HOURS = 64, 2000, float(opts.get("hours", 24))
    STEPS = len(S) // BATCH
    say(f"parameters: {n_params(net)}; {len(S)} fit sentences; {STEPS} steps (1 epoch); mdev subset {len(mdev_sel)}")
    decay = [p for n, p in net.named_parameters() if p.dim() > 1]
    rest = [p for n, p in net.named_parameters() if p.dim() <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": rest, "weight_decay": 0.0}], lr=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 500))
    draws = rng.choice(len(S), size=STEPS * BATCH, replace=True, p=w)
    best, step, t0 = (-1, None), 0, time.time()
    net.train()
    while True:
        b = [S[j] for j in draws[step * BATCH: (step + 1) * BATCH]]
        enc = [encode_input(sp, rel, t, s) for rel, t, s, _ in b]
        ids = pad([e[0] for e in enc])
        tags = torch.full(ids.shape, -100, dtype=torch.long)
        for i, ((_, offs), x) in enumerate(zip(enc, b)):
            tags[i, : len(offs)] = torch.tensor(bio(offs, x[3]))
        loss = F.cross_entropy(net(ids).reshape(-1, 3), tags.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        step += 1
        el = time.time() - t0
        if step % 200 == 0 or step in (10, 50):
            say(f"step {step} loss {loss.item():.3f} {el / 60:.1f} min")
        done = step >= STEPS or el > HOURS * 3600
        if step % EVAL == 0 or done:
            s_ = summary(evaluate(net, mdev_sel, [0.5], False)[0.5]); net.train()
            say(f"step {step} mdev exact {s_['all']['em']} (empty {s_['empty_rate']}%)")
            if s_["all"]["em"] > best[0]:
                best = (s_["all"]["em"], step)
                torch.save({"trunk": ck["config"], "state": net.state_dict(), "step": step, "mdev": s_}, out / "extractor.pt")
                say(f"  best so far: step {step}, {best[0]}")
        if done:
            break
    say(f"done: {step} steps in {(time.time() - t0) / 3600:.2f} h (cap hit: {step < STEPS}); best step {best[1]} mdev exact {best[0]}")
else:
    ck = torch.load(out / "extractor.pt", map_location="cpu", weights_only=False)
    net = Extractor(ck["trunk"]); net.load_state_dict(ck["state"])
    E = pickle.load(open(ex / "extract_eval.pkl", "rb"))
    if stage == "threshold":
        res = evaluate(net, E["mdev"], GRID, True)
        grid = {t: summary(r)["all"]["em"] for t, r in res.items()}
        t_best = max(GRID, key=lambda t: (grid[t], t))
        say(f"mdev date-rules EM by t: {grid}; chosen t = {t_best}")
        json.dump({"grid": grid, "t": t_best, "step": ck["step"]}, open(out / "threshold.json", "w"), indent=1)
    else:
        t = json.load(open(out / "threshold.json"))["t"]
        test = E["test"]
        rows = {"model": evaluate(net, test, [t], False)[t], "date_rules": evaluate(net, test, [t], True)[t]}
        res = {"parameters": n_params(net), "step": ck["step"], "t": t}
        for name, rr in rows.items():
            res[name] = summary(rr); print(name, json.dumps(res[name]["all"]), flush=True)
        v1 = [json.loads(l) for l in open(ex / "extract" / "extract_test_rows.jsonl")]
        d = [(a["ok_rules"], b["ok"]) for a, b in zip(v1, rows["date_rules"])]
        assert all(a["subject"] == b["subject"] and a["relation"] == b["relation"] for a, b in zip(v1, rows["date_rules"]))
        x, y = sum(b and not a for a, b in d), sum(a and not b for a, b in d)
        res["mcnemar_v2_vs_v1_date_rules"] = [x, y, float(binomtest(x, x + y).pvalue) if x + y else 1.0]
        dr = res["date_rules"]
        res["pass"] = {"overall": dr["all"]["em"], **{r: dr[r]["em"] for r in ("father", "mother", "spouse", "place of birth")}}
        res["pass"]["passed"] = dr["all"]["em"] >= 75.0 and all(dr[r]["em"] >= 50.0 for r in ("father", "mother", "spouse", "place of birth"))
        json.dump(res, open(out / "extract_test.json", "w"), indent=1)
        with open(out / "extract_test_rows.jsonl", "w") as f:
            for a, b in zip(rows["model"], rows["date_rules"]):
                f.write(json.dumps({**a, "pred_rules": b["pred"], "ok_rules": b["ok"], "source_rules": b["source"],
                                    "score_rules": b["score"]}) + "\n")
        print(json.dumps(res["pass"]))
