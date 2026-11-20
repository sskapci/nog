"""Step 2: train our bi-encoder from scratch (rules: reports/experiment_log.md, "splits and rules fixed
before any training (steps 1–2)").

InfoNCE (temperature 0.05) over in-batch passages: the 256 positives and the hard negatives of the
batch; a passage whose text equals the query's positive is masked (not a negative). Hard negative per
pair: the highest-BM25 non-supporting sentence among the question's own context sentences ("ti" pairs
have none). Passage format: the rule chooses "sentence" or "title. sentence" on mdev, so training sees
both (each passage formatted either way at random), and every checkpoint is scored on mdev in both
formats. AdamW lr 5e-4, weight decay 0.01, warmup 1,000 steps, cosine decay to 0 over the planned
steps; wall-clock budget; checkpoint and mdev recall@5 every 2,000 steps; best kept.

--resume=CK.pt (follow-up of decisions 2026-10-25, labeled so): start from that checkpoint's weights
and step; fresh AdamW moments with a 200-step re-warmup; the schedule is the original one evaluated
at the global step with --steps as the planned total (e.g. 1 epoch = 6,021 steps at batch 256).

usage: train_encoder.py EXPERT_DIR OUT_DIR [--hours=8] [--steps=N planned for the schedule] [--resume=CK.pt]
"""
import json, math, pickle, sys, time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F

from mxn.expert.bm25 import BM25
from mxn.expert.encoder import BiEncoder, n_params, pad
from mxn.expert.retrieval_eval import first_k_titles, recall
from mxn.common.safeload import pickle_load  # no code execution from data files (2026-11-20)

opts = dict(a[2:].split("=", 1) for a in sys.argv if a.startswith("--"))
ex, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
HOURS, STEPS, BATCH, EVAL = float(opts.get("hours", 8)), int(opts.get("steps", 20000)), 256, 2000
torch.manual_seed(20261203); rng = np.random.default_rng(20261203)
torch.set_num_threads(8)
sp = spm.SentencePieceProcessor(model_file=str(ex / "tok16k.model"))
log = open(out / "train.log", "a")


def say(m):
    print(m, flush=True); log.write(m + "\n"); log.flush()


t0 = time.time()
pairs = pickle_load(ex / "pairs_fit.pkl")
hard = []
cache = {}
for kind, q, t, pos, negs in pairs:                 # BM25 over the question's own context sentences
    if not negs:
        hard.append(None); continue
    key = id(negs)
    if key not in cache:
        cache.clear(); cache[key] = BM25([s for _, s in negs])
    hard.append(negs[int(np.argmax(cache[key].scores([q])[0]))])     # (title, sentence)
say(f"{len(pairs)} pairs, hard negatives in {time.time() - t0:.0f}s")
M = pickle_load(ex / "pool_mdev.pkl")

config = dict(vocab=sp.get_piece_size())
net = BiEncoder(**config)
START = 0
if "resume" in opts:
    rk = torch.load(opts["resume"], map_location="cpu", weights_only=True)
    net.load_state_dict(rk["state"]); START = rk["step"]
    say(f"resumed from {opts['resume']} at step {START}")
say(f"parameters: {n_params(net)}")
decay = [p for n, p in net.named_parameters() if p.dim() > 1]
rest = [p for n, p in net.named_parameters() if p.dim() <= 1]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": rest, "weight_decay": 0.0}], lr=5e-4)
def lr_factor(s):
    g = s + START                                   # global step
    warm = min(1, (g + 1) / 1000) * (min(1, (s + 1) / 200) if START else 1)
    return warm * 0.5 * (1 + math.cos(math.pi * min(g, STEPS) / STEPS))


sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)


def enc(texts, kind):
    return net(pad([sp.encode(x)[: net.max_len] for x in texts]), kind)


def fmt(title, s, with_title):
    return f"{title}. {s}" if with_title else s


def mdev_recall():
    net.eval(); res = {}
    with torch.inference_mode():
        def encode_all(texts, kind):
            order = np.argsort([len(x) for x in texts]); V = np.zeros((len(texts), 256), np.float32)
            for i in range(0, len(texts), 512):
                idx = order[i: i + 512]; V[idx] = enc([texts[j] for j in idx], kind).numpy()
            return V
        Q = encode_all([q["question"] for q in M["questions"]], 0)
        gold = [q["gold"] for q in M["questions"]]
        for wt in (False, True):
            P = encode_all([fmt(M["titles"][o], s, wt) for s, o in zip(M["sentences"], M["owner"])], 1)
            S = Q @ P.T
            res["title" if wt else "sentence"] = recall([first_k_titles(S[i], M["owner"], 5) for i in range(len(Q))], gold)["recall"]
    net.train(); return res


best, step, order, pos_i = (-1, None, None), START, rng.permutation(len(pairs)), 0
if START:                                           # a fresh shuffle (the first run's order is not replayed)
    rng = np.random.default_rng(20261203 + START); order = rng.permutation(len(pairs))
t0 = time.time()                                    # the budget counts training only, not data loading
net.train()
while True:
    if pos_i + BATCH > len(order):
        order, pos_i = rng.permutation(len(pairs)), 0
    idx = order[pos_i: pos_i + BATCH]; pos_i += BATCH
    qs, ps, keys = [], [], []
    for j in idx:
        kind, q, t, p, _ = pairs[j]
        qs.append(q); ps.append(fmt(t, p, rng.random() < 0.5)); keys.append(p)
    for j in idx:
        h = hard[j]
        if h is not None:                           # formatted at random too: no format shortcut
            ps.append(fmt(h[0], h[1], rng.random() < 0.5)); keys.append(h[1])
    qv, pv = enc(qs, 0), enc(ps, 1)
    logits = qv @ pv.T / 0.05
    same = torch.tensor([[keys[c] == keys[r] and c != r for c in range(len(ps))] for r in range(len(qs))])
    logits = logits.masked_fill(same, -1e4)
    loss = F.cross_entropy(logits, torch.arange(len(qs)))
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    step += 1
    el = time.time() - t0
    if step % 100 == 0 or step in (10, 30):
        say(f"step {step} loss {loss.item():.3f} lr {sched.get_last_lr()[0]:.2e} {el / 60:.0f} min")
    done = el > HOURS * 3600 or step >= STEPS
    if step % EVAL == 0 or done:
        r = mdev_recall()
        fmt_best = max(r, key=r.get)
        say(f"step {step} mdev recall@5 {json.dumps({k: round(v, 2) for k, v in r.items()})}")
        torch.save({"config": config, "state": net.state_dict(), "step": step}, out / f"ck{step}.pt")
        if r[fmt_best] > best[0]:
            best = (r[fmt_best], step, fmt_best)
            torch.save({"config": config, "state": net.state_dict(), "step": step,
                        "passage_title": fmt_best == "title", "mdev_recall5": r}, out / "model.pt")
            say(f"  best so far: step {step}, {fmt_best}, {r[fmt_best]:.2f}")
    if done:
        break
say(f"done: {step} steps in {(time.time() - t0) / 3600:.2f} h; best step {best[1]} ({best[2]}) mdev recall@5 {best[0]:.2f}")
