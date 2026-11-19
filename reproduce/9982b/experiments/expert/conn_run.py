"""Chapter 3 step 2: KU-owned connections learned from verified traffic (design reports/design_connections.md; rules in
reports/experiment_log.md, "rules: connections").

System: the compliant gateway with replication (RepGateway / RepHost, R = 2, H = 8, 4 simulated machine labels),
planner v2, extractor v2, c2 at τ 0.4027 with the connection bands added to its table.
Pools: test (14,128 KUs; EVAL = the 2,000 step 8 test questions; spent by 9947) and mdev (24,820 KUs; dev choices:
τ_c, bands; EVAL = 1,000 of the 5,000 mdev questions by seed). New attempt (decision 2026-10-30): fresh (EVAL = 2,000
new 2Wiki dev questions, fresh_prep.py), vktest (test3 of the value-kind attempt, pool_vktest.pkl from
vk_prep.py; base mode only: the labeled baseline, not neurons), and two runs on the mdev KUs: mdev_fit (EVAL = the same 1,000 as mdev; guard
off; τ_c, bands and τ_B fit) and mdev_pick (EVAL = the next 1,000 of the same permutation; arm choice). TRAFFIC per
pool = 2Wiki train questions whose supporting facts and gold hops are all KUs of that pool, chosen without looking at
EVAL, shuffled by seed.
Confidence arms (decision 2026-10-30): A = connection hops keep their RESOLVE-rule entry, τ 0.4027 (row keys
c2_plain / confident_plain); B = band entries added, τ_B fit on dev (c2_B / confident_B). c2 / confident = the
applied arm (params "arm", default A); base rows: c2 at τ 0.4027.

Stages (separate processes):
  --stage=prep     reads KU text and gold: question lists, verifier gold (traffic), score gold (eval); mdev packages;
  --stage=run      --mode=learned|base: hosts (learned: --tau-c), learned: traffic in batches of 100 — the driver
                   answers a batch, the verifier (subprocess) sends that batch's FEEDBACK; then EVAL (nothing learned).
                   Rows per question, resumable per batch (connections restored by replaying the feedback log);
  --stage=verify   the verifier: --batch=i sends FEEDBACK for traffic batch i; --replay re-sends the whole log.
                   A failed hop also gets a correction (design amendment 2026-10-29): the verified target, when the
                   gold names exactly one for that source and relation (else logged as ambiguous, not sent);
  --stage=score    conn_score.py: gold answers, hop accuracy, McNemar, pass rule (test) / τ_c sweep, bands (mdev).
The question driver holds no KU text, titles, gold answers or connections (audit, original rule).

usage: conn_run.py DATA_DIR EXPERT_DIR --pool=test|mdev|fresh|vktest|mdev_fit|mdev_pick --stage=... [--mode=learned|base]
       [--tau-c=X] [--no-guard] [--params=P.json] [--tag=T] [--n=N smoke]
"""
import gc, json, os, pickle, re, signal, subprocess, sys, time
from pathlib import Path

import numpy as np

opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv if a.startswith("--"))
data, ex = [Path(a) for a in sys.argv[1:] if not a.startswith("--")][:2]
pool, stage, mode = opts["pool"], opts.get("stage", "run"), opts.get("mode", "learned")
H, R, N_LABELS, PORT0, BATCH, SEED = 8, 2, 4, 7100, 100, 20261029
HALF_LIFE = 100000.0                     # in traffic questions: decay exercised but negligible here (no real time)
NQ = int(opts.get("n", 0)) or None       # smoke: first N traffic and N eval questions
out = ex / "connections" / pool; out.mkdir(parents=True, exist_ok=True)
tag = mode + (f"_{opts['tag']}" if "tag" in opts else "") + (f"_n{NQ}" if NQ else "")
kpool = "mdev" if pool.startswith("mdev") else pool          # whose KUs (pool file, packages, audit hashes)
strip = lambda t: re.sub(r"\s*\([^)]*\)$", "", t)


def say(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def gold_hops(d, tk):
    """Evidence triples (s, r, o) whose s and o are supporting-fact titles that are KUs of the pool → [[s_k, r, o_k]]."""
    sf = [t for t, _ in d["supporting_facts"]]; res = []
    for s, r, o in d["evidences"]:
        st = next((t for t in sf if t == s or strip(t) == s), None); ot = next((t for t in sf if t == o or strip(t) == o), None)
        if st in tk and ot in tk:
            res.append([tk[st], r, tk[ot]])
    return res


if stage == "prep":                      # the only stage that reads KU text and gold
    P = pickle.load(open(ex / f"pool_{kpool}.pkl", "rb")); tk = {t: k for k, t in enumerate(P["titles"])}
    dev = {r["_id"]: r for r in json.load(open(data / "dev.json"))}
    train = json.load(open(data / "train.json"))
    in_pool = lambda d: (h := gold_hops(d, tk)) and all(t in tk for t, _ in d["supporting_facts"]) and len(h) == sum(
        1 for s, r, o in d["evidences"] if any(t == o or strip(t) == o for t, _ in d["supporting_facts"]))
    own = {q["id"] for q in P["questions"]}                    # the pool's own questions are never traffic (mdev: train)
    tr = [d for d in train if in_pool(d) and d["_id"] not in own]
    tr = [tr[i] for i in np.random.default_rng(SEED).permutation(len(tr))]
    if pool in ("test", "fresh", "vktest"):
        ev = [[q["id"], q["question"]] for q in P["questions"]]
    else:                                                       # mdev / mdev_fit: perm[:1000]; mdev_pick: perm[1000:2000]
        qs = P["questions"]; lo = 1000 if pool == "mdev_pick" else 0
        ev = [[qs[i]["id"], qs[i]["question"]] for i in sorted(np.random.default_rng(SEED).permutation(len(qs))[lo:lo + 1000])]
    json.dump({"traffic": [[d["_id"], d["question"]] for d in tr], "eval": ev}, open(out / "questions.json", "w"))
    json.dump({d["_id"]: gold_hops(d, tk) for d in tr}, open(out / "gold_traffic.json", "w"))                 # verifier
    src_q = {**{d["_id"]: d for d in train if d["_id"] in own}, **{i: d for i, d in dev.items() if i in own}}
    json.dump({i: gold_hops(src_q[i], tk) for i, _ in ev}, open(out / "gold_eval_hops.json", "w"))            # score
    if pool in ("mdev", "fresh", "vktest") and not (out / "hostlist.json").exists():
        import torch
        from mxn.expert.encoder import Encoder
        from mxn.expert.hostindex import title_vector
        from mxn.expert.kugateway import ku_content_hashes
        from mxn.expert.replica import placement
        torch.set_num_threads(4)
        enc = Encoder(str(ex / "enc"), str(ex / "tok16k.model"))
        by = {}
        for s_, o_ in zip(P["sentences"], P["owner"]):         # = load_kus(P, kus_dir): checked on mdev, 24,820 / 24,820
            by.setdefault(int(o_), []).append(s_)
        KU = [(t_, by.get(k, [])) for k, t_ in enumerate(P["titles"])]
        V = np.zeros((len(KU), 256), np.float32)
        for k, (t, _) in enumerate(KU):
            V[k] = title_vector(enc, t)
            if (k + 1) % 5000 == 0:
                say(f"title vectors {k + 1}/{len(KU)}")
        names = [f"h{i}" for i in range(H)]
        machines = {h: f"m{i % N_LABELS}" for i, h in enumerate(names)}
        place = placement(range(len(KU)), machines, R)
        for h in names:
            ks = sorted(place[h])
            pickle.dump({"units": {k: (KU[k][0], list(KU[k][1])) for k in ks}, "title_vectors": {k: V[k] for k in ks},
                         "meta": place[h]}, open(out / f"package_{h}.pkl", "wb"))
        json.dump({h: {"addr": f"tcp:127.0.0.1:{PORT0 + i}", "machine": machines[h]} for i, h in enumerate(names)},
                  open(out / "hostlist.json", "w"), indent=1)
        json.dump(sorted(ku_content_hashes(KU)), open(out / "ku_sha1.json", "w"))
    say(f"prep {pool}: traffic {len(tr)}, eval {len(ev)}")
    sys.exit(0)


def packages():
    if pool == "test":
        src = ex / "replication"
        return src / "hostlist_h8_R2.json", lambda h: src / f"package_h8_R2_{h}.pkl", ex / "step9_hosts" / "ku_sha1.json"
    kd = ex / "connections" / kpool
    return kd / "hostlist.json", lambda h: kd / f"package_{h}.pkl", kd / "ku_sha1.json"


hostlist_path, package_of, sha1_path = packages()
hostlist = json.load(open(hostlist_path))
Qs = json.load(open(out / "questions.json"))
traffic, evals = Qs["traffic"][:NQ], Qs["eval"][:NQ]
fb_log = out / f"feedback_{tag}.jsonl"

if stage == "verify":                     # the verifier: gold → FEEDBACK messages (ids and outcomes only)
    from mxn.expert.replica import FEEDBACK, RepClient
    client = RepClient({h: v["addr"] for h, v in hostlist.items()})
    if "replay" in opts:
        msgs = [json.loads(l) for l in open(fb_log)] if fb_log.exists() else []
    else:
        gold = json.load(open(out / "gold_traffic.json"))
        b = int(opts["batch"]); ids = {i for i, _ in traffic[b * BATCH:(b + 1) * BATCH]}
        rows = [json.loads(l) for l in open(out / f"traffic_rows_{tag}.jsonl")]
        from experiments.expert.conn_verify import feedback_messages
        msgs = feedback_messages([r for r in rows if r["id"] in ids], gold, b)
    applied = 0
    for m in msgs:
        if m["outcome"] == "correction_ambiguous":       # logged, not sent
            continue
        r = client.manage(FEEDBACK, **{k: v for k, v in m.items() if k not in ("batch", "qid")})
        applied += bool(r and r.get("applied"))
    if "replay" not in opts:
        with open(fb_log, "a") as f:
            for m in msgs:
                f.write(json.dumps(m) + "\n")
    client.close()
    print(json.dumps({"messages": len(msgs), "applied": applied, "replay": "replay" in opts}))
    sys.exit(0)

if stage == "score":
    sys.exit(subprocess.run([sys.executable, str(Path(__file__).with_name("conn_score.py")), *sys.argv[1:]]).returncode)

# ---- the question driver: no KU text, titles, gold or connections
from mxn.expert.confidence import c2  # noqa: E402
from mxn.expert.audits import audit_content, audit_index  # noqa: E402
from mxn.expert.replica import RepClient, RepGateway  # noqa: E402
from mxn.expert.rgateway import load_resolve_gateway  # noqa: E402

s7 = json.load(open(ex / "step7_dev.json"))
ab = json.load(open(ex / "abstain_dev.json")); tab, tau = dict(ab["table"]), ab["tau"]
params = json.load(open(opts["params"])) if "params" in opts else {}      # test: τ_c, bands, band accuracies from dev
tab_B = {**tab, **params.get("tab", {})}; tau_B = params.get("tau_B")                  # arm B (only with bands fit)
arm = params.get("arm", "A")
tau_c = float(opts["tau-c"]) if "tau-c" in opts else params.get("tau_c")
guard = not ("no-guard" in opts or params.get("exact_guard") is False)
if mode == "base":
    tau_c = None
content_h = set(json.load(open(ex / "audit" / f"{kpool}_content_sha1.json")))     # made outside, audit_hashes.py
title_h = set(json.load(open(ex / "audit" / f"{kpool}_title_sha1.json")))
G0 = load_resolve_gateway(ex, entity_form=s7["form"], planner_dir="planner2")
G = RepGateway(G0.planner, G0.sp, G0.enc, G0.equiv, G0.entity_form, None, R=R); del G0
G.conn_bands = params.get("bands", [[0.0, "b0"]]); G.exact_guard = guard
HOP_KEYS = ("src", "src_relation", "value_sha1", "ku", "resolved_by", "resolve_k", "resolve_rule", "conn", "conn_used",
            "conn_conflict", "conn_exact_differs", "conn_check_failed", "via_s", "resolve_s", "check_s")
tab_plain = dict(ab["table"])                            # arm A: connection hops counted as their RESOLVE rule


def ask(q):
    t0 = time.time()
    rec = G.run([q])[0]
    plain = {**rec, "finds": [{**f, "resolved_by": f["resolve_rule"]} if f.get("conn_used") else f for f in rec["finds"]]}
    cp = c2(plain, tab_plain) if rec["answer"] is not None else 0.0
    cb = (c2(rec, tab_B) if rec["answer"] is not None else 0.0) if tau_B is not None else None
    arms = {"c2_plain": cp, "confident_plain": bool(rec["answer"] is not None and cp >= tau),
            "c2_B": cb, "confident_B": None if cb is None else bool(rec["answer"] is not None and cb >= tau_B)}
    c, conf = (cb, arms["confident_B"]) if arm == "B" and mode == "learned" else (cp, arms["confident_plain"])
    return {"answer": rec["answer"], "c2": c, "confident": conf, "arm": arm if mode == "learned" else "base", **arms,
            "seconds": time.time() - t0, "entities": [f["ku"] for f in rec["finds"] if "resolved_by" not in f],
            "hops": [{k: f.get(k) for k in HOP_KEYS} for f in rec["finds"] if "resolved_by" in f]}


def audit():
    """The two audits (pre-result amendment 2026-10-29, mxn/expert/audits.py): 1 KU content (this whole process),
    2 index (the gateway object). Clean = both clean."""
    a = {"content": audit_content(content_h), "index": audit_index(G, title_h)}
    a["clean"] = a["content"]["clean"] and a["index"]["clean"]
    return a


import socket                                          # 2026-10-30: a run whose hosts cannot bind talked to another
for i in range(len(hostlist)):                         # run's hosts on the same ports; refuse to start instead
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", PORT0 + i)) == 0:
            sys.exit(f"port {PORT0 + i} in use (another run's hosts?): refusing to start")
procs = {}
for name in hostlist:
    cmd = [sys.executable, "-m", "mxn.expert.replica", "--package", str(package_of(name)), "--name", name,
           "--hostlist", str(hostlist_path), "--R", str(R), "--expert", str(ex), "--extract", "extract2",
           "--half-life", str(HALF_LIFE)] + (["--tau-c", str(tau_c)] if tau_c is not None else [])
    procs[name] = subprocess.Popen(cmd, stdout=open(out / f"host_{tag}_{name}.log", "w"), stderr=subprocess.STDOUT)
runp = out / f"run_{tag}.json"
res = json.load(open(runp)) if runp.exists() else {"pool": pool, "mode": mode, "tau_c": tau_c, "half_life": HALF_LIFE,
                                                   "bands": G.conn_bands, "tab_added": params.get("tab", {}),
                                                   "exact_guard": guard, "tau_B": tau_B, "arm": arm,
                                                   "params_file": opts.get("params")}
verify = lambda *a: subprocess.run([sys.executable, sys.argv[0], *[x for x in sys.argv[1:] if not x.startswith("--stage")],
                                    "--stage=verify", *a], check=True, capture_output=True, text=True).stdout.strip()
try:
    for name, p in procs.items():
        while "ready" not in open(out / f"host_{tag}_{name}.log").read():
            if p.poll() is not None:
                raise RuntimeError(f"host {name} exited")
            time.sleep(1)
    say("hosts ready")
    G.client = RepClient({h: v["addr"] for h, v in hostlist.items()})
    res["audit_after_setup"] = audit(); say(f"driver audit after setup: clean={res['audit_after_setup']['clean']}")
    if mode == "learned" and fb_log.exists():
        say(f"replay: {verify('--replay')}")              # fresh hosts: connections restored from the feedback log
    tp = out / f"traffic_rows_{tag}.jsonl"
    if mode == "learned":
        done = {json.loads(l)["id"] for l in open(tp)} if tp.exists() else set()
        n_b = (len(traffic) + BATCH - 1) // BATCH
        done_b = {json.loads(l)["batch"] for l in open(fb_log)} if fb_log.exists() else set()
        with open(tp, "a") as f:
            for b in range(n_b):
                if b in done_b:
                    continue                              # answered and fed back before a restart
                for j, (i, q) in enumerate(traffic[b * BATCH:(b + 1) * BATCH]):
                    if i in done:
                        continue
                    G.clock = (lambda t: (lambda: float(t)))(b * BATCH + j)
                    f.write(json.dumps({"id": i, "t": b * BATCH + j, **ask(q)}) + "\n"); f.flush()
                say(f"traffic batch {b + 1}/{n_b}: {verify(f'--batch={b}')}")
        res["stats_after_traffic"] = json.loads(json.dumps(G.stats))
    G.clock = lambda: float(len(traffic))
    ep = out / f"eval_rows_{tag}.jsonl"
    done = {json.loads(l)["id"] for l in open(ep)} if ep.exists() else set()
    with open(ep, "a") as f:
        for n, (i, q) in enumerate(evals):
            if i not in done:
                f.write(json.dumps({"id": i, **ask(q)}) + "\n"); f.flush()
            if (n + 1) % 100 == 0:
                say(f"eval {n + 1}/{len(evals)}")
    res["audit_at_end"] = audit(); say(f"driver audit at the end: clean={res['audit_at_end']['clean']}")
    res["audit_clean"] = res["audit_after_setup"]["clean"] and res["audit_at_end"]["clean"]
    res["stats"] = G.stats; res["errors"] = G.errors
    G.client.close()
finally:
    for p in procs.values():
        if p.poll() is None:
            os.kill(p.pid, signal.SIGTERM)
    for p in procs.values():
        p.wait()
json.dump(res, open(runp, "w"), indent=1, default=str)
say(f"run {pool} {tag} done")
