"""KU store for the current system (cleanup 2026-10-30; replaces legacy rebuild_kus.py): one KU per title of a step 2
pool (prep.py), its sentences in pool order, sentence vectors from our encoder ("title. sentence", passage side).
No walk graph and no KU calibration (retired 2026-10-30; stores built by rebuild_kus.py carry it as an unused field).

usage: build_kus.py EXPERT_DIR ENCODER_DIR OUT_DIR --pool=pool_test.pkl [--limit=N smoke]
"""
import json, pickle, sys, time
from pathlib import Path

import numpy as np
import yaml

from mxn.expert.encoder import Encoder
from mxn.ku.store import KU, write_store

opts = dict(a[2:].split("=", 1) for a in sys.argv if a.startswith("--"))
ex, mdir, out = [Path(a) for a in sys.argv[1:] if not a.startswith("--")][:3]
model = yaml.safe_load(open("config_expert.yaml"))["embedding"]["model"]
P = pickle.load(open(ex / opts["pool"], "rb")); by = {}
for s_, o in zip(P["sentences"], P["owner"]):
    by.setdefault(o, []).append(s_)
src = [(f"m{k}", t, by[k]) for k, t in enumerate(P["titles"]) if by.get(k)][: int(opts.get("limit", 10 ** 9))]
print(f"{len(src)} KUs, {sum(len(ss) for _, _, ss in src)} sentences (from {opts['pool']})", flush=True)
enc = Encoder(str(mdir), str(ex / "tok16k.model"))
t = time.time()
V = enc.encode([f"{ti}. {s}" for _, ti, ss in src for s in ss], "passage").astype(np.float32)
print(f"encoded in {time.time() - t:.0f}s", flush=True)
kus, off = [], 0
for i, ti, ss in src:
    kus.append(KU(i, ti, ss, V[off: off + len(ss)])); off += len(ss)
write_store(out / "net", kus, model)
json.dump({"units": len(kus), "sentences": off, "pool": opts["pool"], "model": model}, open(out / "build.json", "w"), indent=1)
print("done", flush=True)
