#!/bin/bash
# 9991g slot-value cache identity on all 10,452 9991b test questions (owner requirement 2026-11-12; rules: log "RULES
# slot-value cache"). Answers only: nothing scored, nothing tuned. Code: worktree t24-sc; neuron arm, 5 hosts, theta_f
# 0.7, settings as 9991b, cache on (default) and memory defaults on. Pass = per question identical answer, sorted
# answer set and complete to 9991b's neuron rows (rows_A and rows_B); K full; incomplete 0; audit 0.
# GATE: 9991f_cache_dev
# VERDICT: artifacts/neuron/slotcache/test_check.json
set -e
cd /home/USER/projects/t24-sc; export PYTHONPATH=$PWD MXN_VOCAB=metaqa; PY=.venv/bin/python
D=artifacts/neuron/slotcache/test; mkdir -p $D
for i in $(seq 1 240); do
  m=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo); [ $m -ge 10 ] && break
  echo "$(date -u +%T) waiting for memory: ${m} GB available, need 10"; sleep 60
done
R="$PY -u experiments/neuron/e2e_run.py artifacts/expert_mqs artifacts/data/metaqa_sets"
A="--pool=mqstest --port=7400 --pick=artifacts/neuron/dev/opt_pick.json --kind --fmins=0.0 --rev=rel:0.1 --sets=0.7"
A="$A --questions=artifacts/neuron/mqs_test/test_ids.json"
[ -s $D/neuron/run.json ] || $R $D/neuron $A --hosts=5
$PY - <<'PY'
import json
from pathlib import Path
D, T = Path("artifacts/neuron/slotcache/test/neuron"), Path("artifacts/neuron/mqs_test/neuron")
key = lambda r: (r["answer"], sorted(r.get("answer_set") or []), r["complete"])
def rows(p):
    out = {}
    for line in open(p):
        r = json.loads(line); out[r["id"]] = key(r)
    return out
c = {}
for f in ("rows_A.jsonl", "rows_B.jsonl"):
    a, b = rows(D / f), rows(T / f)
    c[f] = {"n": len(b), "identical": sum(a.get(i) == b[i] for i in b), "same_ids": a.keys() == b.keys()}
r = json.load(open(D / "run.json"))
c["K_full"] = r["K_identical"] == r["questions"]; c["incomplete_0"] = r["incomplete"] == {"A": 0, "B": 0}
c["audit_0"] = all(x["sentences_in_memory"] == 0 for x in r.get("audit_hosts", [])) if r.get("audit_hosts") else None
rc = next(v for k, v in r.items() if k.startswith("reverse_cost_A"))
c["cost_A"] = {k[len("reverse_"):]: v for k, v in rc.items()}
c["pass"] = all(c[f]["identical"] == c[f]["n"] > 0 and c[f]["same_ids"] for f in ("rows_A.jsonl", "rows_B.jsonl")) \
    and c["K_full"] and c["incomplete_0"] and c["audit_0"] is not False
json.dump(c, open("artifacts/neuron/slotcache/test_check.json", "w"), indent=1); print(json.dumps(c))
PY
