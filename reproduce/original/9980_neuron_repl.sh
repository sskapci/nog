#!/bin/bash
# Neuron replication R = 2, two live host deaths (rules: reports/design_neuron_replication.md, log "RULES neuron
# replication"): subset of the fresh pool (first 1,000 questions, their KUs, every third other KU; identity only),
# arm neuron at the dev choices, 5 hosts; replicate, then h2 and h4 killed by the test harness during question cycles.
# VERDICT: artifacts/neuron/repl/run/repl_check.json
set -e
cd /home/USER/projects/t24-e2e; export PYTHONPATH=$PWD; PY=.venv/bin/python
D=${D:-artifacts/neuron/repl}; P=${P:-artifacts/neuron/e2e_dev/pick}; PICK=${PICK:-artifacts/neuron/dev/opt_pick.json}
mkdir -p $D
for i in $(seq 1 240); do                           # memory guard: R = 2 on the subset needs about 11 GB
  m=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo); [ $m -ge ${NEED:-12} ] && break
  echo "$(date -u +%T) waiting for memory: ${m} GB available, need ${NEED:-12}"; sleep 60
done
[ $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) -ge ${NEED:-12} ] || { echo "not enough memory after 4 h"; exit 3; }
[ -s $D/pool.pkl ] || $PY - <<PYEOF
import pickle, numpy as np
M = pickle.load(open('artifacts/expert/pool_fresh.pkl', 'rb'))
qs = M['questions'][:1000]
keep = sorted({g for q in qs for g in q['gold']} | set(range(0, len(M['titles']), 3)))
remap = {k: i for i, k in enumerate(keep)}
own = np.asarray(M['owner']); sel = [i for i, o in enumerate(own) if int(o) in remap]
T = {'titles': [M['titles'][k] for k in keep], 'sentences': [M['sentences'][i] for i in sel],
     'owner': np.array([remap[int(own[i])] for i in sel]),
     'questions': [{**q, 'gold': [remap[g] for g in q['gold']]} for q in qs]}
pickle.dump(T, open('$D/pool.pkl', 'wb')); print(len(keep), 'KUs', len(qs), 'questions')
PYEOF
FM=$($PY -c "import json;print(json.load(open('$P/e2e_fmin.json'))['fmin'])")
[ -s $D/run/repl_run.json ] || $PY -u experiments/neuron/e2e_run.py artifacts/expert artifacts/data/2wiki/extracted \
    $D/run --pool=fresh --poolfile=$D/pool.pkl --hosts=5 --cold --repl=2 --victims=h2,h4 --idle=60 --port=7400 \
    --pick=$PICK --fmins=$FM --tau=$P/e2e_tau.json
$PY experiments/neuron/repl_check.py $D/run
