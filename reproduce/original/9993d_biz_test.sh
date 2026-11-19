#!/bin/bash
# 9993d Demo corpus A test, one attempt (reports/design_demo_rules.md §7; approved 2026-11-13). Fictional companies
# (synthetic data generated for this demo). Any resemblance to real companies or people is unintended.
# As 9991b: only the cells whose dev-pick gate passed in 9993c (1h_company, 2h_company, 2h_value), at the frozen
# theta_f (0.7): neuron (5 hosts), all fitting slots (5 hosts, as in dev), first slot (2 hosts), neuron 1 host → tier per
# cell (set_score --tier: FLIES = neurons beat all fitting slots on exact-set McNemar p < 0.05, K identical, audit 0
# after the text drop and > 0 before, complete, 5 hosts = 1 host); then the oracle (ceiling, report only). Test
# questions use only the held-out wording of each relation. Results labelled "fictional companies (synthetic)".
# Code pinned at 099e99d (worktree t24-biz2).
# GATE: 9993c_biz_dev
# VERDICT: artifacts/neuron/biz_test/test_verdict.json
# (pass = some tested cell FLIES)
cd /home/USER/projects/t24-biz2; export PYTHONPATH=$PWD MXN_VOCAB=biz; PY=.venv/bin/python
D=artifacts/neuron/biz_test; P=artifacts/neuron/biz_dev/pick; mkdir -p $D
T=$($PY -c "import json;print(json.load(open('$P/set_pick.json'))['theta_f'])")
$PY -c "
import json
cells = json.load(open('$P/set_dev_pick.json'))['cells_to_test']
ids = [i for c in cells for i in json.load(open(f'reports/neuron/biz_{c}_test_ids.json'))]
json.dump(ids, open('$D/test_ids.json', 'w')); print('cells', cells, 'questions', len(ids))"
R="$PY -u experiments/neuron/e2e_run.py artifacts/expert_biz artifacts/data/biz"
A="--pool=biztest --port=7400 --pick=artifacts/neuron/dev/opt_pick.json --kind --fmins=0.0 --rev=rel:0.1 --sets=$T --questions=$D/test_ids.json"
for i in $(seq 1 240); do
  m=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo); [ $m -ge 10 ] && break
  echo "$(date -u +%T) waiting for memory: ${m} GB available, need 10"; sleep 60
done
[ -s $D/neuron/run.json ]    || $R $D/neuron $A --hosts=5
[ -s $D/all/run.json ]       || $R $D/all $A --hosts=5 --arm=all
[ -s $D/first/run.json ]     || $R $D/first $A --hosts=2 --arm=first
[ -s $D/neuron_h1/run.json ] || $R $D/neuron_h1 $A --hosts=1
$PY experiments/neuron/set_score.py artifacts/data/biz $D/test_score.json --ids=$D/test_ids.json \
    --neuron=$D/neuron --all=$D/all --first=$D/first --onehost=$D/neuron_h1 --tier
$PY -c "
import json; s = json.load(open('$D/test_score.json'))
json.dump({'tiers': s['tiers'], 'theta_f': $T, 'pass': s['pass']}, open('$D/test_verdict.json', 'w'), indent=1)
print(s['tiers'])"
[ -s $D/oracle/run.json ] || $R $D/oracle $A --hosts=2 --arm=oracle
$PY experiments/neuron/set_score.py artifacts/data/biz $D/test_score_oracle.json --ids=$D/test_ids.json \
    --neuron=$D/neuron --all=$D/all --first=$D/first --oracle=$D/oracle --onehost=$D/neuron_h1 --tier
