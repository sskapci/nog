#!/bin/bash
# 9993c Demo corpus A dev (reports/design_demo_rules.md §6; approved 2026-11-13). Fictional companies (synthetic data
# generated for this demo). Any resemblance to real companies or people is unintended.
# As 9991a: stage fit: neuron arm, 5 hosts, theta_f grid on dev-fit + dev-pick ids (biz_devfit / biz_devpick, <= 2,000
# per cell) → set_dev_pick fit (theta_f on dev-fit); stage pick: all (5 hosts), first and oracle (2 hosts), neuron on 1
# host at the chosen theta_f → per-cell gates on dev-pick (neurons beat all fitting slots, exact-set McNemar p < 0.05).
# CPU and wall per question recorded (benchmark 3). Code pinned at 099e99d (worktree t24-biz2).
# GATE: 9993b2_biz_dryrun
# VERDICT: artifacts/neuron/biz_dev/pick/set_dev_pick.json
# (pass = some cell's dev-pick gate passed)
set -e
cd /home/USER/projects/t24-biz2; export PYTHONPATH=$PWD MXN_VOCAB=biz; PY=.venv/bin/python
D=artifacts/neuron/biz_dev; mkdir -p $D
[ -s $D/dev_ids.json ] || $PY -c "import json; json.dump(json.load(open('reports/neuron/biz_devfit_ids.json')) + json.load(open('reports/neuron/biz_devpick_ids.json')), open('$D/dev_ids.json', 'w'))"
GRID=0.05,0.1,0.2,0.3,0.5,0.7,1.0
R="$PY -u experiments/neuron/e2e_run.py artifacts/expert_biz artifacts/data/biz"
A="--pool=bizdev --port=7400 --pick=artifacts/neuron/dev/opt_pick.json --kind --fmins=0.0 --rev=rel:0.1 --questions=$D/dev_ids.json"
S="$PY experiments/neuron/set_dev_pick.py artifacts/data/biz $D reports/neuron $D/pick --grid=$GRID --ids=biz"
for i in $(seq 1 240); do
  m=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo); [ $m -ge 10 ] && break
  echo "$(date -u +%T) waiting for memory: ${m} GB available, need 10"; sleep 60
done
[ -s $D/neuron/run.json ] || $R $D/neuron $A --hosts=5 --sets=$GRID
[ -s $D/pick/set_pick.json ] || $S --stage=fit
T=$($PY -c "import json;print(json.load(open('$D/pick/set_pick.json'))['theta_f'])")
[ -s $D/all/run.json ]       || $R $D/all $A --hosts=5 --arm=all --sets=$T
[ -s $D/first/run.json ]     || $R $D/first $A --hosts=2 --arm=first --sets=$T
[ -s $D/oracle/run.json ]    || $R $D/oracle $A --hosts=2 --arm=oracle --sets=$T
[ -s $D/neuron_h1/run.json ] || $R $D/neuron_h1 $A --hosts=1 --sets=$T
$S --stage=pick
