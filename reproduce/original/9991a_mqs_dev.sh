#!/bin/bash
# 9991a MetaQA set answers / person heads dev. Rules: log "RULES MetaQA set answers and person heads" (b24b1ad, log 2026-11-06 23:25 UTC).
# GATE: 9991a1_mqs_hostsreplay
# Code: rules commit b24b1ad + row-streaming fix c928e63 + scorer memory fix 368ce83 (worktree t24-sr3; execution only, OOM kills; log 2026-11-07/08). Stage fit: neuron arm, 5 hosts, theta_f grid → set_dev_pick fit.
# Stage pick: all fitting slots (5 hosts), first slot and oracle (2 host processes: user decision 2026-11-07 option c,
# llama-server OOM-killed 3 times; identity shown by 9991a1), neuron on 1 host at the chosen theta_f → gates per cell.
# VERDICT: artifacts/neuron/mqs_dev/pick/set_dev_pick.json
# (pass = some cell's dev-pick gate passed)
set -e
cd /home/USER/projects/t24-sr3; export PYTHONPATH=$PWD MXN_VOCAB=metaqa; PY=.venv/bin/python
D=artifacts/neuron/mqs_dev; mkdir -p $D
GRID=0.05,0.1,0.2,0.3,0.5,0.7,1.0
R="$PY -u experiments/neuron/e2e_run.py artifacts/expert_mqs artifacts/data/metaqa_sets"
A="--pool=mqsdev --port=7400 --pick=artifacts/neuron/dev/opt_pick.json --kind --fmins=0.0 --rev=rel:0.1"
S="$PY experiments/neuron/set_dev_pick.py artifacts/data/metaqa_sets $D reports/neuron $D/pick --grid=$GRID"
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
