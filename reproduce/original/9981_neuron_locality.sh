#!/bin/bash
# Neuron locality placement by connection weights, live (rules: reports/design_neuron_locality.md on branch loc, log
# "RULES neuron locality"): fresh pool (14,177 KUs, 2,000 questions; identity and routing only), arm neuron at the
# dev choices, 5 hosts. Runs from the loc worktree (kept apart from e2e until the value-kind test is done).
# VERDICT: artifacts/neuron/locality/run/loc_check.json
set -e
cd /home/USER/projects/t24-loc; export PYTHONPATH=$PWD; PY=.venv/bin/python
D=${D:-artifacts/neuron/locality}; P=${P:-artifacts/neuron/e2e_dev/pick}; PICK=${PICK:-artifacts/neuron/dev/opt_pick.json}
mkdir -p $D
for i in $(seq 1 240); do                           # memory guard: 5 hosts need about 12 GB
  m=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo); [ $m -ge ${NEED:-13} ] && break
  echo "$(date -u +%T) waiting for memory: ${m} GB available, need ${NEED:-13}"; sleep 60
done
[ $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) -ge ${NEED:-13} ] || { echo "not enough memory after 4 h"; exit 3; }
FM=$($PY -c "import json;print(json.load(open('$P/e2e_fmin.json'))['fmin'])")
[ -s $D/run/loc_run.json ] || $PY -u experiments/neuron/e2e_run.py artifacts/expert artifacts/data/2wiki/extracted \
    $D/run --pool=fresh --hosts=5 --locality --port=7400 --pick=$PICK --fmins=$FM --tau=$P/e2e_tau.json
$PY experiments/neuron/loc_check.py $D/run
