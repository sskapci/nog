#!/bin/bash
# Neuron cold copies (R = 1 recovery), two live host deaths (rules: reports/design_neuron_cold.md, log "RULES neuron
# cold copies"): fresh pool (14,177 KUs, 2,000 questions; identity only), arm neuron at the dev choices, 5 hosts;
# h2 then h4 killed by the test harness during question cycles. Rules f8894b3.
# VERDICT: artifacts/neuron/cold/run/cold_check.json
set -e
cd /home/USER/projects/t24-e2e; export PYTHONPATH=$PWD; PY=.venv/bin/python
D=${D:-artifacts/neuron/cold}; P=${P:-artifacts/neuron/e2e_dev/pick}; PICK=${PICK:-artifacts/neuron/dev/opt_pick.json}
mkdir -p $D
for i in $(seq 1 240); do                           # memory guard: 5 hosts need about 12 GB; 3 survivors hold all KUs
  m=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo); [ $m -ge 13 ] && break
  echo "$(date -u +%T) waiting for memory: ${m} GB available, need 13"; sleep 60
done
[ $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) -ge 13 ] || { echo "not enough memory after 4 h"; exit 3; }
FM=$($PY -c "import json;print(json.load(open('$P/e2e_fmin.json'))['fmin'])")
[ -s $D/run/cold_run.json ] || $PY -u experiments/neuron/e2e_run.py artifacts/expert artifacts/data/2wiki/extracted \
    $D/run --pool=fresh --hosts=5 --cold --victims=h2,h4 --idle=60 --port=7400 --pick=$PICK --fmins=$FM \
    --tau=$P/e2e_tau.json
$PY experiments/neuron/cold_check.py $D/run
