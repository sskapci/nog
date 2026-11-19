#!/bin/bash
# Value-kind attempt, TEST3, one attempt (rules: log "RULES value-kind attempt"): vktest pool (14,114 KUs, 2,000 new
# 2Wiki dev-split questions), 5 hosts, --kind, dev choices frozen: neuron, first-slot, neuron on 1 host -> tier with
# the vkdev bars. Then (report only, after the tier is written): oracle ceiling on test3 and the drift table
# dev-pick -> test3. Runs only if the vkdev gate passed.
# VERDICT: artifacts/neuron/vk_test/e2e_test_score.json
set -e
cd /home/USER/projects/t24-e2e; export PYTHONPATH=$PWD; PY=.venv/bin/python
D=${D:-artifacts/neuron/vk_test}; P=${P:-artifacts/neuron/vk_dev/pick}; V=${V:-artifacts/neuron/vk_dev}
PICK=${PICK:-artifacts/neuron/dev/opt_pick.json}; mkdir -p $D
for i in $(seq 1 240); do                           # memory guard: 5 hosts need about 12 GB
  m=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo); [ $m -ge ${NEED:-12} ] && break
  echo "$(date -u +%T) waiting for memory: ${m} GB available, need ${NEED:-12}"; sleep 60
done
[ $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) -ge ${NEED:-12} ] || { echo "not enough memory after 4 h"; exit 3; }
[ -n "$SMOKE" ] || $PY -c "import json,sys;sys.exit(0 if json.load(open('$P/e2e_dev_pick.json'))['pass'] else 'dev-pick gate not passed')"   # SMOKE: dry run only
FM=$($PY -c "import json;print(json.load(open('$P/e2e_fmin.json'))['fmin'])")
R="$PY -u experiments/neuron/e2e_run.py artifacts/expert artifacts/data/2wiki/extracted ${POOLFILE:+--poolfile=$POOLFILE}"
A="--pool=vktest --port=7400 --pick=$PICK --kind --tau=$P/e2e_tau.json"
[ -s $D/neuron/run.json ]    || $R $D/neuron $A --hosts=5 --fmins=$FM
[ -s $D/first/run.json ]     || $R $D/first $A --hosts=5 --arm=first
[ -s $D/neuron_h1/run.json ] || $R $D/neuron_h1 $A --hosts=1 --fmins=$FM
$PY experiments/neuron/e2e_score.py artifacts/data/2wiki/extracted vktest $D/e2e_test_score.json --neuron=$D/neuron \
    --first=$D/first --onehost=$D/neuron_h1 --bars=$P/e2e_bars.json ${IDS:+--ids=$IDS}
# report only (committed in the rules; the tier above is final before this runs)
[ -s $D/oracle/run.json ]    || $R $D/oracle $A --hosts=5 --arm=oracle --fmins=$FM
$PY experiments/neuron/e2e_score.py artifacts/data/2wiki/extracted vktest $D/oracle_test_score.json --neuron=$D/neuron \
    --first=$D/first --oracle=$D/oracle
$PY experiments/neuron/e2e_drift.py artifacts/data/2wiki/extracted artifacts/expert $D/drift.json \
    "devpick:vkdev:$V/neuron/rows_B_f$FM.jsonl:${PICKIDS:-reports/neuron/vk_devpick_ids.json}:$V/first/rows_B.jsonl:$V/oracle/rows_B_f$FM.jsonl" \
    "test3:vktest:$D/neuron/rows_B.jsonl::$D/first/rows_B.jsonl:$D/oracle/rows_B.jsonl"
