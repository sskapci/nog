#!/bin/bash
# MetaQA first version, DEV (rules: log "RULES MetaQA first version", 49efe13; design reports/design_metaqa.md):
# mqdev pool (18,127 KUs, 4,480 MetaQA 1-hop dev questions), MXN_VOCAB=metaqa, 5 hosts, --kind: neuron (fmin grid),
# first (control), oracle (ceiling, measurement only), neuron on 1 host; choices on dev-fit, gate on dev-pick.
# VERDICT: artifacts/neuron/mq_dev/pick/e2e_dev_pick.json
set -e
# code pinned at the rules commit 49efe13 (worktree t24-mq1, created 2026-11-04 19:22 before this job ran)
cd /home/USER/projects/t24-mq1; export PYTHONPATH=$PWD MXN_VOCAB=metaqa; PY=.venv/bin/python
D=artifacts/neuron/mq_dev; PICK=artifacts/neuron/dev/opt_pick.json; mkdir -p $D
for i in $(seq 1 240); do                           # memory guard: 5 hosts on 18,127 KUs need about 10 GB
  m=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo); [ $m -ge 10 ] && break
  echo "$(date -u +%T) waiting for memory: ${m} GB available, need 10"; sleep 60
done
[ $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) -ge 10 ] || { echo "not enough memory after 4 h"; exit 3; }
R="$PY -u experiments/neuron/e2e_run.py artifacts/expert_mq artifacts/data/metaqa_2w"
G=0,0.1,0.25,0.5; A="--pool=mqdev --port=7400 --pick=$PICK --kind"
[ -s $D/neuron/run.json ]    || $R $D/neuron $A --hosts=5 --fmins=$G
[ -s $D/first/run.json ]     || $R $D/first $A --hosts=5 --arm=first
[ -s $D/oracle/run.json ]    || $R $D/oracle $A --hosts=5 --arm=oracle --fmins=$G
[ -s $D/neuron_h1/run.json ] || $R $D/neuron_h1 $A --hosts=1 --fmins=$G
$PY experiments/neuron/e2e_dev_pick.py artifacts/data/metaqa_2w $D reports/neuron/mq_devfit_ids.json \
    reports/neuron/mq_devpick_ids.json $D/pick --pool=mqdev
