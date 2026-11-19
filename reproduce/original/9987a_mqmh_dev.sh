#!/bin/bash
# 9987a MetaQA 2- / 3-hop dev (reverse lookup). Rules: log 2026-11-05 17:04 UTC (f871f38).
# Gate (decision 2026-11-05, 6a2b5d7): 9985b tier FLIES or PARTLY. The gate daemon releases on PASS only, so this job
# is queued directly after the rules commit and checks the tier itself.
# code pinned at the rules commit f871f38 (worktree t24-mh, created 2026-11-05 17:05 before this job ran)
# VERDICT: artifacts/neuron/mqmh_dev/pick/rev_dev_pick.json
# (pass = some hop count's dev-pick gate passed)
set -e
T=/home/USER/projects/t24-mq1/artifacts/neuron/mq_test/e2e_test_score.json
python3 -c "import json,sys;t=json.load(open('$T'))['tier'];print('9985b tier',t);sys.exit(0 if t in ('FLIES','PARTLY') else 'gate: 9985b '+t)"
cd /home/USER/projects/t24-mh; export PYTHONPATH=$PWD MXN_VOCAB=metaqa; PY=.venv/bin/python
D=artifacts/neuron/mqmh_dev; mkdir -p $D
PICK=artifacts/neuron/dev/opt_pick.json; FM=0.0
GRID=rel:0.05,rel:0.1,rel:0.2,rel:0.3,rel:0.5,rel:0.7,rel:1.0,abs:0.05,abs:0.1,abs:0.2,abs:0.3,abs:0.5,abs:0.7,abs:1.0
TAGS=$(echo $GRID | tr -d ':')
R="$PY -u experiments/neuron/e2e_run.py artifacts/expert_mqmh artifacts/data/metaqa_mh"
A="--pool=mqmhdev --port=7400 --pick=$PICK --kind --fmins=$FM"
S="$PY experiments/neuron/rev_dev_pick.py artifacts/data/metaqa_mh $D reports/neuron/mq2_devfit_ids.json \
   reports/neuron/mq2_devpick_ids.json reports/neuron/mq3_devfit_ids.json reports/neuron/mq3_devpick_ids.json $D/pick \
   --pool=mqmhdev --grid=$TAGS"
for i in $(seq 1 240); do
  m=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo); [ $m -ge 10 ] && break
  echo "$(date -u +%T) waiting for memory: ${m} GB available, need 10"; sleep 60
done
# stage fit: neuron arm, 5 hosts, whole grid → rule and theta on dev-fit
[ -s $D/neuron/run.json ] || $R $D/neuron $A --hosts=5 --rev=$GRID
[ -s $D/pick/rev_pick.json ] || $S --stage=fit
B=$($PY -c "import json;d=json.load(open('$D/pick/rev_pick.json'));print(d['rule']+':'+str(d['theta']))")
# stage pick: first (control), oracle (ceiling), neuron on 1 host at the chosen setting → gates per hop count
[ -s $D/first/run.json ]     || $R $D/first $A --hosts=5 --arm=first --rev=$B
[ -s $D/oracle/run.json ]    || $R $D/oracle $A --hosts=5 --arm=oracle --rev=$B
[ -s $D/neuron_h1/run.json ] || $R $D/neuron_h1 $A --hosts=1 --rev=$B
$S --stage=pick
