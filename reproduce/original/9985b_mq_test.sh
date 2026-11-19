#!/bin/bash
# GATE: 9985a_mq_dev
# MetaQA first version, TEST, one attempt (rules: log "RULES MetaQA first version", 49efe13): mqtest pool (18,127 KUs,
# 2,000 MetaQA 1-hop test questions), MXN_VOCAB=metaqa, 5 hosts, --kind, dev choices frozen: neuron, first, neuron on
# 1 host -> tier with the dev bars. Then (report only, after the tier is written): oracle ceiling and drift.
# VERDICT: artifacts/neuron/mq_test/e2e_test_score.json
set -e
# code pinned at the rules commit 49efe13 (worktree t24-mq1, created 2026-11-04 19:22 before this job ran)
cd /home/USER/projects/t24-mq1; export PYTHONPATH=$PWD MXN_VOCAB=metaqa; PY=.venv/bin/python
D=artifacts/neuron/mq_test; P=artifacts/neuron/mq_dev/pick; V=artifacts/neuron/mq_dev
PICK=artifacts/neuron/dev/opt_pick.json; mkdir -p $D
for i in $(seq 1 240); do
  m=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo); [ $m -ge 10 ] && break
  echo "$(date -u +%T) waiting for memory: ${m} GB available, need 10"; sleep 60
done
[ $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) -ge 10 ] || { echo "not enough memory after 4 h"; exit 3; }
$PY -c "import json,sys;sys.exit(0 if json.load(open('$P/e2e_dev_pick.json'))['pass'] else 'dev-pick gate not passed')"
FM=$($PY -c "import json;print(json.load(open('$P/e2e_fmin.json'))['fmin'])")
R="$PY -u experiments/neuron/e2e_run.py artifacts/expert_mq artifacts/data/metaqa_2w"
A="--pool=mqtest --port=7400 --pick=$PICK --kind --tau=$P/e2e_tau.json"
[ -s $D/neuron/run.json ]    || $R $D/neuron $A --hosts=5 --fmins=$FM
[ -s $D/first/run.json ]     || $R $D/first $A --hosts=5 --arm=first
[ -s $D/neuron_h1/run.json ] || $R $D/neuron_h1 $A --hosts=1 --fmins=$FM
$PY experiments/neuron/e2e_score.py artifacts/data/metaqa_2w mqtest $D/e2e_test_score.json --neuron=$D/neuron \
    --first=$D/first --onehost=$D/neuron_h1 --bars=$P/e2e_bars.json
# report only (committed in the rules; the tier above is final before this runs)
[ -s $D/oracle/run.json ]    || $R $D/oracle $A --hosts=5 --arm=oracle --fmins=$FM
$PY experiments/neuron/e2e_score.py artifacts/data/metaqa_2w mqtest $D/oracle_test_score.json --neuron=$D/neuron \
    --first=$D/first --oracle=$D/oracle
$PY experiments/neuron/e2e_drift.py artifacts/data/metaqa_2w artifacts/expert_mq $D/drift.json \
    "devpick:mqdev:$V/neuron/rows_B_f$FM.jsonl:reports/neuron/mq_devpick_ids.json:$V/first/rows_B.jsonl:$V/oracle/rows_B_f$FM.jsonl" \
    "test:mqtest:$D/neuron/rows_B.jsonl::$D/first/rows_B.jsonl:$D/oracle/rows_B.jsonl"
