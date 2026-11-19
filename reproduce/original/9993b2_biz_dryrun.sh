#!/bin/bash
# Demo corpus A dry runs (reports/design_demo_rules.md §5; approved 2026-11-13; one attempt). Fictional companies
# (synthetic data generated for this demo). Any resemblance to real companies or people is unintended.
# Smoke pool artifacts/neuron/biz_smoke (60 questions, 12 dev-fit per cell; 539 companies). Code pinned at 099e99d
# (worktree t24-biz2). Pass = every check of experiments/neuron/biz_dryrun_check.py.
# Attempt 2 (owner 2026-11-13): 9993b FAIL rule-writing (checker compared file bytes; rows match by id on every field).
# VERDICT: artifacts/neuron/biz_dryrun2/biz_dryrun_check.json
set -e
cd /home/USER/projects/t24-biz2; export PYTHONPATH=$PWD; PY=.venv/bin/python
D=artifacts/neuron/biz_dryrun2; mkdir -p $D; cp -n artifacts/neuron/biz_smoke/pool.pkl artifacts/neuron/biz_smoke/ids.json $D/
[ -s $D/regress_mqs/regress.json ] || bash tools/regress_mqs_sets.sh $D/regress_mqs 7400
export MXN_VOCAB=biz
GP=artifacts/expert_biz/gold_plans.json; BP=$D/gold_plans_badrel.json
[ -s $BP ] || $PY experiments/neuron/biz_badrel_plans.py $GP $BP
R="$PY -u experiments/neuron/e2e_run.py artifacts/expert_biz artifacts/data/biz"
P="--pool=bizdev --poolfile=$D/pool.pkl --pick=artifacts/neuron/dev/opt_pick.json --fmins=0.0 --kind --rev=rel:0.1 --port=7400 --questions=$D/ids.json"
G="--goldplans=$GP"
[ -s $D/gold_h3/run.json ]   || $R $D/gold_h3 $P $G --hosts=3 --sets=0.3,1.0
[ -s $D/gold_h3b/run.json ]  || $R $D/gold_h3b $P $G --hosts=3 --sets=0.3,1.0
[ -s $D/h1/run.json ]        || $R $D/h1 $P $G --hosts=1 --sets=0.3
for a in all first oracle; do [ -s $D/$a/run.json ] || $R $D/$a $P $G --hosts=3 --sets=0.3 --arm=$a; done
[ -s $D/badrel_h3/run.json ] || $R $D/badrel_h3 $P --goldplans=$BP --hosts=3 --sets=0.3
[ -s $D/plan_h3/run.json ]   || $R $D/plan_h3 $P --hosts=3 --sets=0.3
$PY experiments/neuron/biz_dryrun_check.py $D artifacts/data/biz
