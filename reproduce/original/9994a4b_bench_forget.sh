#!/bin/bash
# 9994a4b Forgetting benchmark on corpus A (reports/design_forget_edit.md §1; rules 5799f50, worktree t24-bench4).
# Fictional companies (synthetic data generated for this demo). Any resemblance to real companies or people is unintended.
# 5 hosts, gold plans ("gold plans" on every number), theta_f 0.7, cold copies, every dev and test question (12,938);
# nested forget sets 30 / 150 / 300 companies. Pass = per set: leaks 0, retain rows 100% identical, audit total 0.
# Attempt 2 (standing authority): 9994a4 FAIL, rule-writing (retain set by gold path only; rows a forgotten KU had wrongly answered changed by design; strict retain 100.000% at every stage, post hoc).
# VERDICT: artifacts/bench_biz/forget2/score.json
set -e
cd /home/USER/projects/t24-bench4; export PYTHONPATH=$PWD MXN_VOCAB=biz; PY=.venv/bin/python
B=artifacts/bench_biz; D=$B/forget2
[ -s $D/run.json ] || $PY -u experiments/neuron/e2e_run.py artifacts/expert_biz $B/data $D --pool=bizdev --poolfile=$B/pool_all.pkl \
    --pick=artifacts/neuron/dev/opt_pick.json --kind --fmins=0.0 --rev=rel:0.1 --sets=0.7 --port=7400 --hosts=5 \
    --goldplans=$B/gold_plans.json --questions=$B/questions_all.json --bench=forget --plan=$B/forget_plan.json --cold
$PY experiments/neuron/bench_score.py forget $D $B/forget_plan.json $B/data $D/score.json
