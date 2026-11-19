#!/bin/bash
# 9994a5 Updating benchmark on corpus A (reports/design_forget_edit.md §2; rules 0519f4a, worktree t24-bench3).
# Fictional companies (synthetic data generated for this demo). Any resemblance to real companies or people is unintended.
# 5 hosts, gold plans ("gold plans" on every number), theta_f 0.7, 500 edits on distinct companies, every dev and test
# question plus the generalization wordings (14,248). Pass = every edit retrained by one host and neighbour rows 100%
# identical; efficacy, generalization, multi-hop, own specificity and seconds reported (founded_year apart).
# GATE: 9994a3b_bench_dry
# VERDICT: artifacts/bench_biz/edit/score.json
set -e
cd /home/USER/projects/t24-bench3; export PYTHONPATH=$PWD MXN_VOCAB=biz; PY=.venv/bin/python
B=artifacts/bench_biz; D=$B/edit
[ -s $D/run.json ] || $PY -u experiments/neuron/e2e_run.py artifacts/expert_biz $B/data $D --pool=bizdev --poolfile=$B/pool_all.pkl \
    --pick=artifacts/neuron/dev/opt_pick.json --kind --fmins=0.0 --rev=rel:0.1 --sets=0.7 --port=7400 --hosts=5 \
    --goldplans=$B/gold_plans.json --questions=$B/questions_edit.json --bench=edit --plan=$B/edit_plan.json
$PY experiments/neuron/bench_score.py edit $D $B/edit_plan.json $D/score.json
