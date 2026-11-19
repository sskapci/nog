#!/bin/bash
# 9994a3b Forget / edit benchmark dry runs (reports/design_forget_edit.md §1–2; rules 0519f4a, worktree t24-bench3;
# one attempt). Fictional companies (synthetic data generated for this demo). Any resemblance to real companies or
# people is unintended. Corpus a3, 5 hosts, gold plans, theta_f 0.7, 300 seeded questions (+ edit questions); forget:
# the 30-company set; edit: the first 20 edits. Runs: forget (cold copies), forget_b (noise floor), forget_keepcold
# (known-bad), forget_noscrub (known-bad), edit, edit_notrain (known-bad). Pass = every check of bench_dry_check.py.
# Attempt 2 (standing authority 2026-11-15): 9994a3 FAIL, rule-writing (edit efficacy bar not conditioned on the KU knowing the fact; 17 / 17 known-fact edits changed).
# VERDICT: artifacts/bench_biz/dry2/bench_dry_check.json
set -e
cd /home/USER/projects/t24-bench3; export PYTHONPATH=$PWD MXN_VOCAB=biz; PY=.venv/bin/python
B=artifacts/bench_biz; D=$B/dry2; mkdir -p $D
$PY - <<'PY'
import json, random
B = "artifacts/bench_biz"; D = B + "/dry2"
qa = json.load(open(f"{B}/questions_all.json")); random.Random(11).shuffle(qa)
F = json.load(open(f"{B}/forget_plan.json"))
q = sorted(set(qa[:240]) | set(F["sets"]["30"]["forget_questions"][:60])); json.dump(q, open(f"{D}/q_forget.json", "w"))
F["sets"] = {"30": F["sets"]["30"]}; json.dump(F, open(f"{D}/forget_plan.json", "w"))
E = json.load(open(f"{B}/edit_plan.json")); E["edits"] = E["edits"][:20]
keep = {e["efficacy_qid"] for e in E["edits"]} | set(qa[:240])
eds = {e["title"] for e in E["edits"]}
E["class"] = {k: v for k, v in E["class"].items() if k in keep}
json.dump(E, open(f"{D}/edit_plan.json", "w")); json.dump(sorted(keep), open(f"{D}/q_edit.json", "w"))
PY
R="$PY -u experiments/neuron/e2e_run.py artifacts/expert_biz $B/data"
A="--pool=bizdev --poolfile=$B/pool_all.pkl --pick=artifacts/neuron/dev/opt_pick.json --kind --fmins=0.0 --rev=rel:0.1 --sets=0.7 --port=7400 --hosts=5 --goldplans=$B/gold_plans.json"
F="--questions=$D/q_forget.json --bench=forget --plan=$D/forget_plan.json --cold"
E="--questions=$D/q_edit.json --bench=edit --plan=$D/edit_plan.json"
[ -s $D/forget/run.json ]          || $R $D/forget $A $F
[ -s $D/forget_b/run.json ]        || $R $D/forget_b $A $F
[ -s $D/forget_keepcold/run.json ] || NHOST_FORGET_KEEP_COLD=1 $R $D/forget_keepcold $A $F
[ -s $D/forget_noscrub/run.json ]  || NHOST_FORGET_NO_SCRUB=1 $R $D/forget_noscrub $A $F
[ -s $D/edit/run.json ]            || $R $D/edit $A $E
[ -s $D/edit_notrain/run.json ]    || NHOST_EDIT_NO_TRAIN=1 $R $D/edit_notrain $A $E
$PY experiments/neuron/bench_score.py forget $D/forget $D/forget_plan.json $B/data $D/f_good.json
$PY experiments/neuron/bench_score.py edit $D/edit $D/edit_plan.json $D/e_good.json || true
$PY experiments/neuron/bench_dry_check.py $D
