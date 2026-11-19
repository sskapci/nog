#!/bin/bash
# Demo corpus A planner (reports/design_demo_rules.md §3; approved 2026-11-13). Fictional companies (synthetic data
# generated for this demo). Any resemblance to real companies or people is unintended.
# Planner v2 recipe, templates chain / prev, trained on corpus A train-split questions only (biz_fit 17,395; hold-out
# biz_mdev 1,932; train wordings only, wording check in demo_prep.py), 2 h. Code pinned at 6bc7678 (worktree t24-biz).
# Pass = planner.pt written and hold-out exact >= 90%.
# VERDICT: artifacts/expert_biz/planner2/train_ok.json
set -e
cd /home/USER/projects/t24-biz; export PYTHONPATH=$PWD MXN_VOCAB=biz; PY=.venv/bin/python
test ! -e artifacts/expert_biz/planner2/planner.pt
$PY -u experiments/expert/planner_run.py artifacts/expert_biz artifacts/expert/enc artifacts/expert_biz/planner2 --stage=train --hours=2 --plans=biz
$PY -c "
import json, re, os
log = open('artifacts/expert_biz/planner2/planner.log').read()
m = re.findall(r'best step (\d+) mdev exact ([\d.]+)', log)
best = float(m[-1][1]) if m else 0.0
json.dump({'notice': 'Fictional companies (synthetic data generated for this demo). Any resemblance to real companies or people is unintended.',
           'pass': os.path.exists('artifacts/expert_biz/planner2/planner.pt') and best >= 90.0, 'best': m[-1] if m else None,
           'rule': 'hold-out exact >= 90%'}, open('artifacts/expert_biz/planner2/train_ok.json', 'w'), indent=1)"
