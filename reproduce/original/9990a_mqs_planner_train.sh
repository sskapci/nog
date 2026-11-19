#!/bin/bash
# MetaQA set answers / person heads (reports/design_metaqa_sets_persons.md; build 63ae37a + 8af9fb8, log "BUILD MetaQA set answers
# and person heads"): planner v2 recipe with templates chain / rev / prev and the r3 head, trained on MetaQA train only
# (mqs_fit 273,106; hold-out mqs_mdev 30,345), 2 h. Code pinned at 8af9fb8 (worktree t24-sb).
# Pass = planner.pt written and hold-out exact >= 90%.
# VERDICT: artifacts/expert_mqs/planner2/train_ok.json
set -e
cd /home/USER/projects/t24-sb; export PYTHONPATH=$PWD MXN_VOCAB=metaqa; PY=.venv/bin/python
test ! -e artifacts/expert_mqs/planner2/planner.pt
$PY -u experiments/expert/planner_run.py artifacts/expert_mqs artifacts/expert/enc artifacts/expert_mqs/planner2 --stage=train --hours=2 --plans=mqs
$PY -c "
import json, re, os
log = open('artifacts/expert_mqs/planner2/planner.log').read()
m = re.findall(r'best step (\d+) mdev exact ([\d.]+)', log)
best = float(m[-1][1]) if m else 0.0
json.dump({'pass': os.path.exists('artifacts/expert_mqs/planner2/planner.pt') and best >= 90.0, 'best': m[-1] if m else None,
           'rule': 'hold-out exact >= 90%'}, open('artifacts/expert_mqs/planner2/train_ok.json', 'w'), indent=1)"
