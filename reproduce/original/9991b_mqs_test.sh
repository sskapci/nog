#!/bin/bash
# GATE: 9991b0_mqs_allhosts
# 9991b MetaQA set answers / person heads test, one attempt. Rules: log "RULES MetaQA set answers and person heads"
# (b24b1ad, log 2026-11-06 23:25 UTC). Code: rules commit + row-streaming fix c928e63 + scorer fix 368ce83 (worktree t24-sr3, same as 9991a). Only the cells whose dev-pick gate passed (9991a), at
# the frozen theta_f: neuron (5 hosts), all fitting slots, first slot, neuron 1 host → tier per cell (set_score --tier);
# then the oracle (ceiling, report only) and drift. First slot and oracle on 2 host processes, as in 9991a (user decision
# 2026-11-07 option c: llama-server OOM-killed 3 times; identity shown by 9991a1). All fitting slots also on 2 hosts
# after its 5-host test run lost a host to the OOM killer (2026-11-08 19:49 UTC; identity shown by 9991b0; dev ran it on 5).
# VERDICT: artifacts/neuron/mqs_test/test_verdict.json
# (pass = some tested cell FLIES)
set -e
cd /home/USER/projects/t24-sr3; export PYTHONPATH=$PWD MXN_VOCAB=metaqa; PY=.venv/bin/python
D=artifacts/neuron/mqs_test; P=artifacts/neuron/mqs_dev/pick; mkdir -p $D
T=$($PY -c "import json;print(json.load(open('$P/set_pick.json'))['theta_f'])")
$PY -c "
import json
cells = json.load(open('$P/set_dev_pick.json'))['cells_to_test']
ids = [i for c in cells for i in json.load(open(f'reports/neuron/mqs_{c}_test_ids.json'))]
json.dump(ids, open('$D/test_ids.json', 'w')); print('cells', cells, 'questions', len(ids))"
R="$PY -u experiments/neuron/e2e_run.py artifacts/expert_mqs artifacts/data/metaqa_sets"
A="--pool=mqstest --port=7400 --pick=artifacts/neuron/dev/opt_pick.json --kind --fmins=0.0 --rev=rel:0.1 --sets=$T --questions=$D/test_ids.json"
for i in $(seq 1 240); do
  m=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo); [ $m -ge 10 ] && break
  echo "$(date -u +%T) waiting for memory: ${m} GB available, need 10"; sleep 60
done
[ -s $D/neuron/run.json ]    || $R $D/neuron $A --hosts=5
[ -s $D/all/run.json ]       || $R $D/all $A --hosts=2 --arm=all
[ -s $D/first/run.json ]     || $R $D/first $A --hosts=2 --arm=first
[ -s $D/neuron_h1/run.json ] || $R $D/neuron_h1 $A --hosts=1
$PY experiments/neuron/set_score.py artifacts/data/metaqa_sets $D/test_score.json --ids=$D/test_ids.json \
    --neuron=$D/neuron --all=$D/all --first=$D/first --onehost=$D/neuron_h1 --tier
$PY -c "
import json; s = json.load(open('$D/test_score.json'))
json.dump({'tiers': s['tiers'], 'theta_f': $T, 'pass': s['pass']}, open('$D/test_verdict.json', 'w'), indent=1)
print(s['tiers'])"
[ -s $D/oracle/run.json ] || $R $D/oracle $A --hosts=2 --arm=oracle
$PY experiments/neuron/set_score.py artifacts/data/metaqa_sets $D/test_score_oracle.json --ids=$D/test_ids.json \
    --neuron=$D/neuron --all=$D/all --first=$D/first --oracle=$D/oracle --onehost=$D/neuron_h1 --tier
