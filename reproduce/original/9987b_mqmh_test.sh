#!/bin/bash
# GATE: 9987a_mqmh_dev
# 9987b MetaQA 2- / 3-hop test (reverse lookup). Rules: log 2026-11-05 17:04 UTC (f871f38). One attempt.
# code pinned at the rules commit f871f38 (worktree t24-mh)
# Only the hop counts whose dev-pick gate passed (9987a rev_dev_pick.json); setting frozen from rev_pick.json.
# VERDICT: artifacts/neuron/mqmh_test/test_verdict.json
# (pass = some tested hop count FLIES)
set -e
cd /home/USER/projects/t24-mh; export PYTHONPATH=$PWD MXN_VOCAB=metaqa; PY=.venv/bin/python
D=artifacts/neuron/mqmh_test; V=artifacts/neuron/mqmh_dev; mkdir -p $D
PICK=artifacts/neuron/dev/opt_pick.json; FM=0.0
for i in $(seq 1 240); do
  m=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo); [ $m -ge 10 ] && break
  echo "$(date -u +%T) waiting for memory: ${m} GB available, need 10"; sleep 60
done
$PY -c "import json,sys;sys.exit(0 if json.load(open('$V/pick/rev_dev_pick.json'))['pass'] else 'no hop count passed dev-pick')"
B=$($PY -c "import json;d=json.load(open('$V/pick/rev_pick.json'));print(d['rule']+':'+str(d['theta']))")
TAG=$($PY -c "import json;print(json.load(open('$V/pick/rev_pick.json'))['tag'])")
HOPS=$($PY -c "import json;g=json.load(open('$V/pick/rev_dev_pick.json'))['gates'];print(' '.join(n[0] for n in ('2hop','3hop') if g[n]['pass']))")
echo "setting $B ($TAG); hop counts tested: $HOPS"
python3 -c "
import json; a=json.load(open('reports/neuron/mq2_test_ids.json')) if '2' in '$HOPS' else []
b=json.load(open('reports/neuron/mq3_test_ids.json')) if '3' in '$HOPS' else []
json.dump(a+b, open('$D/test_ids.json','w')); print(len(a), len(b))"
R="$PY -u experiments/neuron/e2e_run.py artifacts/expert_mqmh artifacts/data/metaqa_mh"
A="--pool=mqmhtest --port=7400 --pick=$PICK --kind --fmins=$FM --rev=$B --questions=$D/test_ids.json"
[ -s $D/neuron/run.json ]    || $R $D/neuron $A --hosts=5
[ -s $D/first/run.json ]     || $R $D/first $A --hosts=5 --arm=first
[ -s $D/neuron_h1/run.json ] || $R $D/neuron_h1 $A --hosts=1
python3 -c "import json;json.dump({'bars':{}, 'rule':'no bars (rules f871f38)'}, open('$D/no_bars.json','w'))"
S="$PY experiments/neuron/e2e_score.py artifacts/data/metaqa_mh mqmhtest"
for n in $HOPS; do
  $S $D/test_score_${n}hop.json --ids=reports/neuron/mq${n}_test_ids.json --tag=$TAG --neuron=$D/neuron \
     --first=$D/first --onehost=$D/neuron_h1 --bars=$D/no_bars.json
done
$PY -c "
import json; T={n: json.load(open(f'$D/test_score_{n}hop.json'))['tier'] for n in '$HOPS'.split()}
json.dump({'tiers': T, 'setting': '$TAG', 'pass': {'passed': any(t == 'FLIES' for t in T.values()), 'tiers': T}},
          open('$D/test_verdict.json', 'w'), indent=1); print(T)"
# report only: oracle ceiling on test, then drift (EM per arm and path-type mix, dev-pick → test) and cost
[ -s $D/oracle/run.json ] || $R $D/oracle $A --hosts=5 --arm=oracle
for n in $HOPS; do
  $S $D/oracle_test_score_${n}hop.json --ids=reports/neuron/mq${n}_test_ids.json --tag=$TAG --neuron=$D/neuron \
     --first=$D/first --oracle=$D/oracle
done
$PY -c "
import json, collections
out = {'post_hoc': False, 'note': 'report only (rules f871f38); nothing tuned on test', 'hops': {}}
for n in '$HOPS'.split():
    d = json.load(open(f'$V/pick/rev_dev_pick_{n}hop.json')); t = json.load(open(f'$D/oracle_test_score_{n}hop.json'))
    mix = lambda s: {k: round(100 * v['neuron']['n'] / s['n'], 1) for k, v in s['by_type'].items()}
    out['hops'][n] = {'em_devpick': {a: v['pct'] for a, v in d['em'].items()}, 'em_test': {a: v['pct'] for a, v in t['em'].items()},
                      'type_mix_devpick': mix(d), 'type_mix_test': mix(t)}
r = json.load(open('$D/neuron/run.json'))
out['cost'] = {k: v for k, v in r.items() if k.startswith('reverse_cost')}
json.dump(out, open('$D/drift_cost.json', 'w'), indent=1)"
