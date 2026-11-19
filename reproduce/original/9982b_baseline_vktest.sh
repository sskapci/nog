#!/bin/bash
# Baseline (not neurons) on test3, step 2 of 2: frozen chapter 2 system, base mode, all 2,000 vktest questions; then
# paired with the 9979 neuron rows by e2e_score.py --baseline into a separate file (the 9979 tier is not touched).
# Labeled "baseline (not neurons)". Report only; pass = rows complete and audit clean.
# VERDICT: artifacts/neuron/vk_test/baseline_ok.json
set -e
cd /home/USER/projects/t24-conn; export PYTHONPATH=$PWD
.venv/bin/python -u experiments/expert/conn_run.py artifacts/data/2wiki/extracted artifacts/expert --pool=vktest --stage=run --mode=base
cd /home/USER/projects/t24-e2e; export PYTHONPATH=$PWD; PY=.venv/bin/python; D=artifacts/neuron/vk_test
B=artifacts/expert/connections/vktest
$PY experiments/neuron/e2e_score.py artifacts/data/2wiki/extracted vktest $D/baseline_test_score.json --neuron=$D/neuron \
    --first=$D/first --oracle=$D/oracle --baseline=$B/eval_rows_base.jsonl
$PY - <<'P'
import json
B = "artifacts/expert/connections/vktest"
rows = [json.loads(l) for l in open(f"{B}/eval_rows_base.jsonl")]
run = json.load(open(f"{B}/run_base.json"))
s = json.load(open("artifacts/neuron/vk_test/baseline_test_score.json"))
ok = {"rows_2000": len({r["id"] for r in rows}) == 2000, "audit_clean": bool(run.get("audit_clean")),
      "scored": "baseline" in s["em"]}
json.dump({"checks": ok, "pass": all(ok.values()), "label": "baseline (not neurons)"},
          open("artifacts/neuron/vk_test/baseline_ok.json", "w"), indent=1); print(ok)
P
