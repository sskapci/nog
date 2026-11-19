#!/bin/bash
# 9994g RAG on demo corpus A, test — baseline (not neurons) (reports/design_rag_rules.md §3; rules 2cc957e, worktree
# t24-ragc). Both models at the dev k (9994f), the fixed seeded subset of 630 test questions (126 per cell, seed
# 20261116, reports/rag/biz_test630.json; held-out wordings), paired with 9993d neuron rows (planner) on the 3 tested cells (exact-set McNemar,
# Wilson 95%). Fictional companies (synthetic data generated for this demo). Any resemblance to real companies or people is unintended. Runs alone in the queue (~18 h). Pass = every question answered by both models.
# GATE: 9994f_rag_biz_dev
# VERDICT: artifacts/baselines/rag/biz/test_score.json
set -e
cd /home/USER/projects/t24-ragc; export PYTHONPATH=$PWD HF_HUB_OFFLINE=1; PY=.venv/bin/python
D=artifacts/baselines/rag/biz
S=artifacts/llamacpp/b11433/bin/llama-b11433/llama-server; G=artifacts/gguf
K=$($PY -c "import json;print(json.load(open('$D/kpick.json'))['k'])")
A="--server=$S --threads=8 --port=7410 --k=$K"
$PY mxn/baselines/rag.py answer $D reports/rag/biz_test630.json $D/test_r15.jsonl $A --model=$G/qwen2.5-1.5b-instruct-q8_0.gguf
$PY mxn/baselines/rag.py answer $D reports/rag/biz_test630.json $D/test_r7.jsonl $A --model=$G/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf
$PY experiments/baselines/rag_score.py artifacts/data/biz reports/rag/biz_test630.json $D/test_score.json \
    --rag=r15:$D/test_r15.jsonl --rag=r7:$D/test_r7.jsonl --neuron=artifacts/neuron/biz_test/neuron/rows_B.jsonl --notice
