#!/bin/bash
# 9994c RAG on MetaQA, test — baseline (not neurons) (reports/design_rag_rules.md §3; rules c05e28d, worktree
# t24-rag2). Both models at the dev k (9994b), the fixed seeded subset of 630 test questions (105 per cell, seed
# 20261115, reports/rag/mqs_test630.json), paired with 9991b's neuron rows on the same questions (exact-set McNemar,
# Wilson 95%). Runs alone in the queue (~20.5 h). Pass = every question answered by both models.
# GATE: 9994b_rag_mq_dev
# VERDICT: artifacts/baselines/rag/mqs/test_score.json
set -e
cd /home/USER/projects/t24-rag2; export PYTHONPATH=$PWD HF_HUB_OFFLINE=1; PY=.venv/bin/python
D=artifacts/baselines/rag/mqs
S=artifacts/llamacpp/b11433/bin/llama-b11433/llama-server; G=artifacts/gguf
K=$($PY -c "import json;print(json.load(open('$D/kpick.json'))['k'])")
A="--server=$S --threads=8 --port=7410 --k=$K"
$PY mxn/baselines/rag.py answer $D reports/rag/mqs_test630.json $D/test_r15.jsonl $A --model=$G/qwen2.5-1.5b-instruct-q8_0.gguf
$PY mxn/baselines/rag.py answer $D reports/rag/mqs_test630.json $D/test_r7.jsonl $A --model=$G/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf
$PY experiments/baselines/rag_score.py artifacts/data/metaqa_sets reports/rag/mqs_test630.json $D/test_score.json \
    --rag=r15:$D/test_r15.jsonl --rag=r7:$D/test_r7.jsonl --neuron=artifacts/neuron/mqs_test/neuron/rows_B.jsonl
