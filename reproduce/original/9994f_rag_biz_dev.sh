#!/bin/bash
# 9994f RAG on demo corpus A, dev stage — baseline (not neurons) (reports/design_rag_rules.md §2, §4; rules 2cc957e,
# worktree t24-ragc). Fictional companies (synthetic data generated for this demo). Any resemblance to real companies or people is unintended.
# Index of the 3,000 corpus A documents (bge-small-en-v1.5, FAISS flat IP); k from {1, 3, 5, 10} by
# dev recall@k of the retriever alone (experiments/baselines/rag_prep.py kpick); dev confirmation at that k, both
# models (1.5B Q8, 7B Q4), 100 dev-pick questions (reports/rag/biz_dev100.json). Runs alone in the queue.
# Pass = k chosen (fits 8,192 tokens) and every dev question answered by both models. Dev numbers: report only.
# VERDICT: artifacts/baselines/rag/biz/dev_check.json
set -e
cd /home/USER/projects/t24-ragc; export PYTHONPATH=$PWD HF_HUB_OFFLINE=1; PY=.venv/bin/python
D=artifacts/baselines/rag/biz; mkdir -p $D
S=artifacts/llamacpp/b11433/bin/llama-b11433/llama-server; G=artifacts/gguf
C=1h_company,1h_year,1h_value,2h_company,2h_value
[ -s $D/index.faiss ] || $PY mxn/baselines/rag.py build artifacts/expert_biz/pool_biztest.pkl $D
[ -s $D/kpick.json ] || $PY experiments/baselines/rag_prep.py kpick $D artifacts/data/biz reports/neuron biz $C $D/kpick.json
K=$($PY -c "import json;print(json.load(open('$D/kpick.json'))['k'])")
A="--server=$S --threads=8 --port=7410 --k=$K"
$PY mxn/baselines/rag.py answer $D reports/rag/biz_dev100.json $D/dev_r15.jsonl $A --model=$G/qwen2.5-1.5b-instruct-q8_0.gguf
$PY mxn/baselines/rag.py answer $D reports/rag/biz_dev100.json $D/dev_r7.jsonl $A --model=$G/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf
$PY experiments/baselines/rag_score.py artifacts/data/biz reports/rag/biz_dev100.json $D/dev_score.json --notice \
    --rag=r15:$D/dev_r15.jsonl --rag=r7:$D/dev_r7.jsonl
$PY -c "
import json
k = json.load(open('$D/kpick.json')); s = json.load(open('$D/dev_score.json'))
json.dump({'note': 'baseline (not neurons); dev, report only', 'k': k['k'], 'pass': bool(k['pass'] and s['pass'])},
          open('$D/dev_check.json', 'w'), indent=1)"
