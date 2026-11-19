#!/bin/bash
# 9994b RAG on MetaQA, dev stage — baseline (not neurons) (reports/design_rag_rules.md §2, §4; rules c05e28d,
# worktree t24-rag2). Index of all 18,127 MetaQA documents (bge-small-en-v1.5, FAISS flat IP); k from {1, 3, 5, 10} by
# dev recall@k of the retriever alone (experiments/baselines/rag_prep.py kpick); dev confirmation at that k, both
# models (1.5B Q8, 7B Q4), 100 dev-pick questions (reports/rag/mqs_dev100.json). Runs alone in the queue.
# Pass = k chosen (fits 8,192 tokens) and every dev question answered by both models. Dev numbers: report only.
# VERDICT: artifacts/baselines/rag/mqs/dev_check.json
set -e
cd /home/USER/projects/t24-rag2; export PYTHONPATH=$PWD HF_HUB_OFFLINE=1; PY=.venv/bin/python
D=artifacts/baselines/rag/mqs; mkdir -p $D
S=artifacts/llamacpp/b11433/bin/llama-b11433/llama-server; G=artifacts/gguf
C=1h_now,1h_sets,2h_person,2h_person_sets,2h_sets,3h_sets
[ -s $D/index.faiss ] || $PY mxn/baselines/rag.py build artifacts/expert_mqs/pool_mqstest.pkl $D
[ -s $D/kpick.json ] || $PY experiments/baselines/rag_prep.py kpick $D artifacts/data/metaqa_sets reports/neuron mqs $C $D/kpick.json
K=$($PY -c "import json;print(json.load(open('$D/kpick.json'))['k'])")
A="--server=$S --threads=8 --port=7410 --k=$K"
$PY mxn/baselines/rag.py answer $D reports/rag/mqs_dev100.json $D/dev_r15.jsonl $A --model=$G/qwen2.5-1.5b-instruct-q8_0.gguf
$PY mxn/baselines/rag.py answer $D reports/rag/mqs_dev100.json $D/dev_r7.jsonl $A --model=$G/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf
$PY experiments/baselines/rag_score.py artifacts/data/metaqa_sets reports/rag/mqs_dev100.json $D/dev_score.json \
    --rag=r15:$D/dev_r15.jsonl --rag=r7:$D/dev_r7.jsonl
$PY -c "
import json
k = json.load(open('$D/kpick.json')); s = json.load(open('$D/dev_score.json'))
json.dump({'note': 'baseline (not neurons); dev, report only', 'k': k['k'], 'pass': bool(k['pass'] and s['pass'])},
          open('$D/dev_check.json', 'w'), indent=1)"
