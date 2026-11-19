#!/bin/bash
# 9994e RAG arms of the forgetting / updating benchmarks on demo corpus A — baseline (not neurons)
# (reports/design_forget_edit.md §7; RAG rules reports/design_rag_rules.md; rules 9a759b0, worktree t24-ragb).
# Fictional companies (synthetic data generated for this demo). Any resemblance to real companies or people is unintended.
# k from {1, 3, 5, 10} by corpus A dev recall@k (retriever alone, rag_prep.py kpick, biz cells); indexes: base (3,000
# documents), forget (the 300-company set removed), edit (500 documents rewritten); both models; 90 forget-set questions
# (60 forget, 30 retain) before / after, 90 edit questions (60 efficacy, 30 neighbour) before / after. Runs alone.
# Pass = every question answered in every pass by both models.
# VERDICT: artifacts/baselines/rag/bench/check.json
set -e
cd /home/USER/projects/t24-ragb; export PYTHONPATH=$PWD HF_HUB_OFFLINE=1; PY=.venv/bin/python
D=artifacts/baselines/rag/bench; B=artifacts/bench_biz; mkdir -p $D
S=artifacts/llamacpp/b11433/bin/llama-b11433/llama-server; G=artifacts/gguf
[ -s $D/meta.json ] || $PY experiments/baselines/rag_bench.py prep artifacts/demo/a3 $B $D
for x in base forget edit; do [ -s $D/idx_$x/index.faiss ] || $PY mxn/baselines/rag.py build $D/pool_$x.pkl $D/idx_$x; done
[ -s $D/kpick.json ] || $PY experiments/baselines/rag_prep.py kpick $D/idx_base artifacts/data/biz reports/neuron biz \
    1h_company,1h_year,1h_value,2h_company,2h_value $D/kpick.json
K=$($PY -c "import json;print(json.load(open('$D/kpick.json'))['k'])")
A="--server=$S --threads=8 --port=7410 --k=$K"
for m in r15 r7; do
  M=$([ $m = r15 ] && echo $G/qwen2.5-1.5b-instruct-q8_0.gguf || echo $G/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf)
  $PY mxn/baselines/rag.py answer $D/idx_base $D/q_forget.json $D/forget_before_$m.jsonl $A --model=$M
  $PY mxn/baselines/rag.py answer $D/idx_forget $D/q_forget.json $D/forget_after_$m.jsonl $A --model=$M
  $PY mxn/baselines/rag.py answer $D/idx_base $D/q_edit.json $D/edit_before_$m.jsonl $A --model=$M
  $PY mxn/baselines/rag.py answer $D/idx_edit $D/q_edit.json $D/edit_after_$m.jsonl $A --model=$M
  $PY experiments/baselines/rag_bench.py score $B $D $m
done
$PY -c "
import json
D = '$D'; n = lambda f: sum(1 for _ in open(f'{D}/{f}'))
ok = all(n(f'{p}_{m}.jsonl') == 90 for p in ('forget_before', 'forget_after', 'edit_before', 'edit_after') for m in ('r15', 'r7'))
json.dump({'note': 'baseline (not neurons)', 'k': json.load(open(f'{D}/kpick.json'))['k'], 'pass': ok}, open(f'{D}/check.json', 'w'), indent=1)"
