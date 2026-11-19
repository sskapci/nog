# Rules for job 9994c_rag_mq_test
Headline result: §6 RAG MetaQA test (baseline, not neurons). Job script as it ran: `reproduce/original/9994c_rag_mq_test.sh`; code: commit c05e28d (release archive `mxn-src-c05e28d.tar.gz`). Rules commit: c05e28d. Verdict file: `9994c_rag_mq_test.verdict.json` (the job's `# VERDICT:` file, `artifacts/baselines/rag/mqs/test_score.json`). Commit hashes and log entries refer to the private development repository; the text below is copied from it unchanged.

---

# Experiment log entries (rules and amendments committed before the run)

## 2026-11-15 23:05Z — DECISION (owner): option (b) + standing authority; corpus A test reported as is; RAG rules committed

Owner 2026-11-15: (b) report corpus A's test as it stands; commit the RAG rules (630 per test set, both models,
~47.6 h) and queue the MetaQA RAG runs; start the forgetting and updating benchmarks (FORGET, EDIT: build, rules, dry
runs, runs) on questions / cells where the planner works or with gold plans, labelled clearly; planner wording
coverage later as a separate track (hand-written paraphrase grammar, never the test wordings, new rules + fresh test
set). **Standing authority**: decide and proceed within the approved plan, principles and checks; ask first only for
changes to principles / the claim, anything public, money or deleting data, fresh test sets outside the plan, other
people's processes; never idle on fillers.
**Corpus A test, reported as is** (fictional companies, synthetic): 1h_company FLIES (neuron 24.45% [22.6, 26.4] vs all
0.0%, first 6.65%); 2h_company and 2h_value 0.0% (do not fly). Cause: the planner mis-plans the held-out test wordings
(0 plans right for ceo, hq, product, parent, subsidiary and every 2-hop type). The oracle arm (gold triples) also falls
with those plans; on dev (train-style wordings) the oracle answers 96–100% and the neuron arm 77.3 / 80.1 / 18.8% on the same cells,
so the KUs hold the knowledge; the planner's paraphrase coverage is the limit. Neuron mechanics held (K, audits, 5 = 1).
**RAG rules commit** (this commit): subsets reports/rag/mqs_test630.json / mqs_dev100.json, tools rag_prep.py (subsets,
kpick) and rag_score.py (tested known-good / known-bad on 9992a outputs), jobs 9994b (dev) and 9994c (test, GATE 9994b).


---

# `design_rag_rules.md` as committed in c05e28d

## RAG baseline on MetaQA test: rules draft — baseline (not neurons)

Status: **rules commit (owner 2026-11-15: commit the RAG rules and queue the MetaQA runs)**. Owner decisions D1 (2026-11-09) and 2026-11-12
(scale, replaced 2026-11-13: seeded subset of 2,000 per test set, both models, target <= ~48 h, §3–4;
retrieval settings fixed on dev only; per-question CPU and wall for RAG and neurons on the same hardware).
Code: mxn/baselines/rag.py (8514317) plus the parse change below. Dry runs: 9992a (FAIL, log 2026-11-12 18:1xZ).

### 1. What 9992a showed (30 dev questions of the 9990b dry-run set; dev, report only)

| arm | exact (parse v1) | exact (parse v2, post hoc) | wall mean s | server CPU mean s |
|---|---|---|---|---|
| 1.5B retrieval k 5 | 4 / 30 | 4 / 30 | 37.7 | 154.7 |
| 1.5B retrieval, again | 4 / 30 (raw replies 30 / 30 identical) | 4 / 30 | 26.3 | 122.1 |
| 1.5B gold passages | 3 / 30 | 3 / 30 | 24.1 | 110.6 |
| 1.5B no passages | 0 / 30 (29 "unknown") | 0 / 30 | 8.7 | 36.2 |
| 7B retrieval k 5 | 0 / 30 | 5 / 30 | 81.2 | 301.4 |

FAIL reasons and class:
- Known-bad not all empty: 1 / 30 answered from pretraining knowledge ("Peter Franzén appears in The Girl with the
  Dragon Tattoo ..."): **rule-writing** (the rule assumed the reader never answers without passages); the leak is a
  real property of RAG on famous films and is reported as such.
- Gold passages 3 < retrieval 4: **real** (small 1.5B reader; 30 questions; difference 1 question).
- 7B 0 exact: **rule-writing** (parse): 7B appends " | unknown" to answers; parse v1 kept "unknown" as an answer.
Wall time varied between identical runs (37.7 vs 26.3 s): fillers shared the CPU; test runs are exclusive (below).

### 2. Fixed settings (post hoc from 9992a where stated)

- Prompt: unchanged (rag.py PROMPT). Greedy. max 96 new tokens. Context 8,192 tokens.
- **Parse v2** (post hoc, from 9992a raws): first line; split on "|"; strip spaces, quotes and one trailing period;
  drop items equal to "unknown" (case-insensitive); dedupe case-insensitively. A sentence reply stays one item and
  is scored as written (counted and reported, not repaired).
- **Known-bad rule v2** (post hoc): no passages → at least 90% of replies "unknown"; every other reply is reported as a
  pretraining leak (closed-book answer).
- **Retrieval k** from {1, 3, 5, 10}: chosen on the full dev split by gold-document recall@k of the retriever alone
  (no LLM; minutes of CPU), smallest k within 1 point of the best recall whose prompts fit 8,192 tokens on all dev
  questions. Then one dev confirmation run per model at that k (identical settings as test; dev-test rule):
  300 dev-pick questions, 50 per cell, seeded — 1.5B ≈ 2.5 h, 7B ≈ 7 h. Dev numbers reported, nothing else tuned.
  **Replaced by §3–4 (owner 2026-11-13): 100 dev-pick questions per set.** Tool: experiments/baselines/rag_prep.py
  kpick (gold documents = context titles + documents titled by a gold answer, as 9992a; prompt fit bound: characters / 2
  <= 8,192). Scorer: experiments/baselines/rag_score.py (tested 2026-11-15 on 9992a outputs: reproduces 4 / 3 / 0 exact;
  known-bad missing rows and answers in the no-passages arm both fail; an arm against itself gives McNemar 0 / 0).

### 3. Test arms (MetaQA 9991b test set, 10,452 questions, 6 cells)

**Scale (owner 2026-11-13; sized from the exclusive timing 9994a, log 2026-11-13):** each test set gets a **fixed
seeded subset of 630 questions**, the **same subset for both models** (paired); total RAG time <= ~48 h. Neuron
results stay on the full sets (unchanged).
- MetaQA 9991b test (10,452, 6 cells: 1h_now, 1h_sets, 2h_person, 2h_person_sets, 2h_sets, 3h_sets): 630 = 105 per
  cell, each drawn by random.Random("20261115:<cell>") from that cell's test ids: **reports/rag/mqs_test630.json**
  (sha256 8a9d1bc3...); dev 100 (17 / 16 / 17 / 17 / 17 / 16) from dev-pick ids, seed 20261116:
  **reports/rag/mqs_dev100.json** (sha256 b0958f91...); reports/rag/mqs_subsets.json.
- Jobs: 9994b_rag_mq_dev (index, k, dev confirmation), 9994c_rag_mq_test (GATE 9994b; test 630 both models; scored
  against 9991b neuron rows_B on the same questions).
- Demo corpus A test (reports/neuron/biz_<cell>_test_ids.json, 5 cells): 630 = 126 per cell, seed **20261116**.
  Fictional companies (synthetic data generated for this demo). Any resemblance to real companies or people is
  unintended.
- Dev confirmation (§2) cut to **100 dev-pick questions per set** (MetaQA 17 / 17 / 17 / 17 / 16 / 16 per cell in
  seeded order, demo A 20 per cell), same seeds + 1.
- Neuron rows for comparison: 9991b's neuron rows on the same questions (same hardware, cost from its run.json:
  host CPU per question = REVERSE + forward thread CPU / questions, plus driver CPU); exact-set McNemar on the paired
  questions; Wilson 95% intervals; per cell and overall. All RAG numbers labelled "baseline (not neurons)".
- Recorded per question: wall s, client CPU s, server CPU s (llama.cpp process, /proc), prompt and generated
  tokens; neurons: per-question share of run CPU (batched; stated as such).
- Exclusive CPU: RAG test jobs run alone (no fillers); llama.cpp server is our own child process on 127.0.0.1:7410.
- Resumable rows; progress line every 50 questions.

### 4. Order and cost (owner 2026-11-13: option A, sized from exclusive timings)

Exclusive timing 9994a (30 dev questions, k 5, 8 threads, no other job): 7B Q4 73.5 s mean wall per question
(median 62.5; server CPU 290.7 s), 1.5B Q8 43.9 s (median 33.6; server CPU 167.1 s); both models 117.4 s. (Beside
fillers 9992a had 81.2 / 37.7 s: sharing cost the 7B ~10%; the 1.5B's 30 s estimate was too low.)
| part | questions | hours |
|---|---|---|
| dev confirmation, both models, 2 sets | 2 x 100 | 6.5 |
| test, both models, MetaQA | 630 | 20.5 |
| test, both models, demo A | 630 | 20.5 |
| **total** | | **47.6** |
Plus index builds and the retrieval-recall choice of k (minutes). Order: demo corpus A neurons first (owner
2026-11-13); then MetaQA RAG (dev 3.3 h, test 20.5 h); demo A RAG after the demo A neuron test. Every RAG job runs
alone in the queue. Options B (1.5B on 2,000, 7B on 150) and C (MetaQA only) are not used: A keeps both models
paired on one subset per set and fits.

### 5. Checks

- **Neuron check:** RAG is not part of the system; no neuron code changes. Labelled baseline (not neurons). Pass.
- **Distribution / architecture check:** not applicable to the baseline; neuron runs unchanged. Pass.
