# Rules for job 9994e_rag_bench
Headline result: §7–8 RAG forget / edit arms (baseline, not neurons). Job script as it ran: `reproduce/original/9994e_rag_bench.sh`; code: commit 9a759b0 (release archive `mxn-src-9a759b0.tar.gz`). Rules commit: 9a759b0. Verdict file: `9994e_rag_bench.verdict.json` (the job's `# VERDICT:` file, `artifacts/baselines/rag/bench/check.json`). Commit hashes and log entries refer to the private development repository; the text below is copied from it unchanged.

---

# Experiment log entries (rules and amendments committed before the run)

## 2026-11-16 04:40Z — RAG arms of the forget / edit benchmarks: rules (§7), job 9994e queued — baseline (not neurons)

Standing authority (within the approved benchmark pack: RAG baseline gets the same measures; within the ~48 h RAG
budget: MetaQA ~16 h at k 3 + this ~7 h + demo A test later). rag_bench.py (prep: base / forget / edit pools and seeded
question samples; score), scorer tested on synthetic rows (known-good: 0 leaks, retain 30 / 30, efficacy 60 / 60;
known-bad: 60 leaks, retain changed, 0 edits taken: each caught). Fictional companies (synthetic data).


---

# `design_forget_edit.md` as committed in 9a759b0

## Forgetting and updating benchmarks on demo corpus A: rules draft

Fictional companies (synthetic data generated for this demo). Any resemblance to real companies or people is
unintended.

Status: **rules commit** (owner 2026-11-15: start the forgetting and updating benchmarks on questions / cells where the planner
works or with gold plans, labelled clearly; standing authority: within the approved plan, reports/design_benchmark_pack.md
§2–3, owner decisions D2 / D5). Committed after the smoke run 9994a2 (2026-11-15 23:35–23:58Z, PASS: 30 KUs and 30
cold copies deleted, audit 0; 5 edits retrained, each probe answered the new value in ~1 s).

### 0. Corpus, questions, plans

- Corpus a3 (reports/design_demo_rules.md §0), 3,000 company KUs, rules 83a2833 code plus FORGET / EDIT (§1–2).
- Questions: every dev- and test-split question of the 5 cells (reports/neuron/biz_*_{devfit,devpick,test}_ids.json).
  **Main arm: gold plans** (artifacts/expert_biz/gold_plans.json), labelled "gold plans" on every number: the planner's
  limit on unseen wordings (9993d) is not what these benchmarks measure. **Secondary arm: trained planner on dev-split
  questions** (train-style wordings, where it works: dev exact 72.2%), labelled "planner".
- theta_f 0.7 (9993c), 5 hosts, set mode, as 9993c/d. Neuron arm only; oracle as ceiling in dry runs.

### 1. FORGET

- Host message FORGET {ids, titles}: every host (broadcast) (a) deletes the KUs it holds among ids (parameters, slots,
  package, probability cache, moved / plan entries), (b) removes them from its title index and its slot-value cache,
  (c) deletes them from its cold store (every generation), (d) sets to None every own KU's pointer whose target is in
  ids. Reply {deleted, cold_deleted, pointers_cleared}. No central list exists: each host acts on its own state only.
- Forget sets: 1%, 5%, 10% of the 3,000 companies (30 / 150 / 300), seeded (20261121), nested (1% ⊂ 5% ⊂ 10%).
- Forget questions: questions whose gold path touches a forgotten company document (head, or a company on the path
  incl. reverse hops). Retain questions: none of their path companies forgotten.
- Measures (Wilson 95%):
  1. Leak rate (**rule 0**): a forget question whose answer set holds a value whose only source documents are forgotten.
     Values also stated in a present document are counted apart ("still sourced"), not as leaks.
  2. Audit (**rule 0 hits**): FAUDIT {ids, titles, values, sha, phash} on every host: forgotten ids held, title-index
     entries, cache keys listing them, slot labels unique to them, sentence sha1 in live strings, parameter tensor hashes
     in memory, cold store files holding them.
  3. Retain rows identical before / after (**rule 100%**, compared by question id on answer set and findings).
  4. Seconds from FORGET sent to all replies and FAUDIT clean.
- Dry-run known-bad: NHOST_FORGET_KEEP_COLD=1 leaves the cold copies → FAUDIT must report them; NHOST_FORGET_NO_SCRUB=1
  keeps other KUs' pointers → audit must report the dangling pointer targets.

### 2. EDIT

- Host message EDIT {k, rel, old, new, sentences}: the owning host (rendezvous route, as TASKS) replaces KU k's text by
  the edited sentences of its own article (old value replaced by the new one; nothing else is read), retrains KU k
  alone with the same settings as TRAIN, rebuilds its slots, re-resolves its own name slots by its own RESOLVE broadcast
  (as WIRE, for this KU only), invalidates its probability cache and slot-value cache entry, drops the text. Reply
  {retrained, seconds}.
- Edits: 500 seeded (20261122) over test- and dev-split companies, relation in {ceo, headquartered_in, industry,
  product, founded_by, founded_year}, new value of the same kind taken from another company (people for ceo /
  founded_by, cities, industries, products not already the company's). parent / subsidiary edits are excluded (they
  change pointers between company KUs; stated, a later step). founded_year: layout-only, expected weak (9985b),
  reported apart.
- Measures:
  1. Efficacy: the 1-hop question of the edited (company, relation) returns the new value (set for multi-valued
     relations: old value removed, new present).
  2. Generalization: the same fact through every wording of the relation (train wordings and the held-out one).
  3. Specificity: (a) the edited KU's other relations — reported; (b) KUs sharing the old value — rows identical
     (**rule 100%**).
  4. Multi-hop: parent → ceo / headquartered_in of the edited company's subsidiaries; reverse 1-hop on the new and old
     value (who has CEO new / old); value 2-hop through the edited fact.
  5. Seconds from EDIT sent to the first correct answer.
- Dry-run known-bad: NHOST_EDIT_NO_TRAIN=1 updates nothing → efficacy must fail.

### 3. Checks

- **Neuron check**: knowledge only in each KU's own parameters and slots; FORGET removes KUs and each host scrubs its
  own index, cache, cold store and own KUs' pointers; EDIT retrains one KU on its own edited article (training local);
  each KU decides alone; no index outside hosts. Pass.
- **Distribution check**: FORGET is a broadcast each host applies to its own state; EDIT goes to the owning host;
  no central state. Pass.
- **Architecture check**: two host messages and an audit message; no new component. Pass.
- Every new checker (leak, FAUDIT, retain identity, edit scorer) runs on a known-good and a known-bad case before its
  first real job (standing rule 2026-11-13).

### 4. Jobs (after the rules commit)

9995a forget dry run (small pool, 3 hosts: known-good, known-bad cold / scrub, noise floor), 9995b forget runs (1 / 5 /
10%), 9995c edit dry run (known-good, known-bad no-train), 9995d edit runs (500). RAG arms of these benchmarks follow
the RAG rules (630-question subsets) after the MetaQA RAG runs.

### 5. Code and tests (this commit)

- Host messages FORGET / FAUDIT / EDIT (mxn/neuron/nhost.py); tests/test_forget.py: known-good audit 0; known-bad
  cold copy kept, pointers kept, a stray parameter reference: each caught.
- experiments/dataset/bench_prep.py → artifacts/bench_biz (forget sets 30 / 150 / 300: 197 / 933 / 1,843 forget
  questions; 500 edits: ceo 85, headquartered_in 87, industry 81, product 87, founded_by 82, founded_year 78; classes:
  efficacy 500, generalization 1,000, multihop 454, specificity_own 1,040, specificity_neighbour 1,970, other 9,284).
- experiments/neuron/bench_drive.py (driver), bench_score.py (scorer), bench_dry_check.py (dry-run checker). Both
  checkers run before first use on the smoke outputs: scorer known-good passes; planted leak, changed retain row,
  changed neighbour row and an edit not retrained each fail; dry checker known-good passes; uncaught cold copy and
  uncaught no-train each fail.
- Jobs: 9994a3 dry runs (one attempt; the 300-question / 30-company / 20-edit pool), 9994a4 forget runs (GATE a3),
  9994a5 edit runs (GATE a3). RAG arms of these benchmarks: after the MetaQA RAG runs, under the RAG rules.
- Oracle ceiling: not run in these benchmarks (the oracle arm trains on gold triples, which EDIT does not carry);
  stated before results.

### 6. Amendment after 9994a3 (2026-11-16; standing authority: checker fix + dry-run rerun after a rule-writing fail)

9994a3 FAIL, rule-writing: the edit known-good bar (probe lists the new value on >= 90% of non-year edits) was set
without estimating its ceiling. 17 / 19 = 89.5%; both misses are founded_by edits whose old founder the KU did not list
before the edit either (dev founded_by forward exact 40.1%, 9993c). New bar: among non-year edits whose old value the KU
listed before the edit, >= 90% list the new value after (17 / 17 on 9994a3's outputs, post hoc, not a verdict); the raw
rate is reported. The scorer adds efficacy_known_fact (new listed and old gone, among those edits). Checker tested:
known-good (9994a3 outputs) passes; known-bad (3 known-fact probes answering the old value) fails. Rerun: 9994a3b
(attempt 2, output artifacts/bench_biz/dry2); 9994a4 / a5 re-gated on 9994a3b and re-pinned to this commit.

### 7. RAG arms (baseline, not neurons; RAG rules apply)

RAG forgets by deleting the forgotten documents' chunks and vectors (index rebuilt without them) and updates by
rewriting the edited documents and re-embedding them (index rebuilt). k from {1, 3, 5, 10} by corpus A dev recall@k
(rag_prep.py kpick, retriever alone). Both models. Seeded (20261123) question samples sized to the ~8 h left in the
RAG budget (MetaQA test estimate fell from 20.5 h to ~12.5 h at k 3): forget set 300 — 60 forget + 30 retain
questions, before / after; edits — 60 efficacy questions of non-year edits + 30 neighbour questions, before / after.
Measures as §1–2 on answer sets (experiments/baselines/rag_bench.py score; tested known-good / known-bad on synthetic
rows: leaks, retain change, unchanged edits each caught). A RAG leak can also come from the LLM's pretraining (reported
as is; the corpus is fictional, so little is expected). Job 9994e, after the MetaQA RAG test, alone in the queue.

---

# `design_rag_rules.md` as committed in 9a759b0

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
