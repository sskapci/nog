# Rules for job 9994a5_bench_edit
Headline result: §8 updating. Job script as it ran: `reproduce/original/9994a5_bench_edit.sh`; code: commit 0519f4a (release archive `mxn-src-0519f4a.tar.gz`). Rules commit: 0519f4a. Verdict file: `9994a5_bench_edit.verdict.json` (the job's `# VERDICT:` file, `artifacts/bench_biz/edit/score.json`). Commit hashes and log entries refer to the private development repository; the text below is copied from it unchanged.

---

# Experiment log entries (rules and amendments committed before the run)

## 2026-11-16 00:00Z — RULES COMMIT forgetting / updating benchmarks on corpus A; smoke 9994a2 PASS

Fictional companies (synthetic data generated for this demo). Any resemblance to real companies or people is unintended.
Under the owner's standing authority (2026-11-15), within the approved benchmark pack §2–3. Smoke 9994a2 (23:35–23:58Z,
code 18ef161 / t24-bench, 5 hosts, gold plans, 300 questions): forget-30 with cold copies: 30 KUs and 30 cold copies
deleted, FAUDIT total 0, rows for every question; 5 edits: each retrained by one host, probe answered the new value
in 0.95–1.36 s (industry ×2, founded_by ×2, headquartered_in). Scorer and dry-run checker tested known-good / known-bad
before first use (design_forget_edit.md §5). Jobs 9994a3 (dry runs) queued; 9994a4 (forget), 9994a5 (edit) held on
GATE 9994a3. Neuron / distribution / architecture checks: design_forget_edit.md §3, pass.

## 2026-11-16 03:40Z — VERDICT 9994a3_bench_dry: FAIL (6 / 7) — classified rule-writing; rerun queued (standing authority)

Fictional companies (synthetic data generated for this demo). Any resemblance to real companies or people is unintended.
Job 02:26Z–03:39Z, rules 28814dc (t24-bench2). Passed: noise floor before / after; forget known-good (leaks 0 of 64
forget questions, retain 100% identical, audit 0); known-bad cold caught (30 cold copies found); known-bad scrub caught
(20 dangling pointers found); edit known-bad caught (0 retrained, probe new-value 0%). **Failed: edit_known_good** —
20 / 20 edits retrained, neighbours 100% identical, probe new-value 17 / 19 = 89.5% < 90%. Both misses: founded_by edits
on multi-founder companies whose old founder the KU never listed before the edit (KU accuracy, dev founded_by 40.1%),
not edits that failed. **Rule-writing** (bar set without a ceiling estimate). Fix (design_forget_edit.md §6): bar
conditioned on the KU having listed the old value (17 / 17 post hoc, not a verdict); checker tested known-good /
known-bad. Rerun 9994a3b queued after 9994c (RAG test, running since 03:39Z, ~12.5 h); 9994a4 / a5 re-gated on 9994a3b.


---

# `design_forget_edit.md` as committed in 0519f4a

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
