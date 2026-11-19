# Rules for job 9993b2_biz_dryrun
Headline result: §5 corpus A dry runs. Job script as it ran: `reproduce/original/9993b2_biz_dryrun.sh`; code: commit 099e99d (release archive `mxn-src-099e99d.tar.gz`). Rules commit: 099e99d. Verdict file: `9993b2_biz_dryrun.verdict.json` (the job's `# VERDICT:` file, `artifacts/neuron/biz_dryrun2/biz_dryrun_check.json`). Commit hashes and log entries refer to the private development repository; the text below is copied from it unchanged.

---

# Experiment log entries (rules and amendments committed before the run)

## 2026-11-13 12:51Z — DECISION (owner) + RULES COMMIT demo corpus A; test suite clean; jobs queued

Fictional companies (synthetic data generated for this demo). Any resemblance to real companies or people is unintended.
Owner 2026-11-13: (1) gateway change approved (set-mode chain with r2 answers its second hop), rules §2 amended
before any corpus-A result, MetaQA regression (10 x 30 identical) recorded in §2; (2) known-bad badrel approved;
(3) ceiling >= 95% on the oracle arm; (4) RAG: option with seeded subsets <= 2,000 per test set and total <= ~48 h,
sized after the exclusive 7B timing on 30 dev questions (approved; job 9994a, 1.5B timed in the same job, stated);
test suite must have no unexplained failures.
- test_resultant_templates fixed (test data only): since 17ba29c the resultant sorts findings by KU id k, which every
  host finding carries; the test's hand-made findings lacked k. Full suite: **93 passed, 0 failed**.
- This commit is the rules commit (reports/design_demo_rules.md; code 6c232ee + ac4b64f + test fix). Jobs pin to it:
  worktree t24-biz. 9993a_biz_planner queued; 9993b_biz_dryrun held (GATE 9993a); 9994a_rag_timing queued after.
Neuron check (rules commit): knowledge only in each company KU (own document); each KU answers with its own forward
pass; reverse by each KU's own slot table (no index); resultant applies exact operations to findings; planner sees
questions only. Pass. Distribution / architecture: unchanged. Pass.

## 2026-11-13 17:28Z — DECISION (owner): 9993b rerun (attempt 2); MetaQA checker fixed; standing checker rule

Fictional companies (synthetic data generated for this demo). Any resemblance to real companies or people is unintended.
Owner 2026-11-13: option (a). **9993b recorded as: FAIL, rule-writing (checker compared file bytes; rows match by id on
every field).** Rules commit with the fixed checker; new pinned worktree t24-biz2; rerun as 9993b2 into
artifacts/neuron/biz_dryrun2 (the old directory's finished runs would be skipped by the job's guards).
- set_dryrun_check.py (MetaQA) given the same fix: noise floor and regressions compare rows by question id. Tests on a
  copy of the 9990b outputs: known-good real outputs pass; known-good with gold_h3b lines shuffled pass; known-bad one
  answer changed in gold_h3b → noise floor fails; known-bad one row changed in reg_mqmh_h3 → that regression fails.
- biz_dryrun_check.py fix (95d746b): known-good pass, known-bad fail (entry 17:0xZ).
- **Standing rule (owner):** every new checker runs on a known-good and a known-bad case before its first real job
  (memory rule-hygiene updated).


---

# `design_demo_rules.md` as committed in 099e99d

## Demo corpus A: rules draft (KU training, planner, dry runs, dev, test)

Fictional companies (synthetic data generated for this demo). Any resemblance to real companies or people is
unintended.

Status: **approved by the owner 2026-11-13 (on 79a4444) with the standard conditions in §10; this file as committed is
the rules commit; nothing is queued before it**. Owner decisions D3 (2026-11-09) and 2026-11-12
(name check v2, then this draft). Mirrors the MetaQA sets / persons pipeline (reports/design_metaqa_sets_persons.md;
jobs 9990a planner, 9990b dry runs, 9991a dev, 9991b test). Results on A show the mechanics, not real-world accuracy:
template text is easier than real writing.

### 0. Corpus fixed before anything runs

- Corpus **a3** (artifacts/demo/a3), name check v2 done by the checked-pool method (owner 2026-11-13; log 2026-11-13 03:20Z):
  demo_gen.py (sets 5a957d5) --make-pool, factor 3, POOL_SEED 20261122: companies 9,000, people 4,500, cities 600,
  products 27,000; demo_namecheck2.py (0b2f1a6) once over the pool (case-insensitive exact, edit distance <= 1, brand
  word of a real company / brand; English + mul labels and aliases, CC0): 2,823 hits; drop list 2,965 (with the v1
  list). The corpus draws every name in pool order from the passed names; gen.json name_pool: 0 names unchecked or
  dropped. Fixed in the rules commit:
  - pool.json sha256 `96794581206eb0ba41cfb2142a3b5f1c9e560b3c1781c253c79aacbd4c92a586`
  - hits.json (drop list) sha256 `83d2ac5be95204db78f9043e6f2c22d36de10ba1814468de7bc5c046c1f71200`
  - a3 pool.pkl sha256 `72bbed35ec857b08dc0459c5408e14bbb757223c136c5d2db88e0dc2d609bac9`
  Any later change to the corpus is a new rules commit. a1 (v1 exact check only) is superseded.
- a3 size: 3,000 company documents (one KU each), 1,500 people, 200 cities, 5,979 products, 30 industries,
  32,265 questions (train / dev / test by company 60 / 20 / 20).
- Stated ceiling on a3 (1-hop forward facts whose gold value occurs in the head company's own document text):
  100% for 7 relations (founded_by 5,345, ceo 3,000, headquartered_in 3,000, product 5,979, industry 3,000,
  founded_year 3,000, parent 899); subsidiary 812 / 899 = 90.32% (seeded 10% of links stated only in the child's
  document, as designed); all 25,035 / 25,122 = 99.65%. (a1 had subsidiary 91.05%; the seeded draw differs.)

### 1. Data adapter (new: experiments/dataset/demo_prep.py)

Writes the same files the MetaQA pipeline reads, from demo_gen.py output only:
- `artifacts/expert_biz/pool_biz.pkl`: titles, sentences, owner (one KU per company) + questions {id, question, type,
  gold}. One pool for dev and test (the documents are the same; the KUs train only on their own document).
- `artifacts/data/biz/dev.json`: dev and test records (`_id`, question, `answers` = gold set, `type`, `cat`,
  `evidences` = KB triples on the path, oracle arm only).
- Planner targets `artifacts/expert_biz/biz_fit.jsonl` / `biz_mdev.jsonl`: every train-split question, seeded 10%
  hold-out; `gold_plans.json` for dry runs only.
- Plans: company 1-hop `f:r` → chain r1; company 2-hop `f:a:f:b` → chain r1 = a, r2 = b (pointer hop to another
  company's KU); value 1-hop `r:r` → prev r1; value 2-hop `r:a:f:b` → prev r1 = a, r2 = b. Spans: the head.
  **Exception, subsidiary → product** (owner 2026-11-13: use the existing multi-finding set machinery first): the
  first hop returns several companies, and set mode follows only one pointer on a non-last forward hop (nhost.step),
  so `f:subsidiary:f:product` is planned as **prev r1 = parent, r2 = product** with the head company's words: the
  head's host broadcasts REVERSE (parent, head); every company KU whose own parent slot holds the head says yes and
  answers product forward; the answer is the union (MetaQA person-head machinery, design_metaqa_sets_persons.md §3;
  no new code). Subsidiaries of X = companies whose parent is X (same KB facts); it also reaches the 10% of links
  stated only in the child's document. Gold set unchanged.
- Symlinks to the reused, corpus-free files: `enc`, `tok16k.model`, `country_table.json`, `step7_dev.json`.
- Ids: `reports/neuron/biz_<cell>_{devfit,devpick,test}_ids.json` and combined files.

Cells (fixed now): `1h_company` (forward, all relations except founded_year), `1h_year` (founded_year, a date:
layout type, expected weak, reported apart), `1h_value` (reverse: founded_by, ceo, headquartered_in, product,
industry), `2h_company` (parent → ceo / headquartered_in; subsidiary → product), `2h_value` (founder's companies →
headquarters; city's companies → chief executives). Dev: dev-split questions, at most 2,000 per cell in seeded order,
halves dev-fit / dev-pick by seed. Test: test-split questions, at most 2,000 per cell in seeded order, **held-out
wording of each relation only** (paraphrase check; the planner never saw it). Expected: some drop from dev to test
because of the wording; reported, not tuned away.

### 2. Code changes needed (each gets a minimal invocation before queueing)

- `e2e_run.py`: allow the oracle arm on pool `biz` (today only mdev / vk* / mq* or `--posthoc`).
- `set_dev_pick.py` / scoring: id-file prefix as an option (`--ids=biz`, default `mqs`, so MetaQA is unchanged).
- **Resultant, set mode (found while building; approved by the owner 2026-11-13):** a chain answered with hop 0's values only;
  parent -> ceo / headquartered_in (chain r1 parent, r2 X) needs hop 1's. Change: a chain with r2 answers with its
  last hop's values (mxn/neuron/ngateway.py `_close`). MetaQA set chains never have r2, so MetaQA rows cannot change.
  Unit test tests/test_neuron_e2e.py::test_set_mode_chain_answer_is_last_hop (passes with, fails without).
  **MetaQA regression** (tools/regress_mqs_sets.sh, 9990b settings, MXN_VOCAB=metaqa, code efcdad7): 10 row files x
  30 questions, 0 answers differ (artifacts/neuron/biz_regress_mqs/regress.json); it runs again first in 9993b.
- Test suite: no unexplained failures (test_resultant_templates fixed: its findings lacked k, which the resultant
  sorts by since 55b6d0d; test data only).
- Nothing else in nhost / nku / gateway changes. Vocabulary via `MXN_VOCAB=biz` (mxn/neuron/vocab/biz.json).
- Regression: the 9990b MetaQA dry-run rows with `MXN_VOCAB=metaqa` stay identical after the change.

### 3. Planner (job 9993a, as 9990a)

`MXN_VOCAB=biz planner_run.py artifacts/expert_biz artifacts/expert/enc artifacts/expert_biz/planner2 --stage=train
--hours=2 --plans=biz`. Trains only on train-split questions (train wordings). PASS: best mdev exact >= 90 (as 9990a).
Hold-out exact on the 10% split reported.

### 4. KU training

Unchanged: each host trains its own KUs at run start (`TRAIN`), settings from `artifacts/neuron/dev/opt_pick.json`
(hidden 128, epochs 600, lr 0.03, rich signal); each KU trains only on its own company document. 3,000 KUs at
~0.9 CPU-s each: ~10 min on 5 hosts. Name slots wire pointers between company KUs (parent / subsidiary) by each
KU's own RESOLVE broadcast, as now.

### 5. Dry runs (job 9993b, gate: planner 9993a PASS; one attempt)

Small pool (artifacts/neuron/biz_smoke, demo_prep.py): 60 questions = 12 dev-fit per cell; their evidence companies
(every company holding a reverse value, else the known-good set is incomplete) plus seeded companies up to 300,
closed under parent / subsidiary links: 539 companies (deviation from 300, stated before any run).
- Known-good: gold plans, 3 hosts, run twice → identical rows (noise floor); ceiling check (**owner 2026-11-13: the
  oracle arm**): the **oracle arm** with gold plans answers the exact gold set on >= 95% of forward questions
  whose gold is in the head's own document (as the MetaQA known-good: oracle is the ceiling); the neuron arm is not
  held to 95% in a dry run (minimal invocation: neuron 2 / 5, oracle 5 / 5).
- Hosts: 3 hosts = 1 host (identical rows).
- Arms: all fitting slots, first, oracle (ceiling); oracle >= neuron; all ⊇ neuron.
- Known-bad (**approved 2026-11-13: badrel replaces keephead**): `NHOST_REV_KEEP_HEAD=1` acts only on rev tasks with a head KU;
  corpus A has no rev plans and prev tasks have no head KU, so it cannot be caught here. Replaced by **badrel**: gold
  plans with every prev r1 swapped to a wrong relation (experiments/neuron/biz_badrel_plans.py: parent→subsidiary,
  founded_by↔ceo, product↔industry, headquartered_in→industry); some prev answer must change and prev exact must drop.
- Trained planner (no gold plans): every template (chain, chain+r2, prev, prev+r2) occurs.
- Audits: text audit 0 after DROPTEXT and > 0 before; K identical; complete.
- Slot kinds: product and industry values are found as name slots (capitalised); founded_year as year slots.
- Set-valued first hop (subsidiary → product) by prev (§1): known-good with gold plans answers the union on the
  dry-run questions whose children state their parent (>= 95% exact set); known-bad: with NHOST_REV_KEEP_HEAD=1 or
  a wrong r1 (subsidiary instead of parent) the answers change. Only if the prev path fails these is the path split
  from `2h_company` and reported as "not supported" — decided by the dry run, before any dev or test result.
- Regression: MetaQA dry-run rows identical (section 2).

Checker: experiments/neuron/biz_dryrun_check.py (every check listed in its docstring); rows are compared by question
id on every field, never by file bytes (rows stream in close order). **Attempt 1 (9993b): FAIL, rule-writing (checker
compared file bytes; rows match by id on every field).** Attempt 2 (owner 2026-11-13): job 9993b2 at the rules commit
with the fixed checker, output artifacts/neuron/biz_dryrun2. Checker fix run on known-good (real runs; shuffled lines)
and known-bad (one answer changed) before the job; standing rule: every new checker the same before its first real job. MetaQA regression (tools/regress_mqs_sets.sh) runs first in the same job.

### 6. Dev (job 999xc, gate: dry runs PASS)

As 9991a: neuron on 5 hosts with θ_f grid {0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}, θ_f picked on dev-fit; at that θ_f
arms all (fitting slots), first, oracle; neuron on 1 host. Cells to test: those where neurons beat all fitting slots
on dev-pick (exact-set McNemar p < 0.05). CPU and wall per question recorded (benchmark 3).

### 7. Test (job 999xd, gate: dev PASS; one attempt)

As 9991b: neuron 5 hosts, all, first, neuron 1 host, oracle (report only), at the dev θ_f, on the test ids. A cell
FLIES when neurons beat all fitting slots on exact-set McNemar p < 0.05, K identical, audit 0 after the text drop
and > 0 before, complete, 5 hosts = 1 host. Wilson 95% intervals. `1h_year` reported apart. PASS = some tested cell
FLIES. Results labelled "fictional companies (synthetic)".

### 8. Checks

- **Neuron check:** (1) knowledge only in each company KU's parameters and slots, trained on its own document;
  (2) each KU answers with its own forward pass and own rule; (3) no index of companies, people or values outside
  the hosts (host slot cache only, host-cache rule); (4) training per KU unchanged; (5) planner sees questions only.
  Pass.
- **Distribution check:** KUs partitioned over hosts as for MetaQA; REVERSE broadcast; per-host cache; no central
  state. Pass.
- **Architecture check:** no new component; adapter and vocabulary only. Pass.

### 9. Order and cost (owner confirmed 2026-11-13: demo corpus A first, then RAG)

Queue: after approval (9991g is done; only fillers run now). CPU: planner 2 h; dry runs ~1 h; dev ~2 h; test ~1 h (3,000 KUs, far fewer than MetaQA's 18,127).
The RAG arms on the full MetaQA test set with 1.5B are ~120 h of exclusive CPU (9992a timing, 41 s per question);
proposed order: demo corpus A (about 6 h) before the long RAG runs, so the benchmarks on A (forgetting, updating)
can start sooner. RAG arms on A follow the same scale rule (7B seeded subset ~12 h, 1.5B full).

### 10. Approval and standing conditions (owner 2026-11-13)

Approved with these conditions; each is checked before the job that needs it is queued:
1. **Neuron check** (§8) repeated in every rules / design commit that touches this pipeline; non-neuron results
   labelled "baseline (not neurons)".
2. **Dry runs** with known-good and known-bad cases (§5) pass before dev; noise floor first for any randomness check.
3. **Held-out wordings in test only**: test questions use only ask[r][-1]; the planner trains on train-split
   questions with the other wordings; a check in demo_prep.py fails if a test wording occurs in planner targets.
4. **Fictional-data notice on every page**: the line "Fictional companies (synthetic data generated for this demo).
   Any resemblance to real companies or people is unintended." heads every file, report, log entry, table and page
   that shows corpus A content or results (generator outputs, adapter outputs, scores, artifacts).
5. **Rules committed before any run**: this file and the code changes of §2 are committed, and their commit is the
   one the jobs pin to (detached worktree), before anything is queued.
