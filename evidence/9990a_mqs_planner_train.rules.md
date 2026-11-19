# Rules for job 9990a_mqs_planner_train
Headline result: §3c sets planner. Job script as it ran: `reproduce/original/9990a_mqs_planner_train.sh`; code: commit 8af9fb8 (release archive `mxn-src-8af9fb8.tar.gz`). Rules commit: 8af9fb8. Verdict file: `9990a_mqs_planner_train.verdict.json` (the job's `# VERDICT:` file, `artifacts/expert_mqs/planner2/train_ok.json`). Commit hashes and log entries refer to the private development repository; the text below is copied from it unchanged.

---

# Experiment log entries (rules and amendments committed before the run)

## 2026-11-06 16:11 UTC — DECISIONS (user): sets / person-heads design approved; set control; item 4 deferred
1. Design reports/design_metaqa_sets_persons.md (360e739) approved. Conditions: rules committed before any result;
   known-good / known-bad dry runs, noise floor and ceilings; one attempt per rule; neuron, distribution and
   architecture checks in the rules commit; claim scope stated.
2. Primary control for set answers: **all fitting slots** (every slot of the asked value kind); first slot also
   reported. The primary set metric (exact set match or set F1, with its justification) is fixed in the rules before
   any result; the other is reported as secondary.
3. Item 4 (multiple reverse hops): deferred; recorded in chapter 4 as untested because MetaQA has no such questions
   (scope.json: 0 on every split). No new dataset now.

## 2026-11-06 18:49 UTC — BUILD MetaQA set answers and person heads (design 360e739, approved); no results
- Code: nku.answer_set (set listing p >= theta_f * p(top); arm "all"), nku.lists with several values; nhost set mode
  (last forward hop and rev hop 0 list values; REVERSE carries all hop-0 values), person-head start tasks (prev: the
  host broadcasts REVERSE r1 for the head words; with r2 every yes-KU answers forward), arm "all"; ngateway template
  "prev" (no RESOLVE; one start task) and set-mode resultant (union, head difference, deduplication only);
  planner.predict knows "prev"; e2e_drive / e2e_run --sets grid; set_score.py (primary exact set match, McNemar vs all
  fitting slots and first slot; secondary F1 / precision / recall / hit), set_dev_pick.py, set_dryrun_check.py.
- Data: experiments/dataset/metaqa_prep_sets.py → artifacts/data/metaqa_sets, artifacts/expert_mqs, reports/neuron
  mqs_* ids. 10 cells; dev at most 2,000 per cell (halves fit / pick): 16,853; test at most 2,000 per cell, fresh
  (heads not in dev, no question of the 9985b / 9987b tests): 14,986 (2h_now and 3h_now: 0 fresh test questions;
  those cells are tested already). Planner targets: 273,106 fit / 30,345 hold-out (prev 119,647, rev 123,824,
  chain 59,980).
- Faults found by minimal invocation (rule-writing, fixed before any run that counts): REVERSE pre-check called
  holds() with a value list.
- Smoke (30 dev-fit questions, 220 KUs, gold plans; mechanics only, not a result): K 30 / 30, audits 0 / > 0,
  complete, 3 hosts = 1 host; all arms run; scorer runs. Planner: 4-step training with "prev" loads and decodes.
- Queued: 9990a planner training (2 h, pass hold-out exact >= 90%); 9990b dry runs (GATE 9990a). Rules after 9990b.


---

# `design_metaqa_sets_persons.md` as committed in 8af9fb8

## MetaQA set answers and person heads (design)

User priorities 2026-11-06 (item 2). Status: **design, sent to the user for approval**; build, dry runs and rules
follow only after approval. Builds on reports/design_metaqa.md (1-hop) and reports/design_reverse_lookup.md (one
reverse hop; 9987b FLIES for 2- and 3-hop, log 2026-11-06 13:52 UTC). Multiple reverse hops (priority 4) come after
this; §1 shows MetaQA has none.

### 1. What becomes answerable (experiments/dataset/metaqa_scope.py → artifacts/data/metaqa_verify/scope.json)

Template paths are found from the KB on train only (at most 20 questions per template, ≥ 90% majority, unique top):
a path is a sequence of forward steps (movie → its values for r) and reverse steps (value → movies listing it for r),
starting forward from a movie head and reverse from a value head; the head is excluded from the final set. Each
question of a split falls into the first category that fits:

| category | meaning |
|---|---|
| now | movie head naming one document, one gold answer, at most one reverse step (current scope) |
| sets | as now, several gold answers |
| person | value head (person, year, language …), one gold answer, exactly one reverse step |
| person_sets | as person, several gold answers |
| multi_reverse | two or more reverse steps (priority 4) |
| ambiguous_head | movie head naming several documents (or none) |
| no_path | template without a train path |

| split | questions | now | + sets | + person | + person_sets | **answerable after design** | ambiguous head | multi reverse | no path |
|---|---|---|---|---|---|---|---|---|---|
| 1-hop train | 96,106 | 42,634 (44.4%) | 17,346 (18.0%) | 15,211 (15.8%) | 10,974 (11.4%) | **86,165 (89.7%)** | 9,940 (10.3%) | 0 | 1 |
| 1-hop dev | 9,992 | 4,469 (44.7%) | 1,776 (17.8%) | 1,594 (16.0%) | 1,107 (11.1%) | **8,946 (89.5%)** | 1,046 (10.5%) | 0 | 0 |
| 1-hop test | 9,947 | 4,415 (44.4%) | 1,743 (17.5%) | 1,552 (15.6%) | 1,195 (12.0%) | **8,905 (89.5%)** | 1,042 (10.5%) | 0 | 0 |
| 2-hop train | 118,980 | 3,121 (2.6%) | 19,596 (16.5%) | 43,790 (36.8%) | 49,672 (41.7%) | **116,179 (97.6%)** | 2,801 (2.4%) | 0 | 0 |
| 2-hop dev | 14,872 | 376 (2.5%) | 2,504 (16.8%) | 5,542 (37.3%) | 6,067 (40.8%) | **14,489 (97.4%)** | 383 (2.6%) | 0 | 0 |
| 2-hop test | 14,872 | 387 (2.6%) | 2,391 (16.1%) | 5,521 (37.1%) | 6,194 (41.6%) | **14,493 (97.5%)** | 379 (2.5%) | 0 | 0 |
| 3-hop train | 114,196 | 17,098 (15.0%) | 84,009 (73.6%) | 0 (0.0%) | 0 (0.0%) | **101,107 (88.5%)** | 13,089 (11.5%) | 0 | 0 |
| 3-hop dev | 14,274 | 2,185 (15.3%) | 10,398 (72.8%) | 0 (0.0%) | 0 (0.0%) | **12,583 (88.2%)** | 1,691 (11.8%) | 0 | 0 |
| 3-hop test | 14,274 | 2,097 (14.7%) | 10,544 (73.9%) | 0 (0.0%) | 0 (0.0%) | **12,641 (88.6%)** | 1,633 (11.4%) | 0 | 0 |

Path shapes (dev): 1-hop now / sets f, person / person_sets r; 2-hop now / sets f r, person / person_sets r f; 3-hop
now / sets f r f. **No MetaQA template needs two reverse steps** (multi reverse 0 on every split): every 3-hop
template has a movie head, and every person-head template has one reverse step. Priority 4 therefore has no MetaQA
questions; it would need another dataset. Check against the current scope: movie-head paths identical to
metaqa_paths.py (2-hop 30 / 30, 3-hop 150 / 150); "now" on dev = 376 (2-hop) and 2,185 (3-hop) as the 9987 sets;
1-hop 4,469 vs 4,480 (relation labelling by template vs by KB path; 11 questions).

Stated ceilings (dev only; exact set match using only triples stated as slots in the movie's own document):

| hop count | now | sets | person | person_sets |
|---|---|---|---|---|
| 1-hop dev | 79.17 | 66.22 | 80.55 | 47.52 |
| 2-hop dev | 86.97 | 27.68 | 65.46 | 43.00 |
| 3-hop dev | 60.50 | 19.63 | — | — |

Exact set match is strict: a stated ceiling well below 100% for sets means several gold values are not slots of the
movies' own documents (genre, tags; values stated only in other documents).

### 2. Set answers

- **Forward hop, set mode.** The asked KU runs its own forward pass for r (value-kind restricted, as now) and lists
  **every** slot of its own that passes the yes rule already used for the reverse hop: relative, p(slot) ≥ θ_f · p of
  its own top slot (its top slot is always listed if f ≥ fmin). One finding carries the listed values and their p.
  The KU decides; nobody else reads its slots.
- **Reverse hop**: unchanged (it already returns sets); frozen rule rel θ = 0.1 (9987a).
- **Hop 1 with several values** (movie head, 2- / 3-hop): one REVERSE per host carries all hop-1 values; a KU says
  yes if it lists any of them (same rule). Still one message out and one reply per host per reverse hop.
- **Resultant: exact set operations over KU findings only**: union over the findings of a hop, difference (head
  excluded), deduplication by the normalized string. No ranking, no truncation, no scores combined. Answer = a set
  (possibly empty).
- **Scoring (stated in the rules)**: primary per question = **exact set match** (answer set = gold set after
  normalization and aliases); McNemar on it. Reported: set precision, recall and F1 per question (macro means;
  empty answer: P = R = F1 = 0 unless the gold set is empty, which MetaQA never has) and "any hit" (answer ∩ gold ≠
  ∅, the Hits@1-like number literature reports; labelled as such). Wilson intervals on exact match and any hit.
- **Controls** (same pipeline, no learning): **all fitting slots** (every slot of the kind; reverse: yes iff X is any
  slot of the kind): the natural set control, **primary**; first slot (singleton) reported. Oracle ceiling as before.
- θ_f picked on dev-fit by exact set match from {0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0} (θ_f = 1.0 = today's single
  answer); fmin stays 0.0.

### 3. Person heads (one reverse step)

- WikiMovies has documents only for movies, so persons, years and languages exist only as slots inside movie KUs.
  No person KUs are created and no index is built.
- **Hop 1 is a reverse hop**: the planner gives the head span X and r1; the gateway broadcasts REVERSE {r1, X} to every
  host; each KU decides by its own forward pass whether it lists X for r1 (same rule, θ = 0.1). 1-hop answer = the
  yes-KUs' titles ("which movies did X direct").
- **2-hop**: each yes-KU answers r2 forward about itself (set mode, §2); answer = union minus X ("who co-starred
  with X", "which genres does X write").
- No MetaQA template needs a second reverse step (§1), so priority 4 has no MetaQA questions.

### 4. Planner

Template "rev" generalised: steps with a direction each (start forward for movie heads, reverse for value heads), up
to 3 relations; template list and heads stored in the checkpoint (older planners load unchanged). Trained on MetaQA
train only (all categories in scope, any answer count), planner v2 recipe, 10% hold-out, pass rule includes hold-out
exact. The plan does not say "set" or "single": every answer in this design is a set.

### 5. Evaluation (rules before results)

- New dev / test sets per category (sets, person, person_sets; now re-scored with set metrics) and hop count; test
  heads that head a dev question are left out; up to 2,000 test per category and hop count; dev halves fit / pick.
- Dev-fit picks θ_f only. Dev-pick gate per category and hop count: neurons beat the all-fitting-slots control
  (exact set match, McNemar p < 0.05) with K, audit 0 after the drop, complete, 5 hosts = 1 host. One test each;
  tiers per category and hop count (FLIES / PARTLY per path type / DOES NOT FLY) as 9987b; oracle and drift report
  only; cost per reverse hop (KUs evaluated, messages, wall, CPU).
- Dry runs: known-good gold plans (oracle KUs give the stated ceiling), known-bad (NHOST_REV_KEEP_HEAD and a shuffled
  θ_f table), 1 vs 5 hosts, K, audit; 2Wiki, MetaQA 1-hop and 9987b regressions (single-answer mode θ_f = 1.0 gives
  identical rows).
- Movie heads naming several documents (10.5% / 2.6% / 11.8% of dev) stay out: which document is meant is not
  decidable from the question.
- Limits stated before results: genre and tags are not slots (lower-case words); uncued relations lose to first
  slot (stated limit, chapter 4 §4.9); value heads that are years or languages make huge reverse sets.

### 6. Checks

**Neuron check.** (1) Knowledge used: each KU's own parameters and slots, forward and reverse: pass. (2) Who
computes: each KU lists its own values (set mode) and decides "do I list X" by its own forward pass; θ_f and θ are
frozen settings: pass. (3) Shared components: REVERSE carries words and a relation code; the resultant only unions,
subtracts and deduplicates strings that KUs sent; no component reads slots or holds a value or person index: pass.
(4) Training: unchanged, local: pass. (5) Supplement question: combining several KUs' findings into one set is not one
KU's job; the person head is handled by every KU checking itself: item 5 = 1.

**Distribution check.** One REVERSE per host per reverse hop (several hop-1 values in one message), replies merged
like RESOLVE; forward set findings come from the KU's own host; no central state: pass.

**Architecture check.** No title, value or person index at gateway, resultant or hosts; set operations are exact and
stateless: pass.
