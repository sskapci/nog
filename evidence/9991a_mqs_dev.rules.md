# Rules for job 9991a_mqs_dev
Headline result: §3c sets dev. Job script as it ran: `reproduce/original/9991a_mqs_dev.sh`; code: commit 368ce83 (release archive `mxn-src-368ce83.tar.gz`). Rules commit: 368ce83. Verdict file: `9991a_mqs_dev.verdict.json` (the job's `# VERDICT:` file, `artifacts/neuron/mqs_dev/pick/set_dev_pick.json`). Commit hashes and log entries refer to the private development repository; the text below is copied from it unchanged.

---

# Experiment log entries (rules and amendments committed before the run)

## 2026-11-06 23:25 UTC — RULES MetaQA set answers and person heads (9991a dev, 9991b test; one attempt per rule)
Design reports/design_metaqa_sets_persons.md (360e739), approved 2026-11-06 with conditions (log "DECISIONS (user):
sets / person-heads design approved"). Build 63ae37a + 8af9fb8; planner 9990a (step 1,944, hold-out exact 99.23%, sha256 6ab77ea0b00753c4…); dry runs 9990b.
**Claim scope**: MetaQA over WikiMovies, 18,127 movie KUs, one machine, 5 hosts; questions whose template path has at
most one reverse step; movie heads naming exactly one document (1-, 2-, 3-hop) and value heads (person / tag; 1- and
2-hop); gold answers of any size, scored as sets. Not: movie heads naming several documents, two reverse steps
(untested, deferred), other datasets.
- **System**: MXN_VOCAB=metaqa; --kind; pick opt_pick.json; fmin 0.0 (9985a); reverse rule rel θ = 0.1 (9987a,
  frozen, not re-picked); planner artifacts/expert_mqs/planner2 (templates chain / rev / prev, r3 head), frozen.
  Set mode: a KU answering a last forward hop (or hop 0 of a movie-head reverse question) lists every own slot of the
  kind with p ≥ θ_f · p(its top slot); REVERSE carries all hop-0 values; resultant = union, head difference,
  deduplication (exact operations only).
- **Primary metric: exact set match** (answer set = gold set after normalization). Justification: one bit per
  question, so neurons and controls compare by exact McNemar (the standing statistics rule for paired comparisons);
  no credit for a missing or an extra value, so a control cannot win by listing everything. **Secondary** (report
  only): set F1 (macro), precision, recall, any hit (answer ∩ gold ≠ ∅); Wilson intervals on exact and hit.
- **Controls**: all fitting slots (every slot of the asked kind; reverse: X is any slot of the kind) = **primary**
  (user decision 2026-11-06); first slot reported. Oracle (gold-label KUs): ceiling, measurement only.
- **Cells** (hop count × category; experiments/dataset/metaqa_prep_sets.py): 1h now / sets / person / person_sets,
  2h now / sets / person / person_sets, 3h now / sets. Dev 16,853 (≤ 2,000 per cell, halves fit / pick); test 14,986,
  fresh (no dev head, no 9985b / 9987b test question): 2h_now and 3h_now have 0 test questions (already tested as
  single answers in 9987b) and get dev only.
- **9991a dev** (pool mqsdev, worktree pinned at this commit): stage fit: neuron arm, 5 hosts, θ_f grid {0.05, 0.1,
  0.2, 0.3, 0.5, 0.7, 1.0} → set_dev_pick --stage=fit: θ_f with the highest neuron exact on dev-fit over all cells
  (ties: higher θ_f). Stage pick at that θ_f: all fitting slots, first slot, oracle (5 hosts), neuron 1 host →
  **gate per cell on dev-pick**: neurons beat all fitting slots on exact (neuron-only > all-only, exact McNemar
  p < 0.05) with K identical, audit 0 after the drop (> 0 before), complete, 5 hosts = 1 host (answer sets identical).
  No bars. A failing cell is "DOES NOT FLY (dev-pick)" and gets no test.
- **9991b test** (GATE 9991a; pool mqstest restricted to passing cells' test ids): neuron (5 hosts), all fitting
  slots, first slot, neuron 1 host at the frozen θ_f → **tier per cell**: FLIES = neurons beat all fitting slots on
  exact (p < 0.05) with every check; PARTLY = some path type of the cell has such a win but FLIES fails; DOES NOT FLY
  otherwise. Job passes if some tested cell FLIES. Then oracle (ceiling, report only); drift dev-pick → test per cell
  (report only); cost per reverse hop (reverse_cost_* in run.json).
- **Limits stated before results**: stated ceilings on dev (exact set match, scope.json): 1-hop now 79.17 / sets
  66.22 / person 80.55 / person_sets 47.52; 2-hop now 86.97 / sets 27.68 / person 65.46 / person_sets 43.00; 3-hop now
  60.50 / sets 19.63. Genre and tags are not slots; uncued relations lose to first slot (chapter 4 §4.9); exact set
  match is strict, so absolute numbers will be low for sets. 2h_now dev-pick n = 188 (low power). Expected: neurons
  above all fitting slots where the relation is cue-worded and the slot kind is shared with other relations (names);
  below or equal where all slots of the kind are the answer (e.g. language).
- **Dry runs** (9990b PASS, artifacts/neuron/mqs_dryrun; log above): known-good (gold plans; oracle ≥ neuron; 3 hosts =
  1 host; K; audits; complete), known-bad NHOST_REV_KEEP_HEAD (head appears in sets: caught), noise floor (two runs
  byte-identical → 0), θ_f = 1.0 lists one value, arm all ⊇ neuron set size, person heads answered, the trained
  planner (chain / rev / prev plans, complete), regressions without set mode (2Wiki, MetaQA 1-hop, MetaQA 2- / 3-hop:
  rows identical to 9986b). Ceilings: stated (scope.json) and oracle arm.
- **Neuron check**: (1) knowledge = each KU's own MLP and slots, forward, set listing and reverse: pass; (2) each KU
  lists its own values and decides "do I list X" by its own forward pass on its host; θ_f, θ frozen settings: pass;
  (3) REVERSE carries words and a relation code; the resultant only unions, subtracts and deduplicates strings that
  KUs sent; no value / person / title index anywhere: pass; (4) training unchanged, local: pass; (5) = 1 (combining
  several KUs' findings into a set is not one KU's job; a person head is answered by every KU checking itself).
  **Distribution check**: one REVERSE per host per reverse hop (several hop-0 values in one message), replies merged
  like RESOLVE; person-head start task on one host chosen by hash of the question id; no central state: pass.
  **Architecture check**: no index at gateway, resultant or hosts; set operations exact and stateless: pass.
- Baseline (not neurons): none planned.


---

# `design_metaqa_sets_persons.md` as committed in 368ce83

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
