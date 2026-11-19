# Rules for job 9985a_mq_dev
Headline result: §3a MetaQA 1-hop dev gate. Job script as it ran: `reproduce/original/9985a_mq_dev.sh`; code: commit 49efe13 (release archive `mxn-src-49efe13.tar.gz`). Rules commit: 49efe13. Verdict file: `9985a_mq_dev.verdict.json` (the job's `# VERDICT:` file, `artifacts/neuron/mq_dev/pick/e2e_dev_pick.json`). Commit hashes and log entries refer to the private development repository; the text below is copied from it unchanged.

---

# Experiment log entries (rules and amendments committed before the run)

## 2026-11-04 18:53 UTC — RULES MetaQA first version (9985a dev, 9985b test; one attempt)
Design reports/design_metaqa.md (2e4a384, 6efe6a5); approved 2026-11-04 §3. **Claim scope: single lookups (MetaQA 1-hop,
movie head, one answer) end to end through planner → RESOLVE → the KU's own answer; not multi-hop, not set answers.**
- **System**: MXN_VOCAB=metaqa; 18,127 KUs (one per WikiMovies document), 5 hosts (R = 1, rendezvous), --kind, pick
  artifacts/neuron/dev/opt_pick.json (as 9977); planner artifacts/expert_mq/planner2/planner.pt (step 3999, sha256
  45d25912d0d6da35…; hold-out exact 99.89%), frozen.
- **9985a dev** (pool mqdev: 4,480 questions; ids mq_devfit / mq_devpick 2,240 / 2,240): arms neuron (fmin grid
  {0, 0.1, 0.25, 0.5}), first (control), oracle (ceiling, measurement only, grid), neuron on 1 host. On dev-fit
  (e2e_dev_pick.py): fmin = best neuron EM (ties → higher); tau as before (reported only; neuron confidence is an
  open problem); bars per relation (type = relation) on cue-worded relations (not in the vocabulary's gaps:
  directed_by, written_by, starred_actors) with n ≥ 30 = floor(0.5 × dev-fit oracle). **Gate on dev-pick**: neurons
  beat first slot (neuron-only > first-only, exact McNemar p < 0.05) with K, audit 0 after the drop (> 0 before),
  complete, 5 hosts = 1 host. Gate fails → "DOES NOT FLY (dev-pick)", reported, test untouched (9985b stays held).
- **9985b test** (pool mqtest: 2,000; mq_test_ids), dev choices frozen, arms neuron, first, neuron on 1 host →
  tier: FLIES = neurons beat first slot (p < 0.05) with K, audits and 5 = 1 holding AND every barred relation reaches
  its bar; PARTLY = some relation with a significant per-relation neuron win but FLIES fails; DOES NOT FLY otherwise.
  Then report only: oracle ceiling on test, drift dev-pick → test (e2e_drift.py). Per relation with Wilson intervals.
- **Limits stated before results**: has_genre and has_tags values are lower-case words, not slots (slot ceiling
  1.8% / 26.4% on dev); imdb relations have no slots; release_year and in_language are mostly uncued (gaps).
  **Expected**: cue-worded relations clearly above first slot; release_year near first slot; genre / tags near 0 in
  every arm.
- **Dry runs**: 9984c PASS (known-good, known-bad, identity, K, audit, 2Wiki regression). Noise floor 0 (training and
  answering deterministic, 9972 / 9976). Ceiling: oracle arm.
- **Neuron check**: (1) knowledge used to answer = each KU's own MLP and slots from its own document: pass; (2) the KU
  computes its own answer on its host: pass; (3) no shared component reads a KU's knowledge; new shared parts are a
  relation vocabulary (N4) and a planner trained on train questions only: pass; (4) training a KU touches no other KU:
  pass; (5) item 5 = 1 (the planner parses questions; RESOLVE is the KUs' own reply). **Distribution check**: same
  components; no cross-host index; rendezvous placement: pass. **Architecture check**: gateway holds no title index or
  placement table (unchanged code path): pass.
- Baseline (not neurons): none planned for MetaQA (chapter 2 extractor is 2Wiki-trained).

## 2026-11-04 19:22 UTC — 9985a / 9985b pinned to the rules commit (no change to the runs)
- The reverse-lookup build edits shared code (planner, driver) in this worktree. So that 9985a / 9985b run exactly the
  code committed with their rules, both jobs now run in a separate worktree /home/USER/projects/t24-mq1 at 49efe13
  (artifacts and .venv linked), created before 9985a started (9985a was still pending; moved to held, edited, moved
  back). 49efe13 holds the planner one-hop fix (be8fa3d) that the 9984c dry runs used.
- The build keeps the 2Wiki planner byte-identical (6,977 vkdev + vktest plans before / after the planner edits).

