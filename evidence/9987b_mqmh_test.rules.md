# Rules for job 9987b_mqmh_test
Headline result: §3b MetaQA 2-hop 49.34% / 3-hop 13.45% FLIES. Job script as it ran: `reproduce/original/9987b_mqmh_test.sh`; code: commit f871f38 (release archive `mxn-src-f871f38.tar.gz`). Rules commit: f871f38. Verdict file: `9987b_mqmh_test.verdict.json` (the job's `# VERDICT:` file, `artifacts/neuron/mqmh_test/test_verdict.json`). Commit hashes and log entries refer to the private development repository; the text below is copied from it unchanged.

---

# Experiment log entries (rules and amendments committed before the run)

## 2026-11-05 17:04 UTC — RULES MetaQA 2- and 3-hop, reverse lookup (9987a dev, 9987b test; one attempt)
Design reports/design_reverse_lookup.md (2d31377; gate 6a2b5d7); user decisions 2026-11-04 ("# Reverse lookup
decisions", log 19:19 UTC) and 2026-11-05 (log 11:06 UTC). **Claim scope: one reverse hop (2-hop) or one reverse hop
plus one forward hop (3-hop), movie head, single gold answer; not set answers, not person heads; genre / tags /
language at hop 3 are limits.**
- **System**: MXN_VOCAB=metaqa; 18,127 KUs (one per WikiMovies document), 5 hosts (R = 1, rendezvous), --kind, pick
  artifacts/neuron/dev/opt_pick.json (as 9985); planner artifacts/expert_mqmh/planner2/planner.pt (step 2000, sha256
  78a3420164275ff551789efff4b408616cae1bd3e4eb6b8e2118f8a2b24ed402; hold-out exact 99.89%, 9986a), frozen; fmin = 0.0
  (the 1-hop frozen value, 9985a), not re-picked.
- **Yes rule grid** (dev-fit only): rule ∈ {rel, abs} × θ ∈ {0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}.
- **9987a dev** (pool mqmhdev; ids mq2 devfit / devpick 188 / 188, mq3 1,092 / 1,093), worktree t24-mh pinned at this
  commit. Gate: runs only if 9985b's tier is FLIES or PARTLY (6a2b5d7; 9985b = PARTLY, log 12:37), checked by the job.
  Stage fit: neuron arm, 5 hosts, the whole grid → rev_dev_pick.py --stage=fit: rule and θ = best dev-fit neuron EM
  over 2- and 3-hop dev-fit together (ties: rel, then higher θ) → rev_pick.json. Stage pick, at the chosen setting:
  first slot (control), oracle (ceiling, measurement only), neuron on 1 host → rev_dev_pick.py --stage=pick.
  **Gate on dev-pick, per hop count**: neurons beat first slot (neuron-only > first-only, exact McNemar p < 0.05)
  with K identical, audit 0 after the drop (> 0 before), complete, 5 hosts = 1 host. No bars. A hop count whose gate
  fails is "DOES NOT FLY (dev-pick)" and gets no test. Verdict PASS = at least one hop count's gate passes.
- **9987b test** (GATE 9987a; pool mqmhtest; mq2_test_ids 379, mq3_test_ids 1,621), setting frozen from 9987a, only
  the hop counts whose dev-pick gate passed: arms neuron (5 hosts), first, neuron on 1 host → e2e_score per hop count
  (--ids, no bars) → **tier per hop count**: FLIES = neurons beat first slot (p < 0.05) with K, audits, complete and
  5 = 1 holding; PARTLY = FLIES fails but some path type (type = "<n>hop:r1>r2[>r3]") has a significant neuron win
  (neuron-only > first-only, p < 0.05); DOES NOT FLY otherwise. Verdict PASS = some tested hop count FLIES. Report
  only: oracle ceiling on test; drift dev-pick → test (EM per arm and path-type mix per hop count); per path type
  with Wilson intervals; one test each.
- **Cost (report, user decision)**: from run.json reverse_cost_* of the 5-host neuron runs: reverse batches, items,
  messages, KUs evaluated, forward passes, wall and CPU ms; per reverse hop (per item and per batch) and per question.
- **Limits stated before results**: 2-hop dev-pick n = 188 → low power (the gate may fail on power alone); slot
  ceilings on dev 2-hop 86.97%, 3-hop 60.50% (metaqa_mh_ceiling); hop 3 often asks genre / tags / language (1-hop
  limits: genre 0%, tags ~10% in every arm on the 1-hop test); hop 1 uses one forward answer although 22.6% / 24.9%
  of questions have several stated values; the 1-hop test showed uncued relations losing to first slot (9985b), and
  hop 1 / hop 3 use the same KU forward pass. **Expected**: 2-hop above first slot when hop 1 is cue-worded and the
  yes rule finds multi-valued slots; 3-hop low in every arm.
- **Dry runs**: 9986b PASS (above). Noise floor 0 (training and answering deterministic; 3 = 1 hosts identical).
  Ceiling: oracle arm.
- **Neuron check**: (1) knowledge = each KU's own MLP and slots, forward and reverse: pass; (2) each KU decides "do I
  list X for r2" with its own forward pass on its host; θ is a frozen setting: pass; (3) REVERSE carries words and a
  relation code; no component holds or reads slot labels on a KU's behalf; no value or title index: pass; (4) training
  unchanged, local: pass; (5) item 5 = 1 (the resultant's set operations: exclude head, union). **Distribution
  check**: REVERSE is one message per host per batch, replies merged like RESOLVE; no central state: pass.
  **Architecture check**: no index at gateway or hosts: pass.
- Baseline (not neurons): none planned.

