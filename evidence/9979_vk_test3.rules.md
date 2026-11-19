# Rules for job 9979_vk_test3
Headline result: §2a 2Wiki test3 FLIES 39.70%. Job script as it ran: `reproduce/original/9979_vk_test3.sh`; code: commit 1a9d710 (release archive `mxn-src-1a9d710.tar.gz`). Rules commit: 68c00ce. Verdict file: `9979_vk_test3.verdict.json` (the job's `# VERDICT:` file, `artifacts/neuron/vk_test/e2e_test_score.json`). Commit hashes and log entries refer to the private development repository; the text below is copied from it unchanged.

---

# Experiment log entries (rules and amendments committed before the run)

## 2026-11-03 00:50 UTC · RULES value-kind attempt (committed before 9977 dev and 9979 test3)
Design reports/design_value_kind.md (749f721; sets e4b8c87). User decision log 2026-11-02 21:00.
- **System.** Neuron KUs as 9974 (options pick: hidden 128, 600 epochs, lr 0.03, rich signal) plus the value-kind
  restriction (--kind): the relation code carries its expected value kind (relation vocabulary, N4); each KU answers
  only from its own slots of that kind (others masked before its softmax; none of that kind → don't know). 5 hosts,
  R = 1, rendezvous; gateway, resultant, frozen tau as before.
- **Arms.** neuron (system), first (control), oracle (gold labels; ceiling, measurement only), neuron on 1 host; all
  with --kind. Baseline (not neurons): the frozen chapter 2 system on test3 only if it runs with pool plumbing only
  (separate job; labeled; no parity rule).
- **Dev (9977, vkdev: 28,645 KUs, 4,977 questions; fit reports/neuron/vk_devfit_ids.json 2,488, pick
  vk_devpick_ids.json 2,489).** Choices on dev-fit: fmin from {0, 0.1, 0.25, 0.5} = best neuron EM (ties → higher);
  tau = smallest conf with dev-fit confident accuracy ≥ 90%; bars per type = floor(0.5 × dev-fit oracle ceiling) on
  cue-word questions, types with n ≥ 30. Gate on dev-pick (one configuration): neurons beat first-slot (neuron-only >
  first-only, exact McNemar p < 0.05) and K, audit 0 (> 0 before), complete, 5 hosts = 1 host. Gate fails → tier
  "DOES NOT FLY (dev-pick)" reported at once; test3 untouched.
- **Test3 (9979, vktest: 14,114 KUs, 2,000 questions), one attempt, dev choices frozen.** FLIES = neurons beat
  first-slot (McNemar p < 0.05) with K, audit 0 and 5 = 1, and every barred type reaches its bar; PARTLY = a type
  with a significant per-type neuron win but FLIES fails; DOES NOT FLY = otherwise. After the tier is written (report
  only): oracle on test3 (ceiling per type), drift table dev-pick → test3 for every arm (e2e_drift.py).
- **Always reported.** EM with Wilson 95% per arm, per type, cue / gap, per relation; confident wrong, abstention by
  stage; per-type McNemar; hop-2 "don't know"; routing; cost.
- **Expected (stated before results).** Comparison and bridge-comparison rise toward first-slot; compositional and
  inference change little; gap relations stay weak (no layout rules).
- **Dry runs.** 9977d PASS (above). Noise floor 0 (deterministic). Ceiling: oracle on dev-fit (bars).
- **Neuron check.** 1 knowledge in the KU's MLP and slots: pass. 2 the KU's own forward pass and masking: pass. 3
  nothing shared reads KU knowledge (kind = relation vocabulary; slot kinds = the KU's own slot table; text deleted,
  audit 0): pass. 4 training local, unchanged: pass. 5 the restriction is applied by the KU itself; the gateway sends
  only the relation code: pass.
- **Distribution check.** No new state or component; no central KU state; no index spanning hosts: pass.

