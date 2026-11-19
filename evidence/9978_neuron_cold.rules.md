# Rules for job 9978_neuron_cold
Headline result: §4b cold copies, two live kills. Job script as it ran: `reproduce/original/9978_neuron_cold.sh`; code: commit 3928753 (release archive `mxn-src-3928753.tar.gz`). Rules commit: f8894b3. Verdict file: `9978_neuron_cold.verdict.json` (the job's `# VERDICT:` file, `artifacts/neuron/cold/run/cold_check.json`). Commit hashes and log entries refer to the private development repository; the text below is copied from it unchanged.

---

# Experiment log entries (rules and amendments committed before the run)

## 2026-11-03 00:05 UTC · RULES neuron cold copies (committed before 9978)
- Design and rules: reports/design_neuron_cold.md (neuron check items 1–5 with 5 = 1; distribution check). Code:
  nhost COLD (write / commit / restore / audit) and COLDPUT: each KU's package (no text) on its successor's disk (the
  second host of its rendezvous order = its owner once the owner is gone); after a death confirmed by probes (3
  misses, 1 s apart) survivors restore their copies, then MEMBERS add / drop / done; the gateway re-asks incomplete
  questions (attempt number; at most 2); copies re-written for the survivors (re-protection). e2e_run.py --cold;
  experiments/neuron/cold_drive.py, cold_check.py.
- Rule-writing fix found in the first known-good dry run, before commit: the death cycle was 28 / 30 identical
  because a RESOLVE broadcast that missed the dead host picked another KU (degraded answer, not a lost request). Now
  a broadcast that did not hear from every host marks the chain lost and the question incomplete, so it is re-asked
  after the takeover (gateway first hop and the KU's own forward).
- Job 9978: fresh pool, 5 hosts, neuron at the dev choices; h2 then h4 killed (harness SIGKILL) during question
  cycles. Pass rules R1–R6 as in the design (identity with P0 in both death cycles after re-asking, P1, P2, K; 0
  incomplete, not_here 0; one copy per KU on its successor, restore = the dead host's KUs; placement = rendezvous
  over the survivors; text 0 in memory and in cold files; deaths live and confirmed by probes).
- Dry runs (tiny fresh pool, 188 KUs, 30 questions, 5 hosts, waves of 10): known-good all hold (restored 41 / 41 and
  60 / 60; re-asked 10 per death; cycles 30 / 30; confirm 2.2 s; copies 188 after each write; audits 0).
  Known-bad NHOST_COLD_SKIP (copies never stored) → R1–R4 fail (restored 0; not_here 17). COLD_NO_REPROTECT → R1–R4
  fail (second death restores 40 / 60). NHOST_COLD_TEXT (text in the copies) → R5 fails only. COLD_NO_RETRY → R1, R2,
  R6 fail. Regression: rebalancing smoke (options pick, waves of 10) rerun with this code: rows P0, P2, P4
  identical 30 / 30 to the run before; all six rebalancing rules hold. Tests: 91 pass.
- Noise floor: 0 (deterministic). Ceiling: 2,000 / 2,000 identity reachable (known-good 30 / 30).

