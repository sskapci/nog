# Rules for job 9976_neuron_rebalance
Headline result: §4b live rebalancing 5→6→5, answers identical. Job script as it ran: `reproduce/original/9976_neuron_rebalance.sh`; code: commit d9946e6 (release archive `mxn-src-d9946e6.tar.gz`). Rules commit: 84ca908. Verdict file: `9976_neuron_rebalance.verdict.json` (the job's `# VERDICT:` file, `artifacts/neuron/rebalance/rebal_check.json`). Commit hashes and log entries refer to the private development repository; the text below is copied from it unchanged.

---

# Experiment log entries (rules and amendments committed before the run)

## 2026-11-02 18:40 UTC · RULES neuron rebalancing (committed before 9976)
- Design and rules: reports/design_neuron_rebalance.md (neuron check items 1–5 with 5 = 1; distribution check).
  Code: nhost MEMBERS (copy / add / drop / done) and MOVE (KU packages, no text); a task for a KU not held goes on to
  its owner (≤ 3); `merge_once` in the neuron path (a KU on two hosts for a moment counts once; chapter 2's `merge`
  unchanged: test_kuops still needs its tie); e2e_run.py --spare=N; experiments/neuron/rebal_drive.py, rebal_check.py.
- Job 9976: fresh pool, 5 hosts + 1 spare, neuron at the dev choices; join and leave live; pass rules R1–R6 as in
  the design (identity with P0 everywhere; no lost request; join moves N/6 within 4 binomial sd, leave moves what the
  spare held and leaves it empty; placement restored; audit 0 after each, > 0 before the deletion; both live).
- Rule-writing fix before commit: R3 was "0.8–1.2 × N/6"; on the tiny dry run (188 KUs) 38 vs 31.3 expected is +1.3
  sd binomial noise, so the band is 4 binomial sd (14,177 KUs: 2,363 ± 177).
- Dry runs (tiny fresh pool, 188 KUs, 30 questions): known-good all hold (moved 38 / 38, 6 live cycles 30 / 30,
  forward_moved 5, not_here 0, audits 0). Known-bad NHOST_SKIP_ADD → R1, R2, R4 fail (not_here 32; cycles 16 / 30);
  NHOST_PERTURB_MOVE → R1 fails (21 / 30); NHOST_KEEP_TEXT_COPY → R5 fails. Noise floor: identity exact (K, P2 = P0).
  Regression: the 9972 dry run (3 hosts, 40 questions) rerun with this code: rows identical 40 / 40 including hosts.
  Tests: 90 pass (new: merge_once).

