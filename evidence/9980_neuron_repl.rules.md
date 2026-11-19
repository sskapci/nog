# Rules for job 9980_neuron_repl
Headline result: §4b replication R = 2. Job script as it ran: `reproduce/original/9980_neuron_repl.sh`; code: commit 1a9d710 (release archive `mxn-src-1a9d710.tar.gz`). Rules commit: 3928753. Verdict file: `9980_neuron_repl.verdict.json` (the job's `# VERDICT:` file, `artifacts/neuron/repl/run/repl_check.json`). Commit hashes and log entries refer to the private development repository; the text below is copied from it unchanged.

---

# Experiment log entries (rules and amendments committed before the run)

## 2026-11-03 00:40 UTC · RULES neuron replication R = 2 (committed before 9980)
- Design and rules: reports/design_neuron_replication.md (neuron check 1–5 with 5 = 1; distribution check). Code:
  MEMBERS takes r (replicas per KU) and dead (hosts gone without a goodbye); copy: the first live holder of each KU
  sends it to every host of its new first-r set that lacks it; drop sets r. Driver: cold_drive.py --repl=2 (replicate
  after P0, takeover = MEMBERS copy / add / drop / done over the survivors naming the dead host; no disk);
  experiments/neuron/repl_check.py. Stated limit: until the takeover, tasks for KUs whose first holder died are
  re-asked (no failover routing yet).
- Job 9980: subset of the fresh pool (first 1,000 questions, their KUs, every third other KU; memory: R = 2 doubles
  the loaded KUs), 5 hosts, neuron at the dev choices; replicate; h2 then h4 killed during question cycles. Pass
  rules R1–R6 as in the design (identity with P0 for PR, both death cycles after re-asking, P1, P2, K; 0
  incomplete, not_here 0; replicas = rendezvous first-R sets, N (R − 1) copies sent, one copy per KU of the dead
  host at each takeover; nothing from disk; text 0; deaths live, confirmed by probes).
- Dry runs (tiny fresh pool, 188 KUs, 30 questions, 5 hosts, waves of 10): known-good all hold (replication sent
  188, PR 30 / 30; re-replicated 80 / 80 and 104 / 104; re-asked 10 per death; takeover 0.4 / 0.5 s). Known-bad
  NHOST_SKIP_ADD → R1, R2, R3 fail; REPL_NO_REREPL (no re-replication) → R1, R2, R3 fail; REPL_NO_DEAD (dead host
  not named, so its KUs' copies are never sent) → R1, R2, R3 fail. Regression after the MEMBERS change: rebalancing
  smoke (options pick) rows P0 / P2 identical 30 / 30 to the first run, all six rebalancing rules hold; cold-copy
  known-good rows P0 / P2 identical 30 / 30 to its first run, all six cold rules hold. Tests 91 pass.

