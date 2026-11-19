# Rules for job 9981_neuron_locality
Headline result: §4b locality placement. Job script as it ran: `reproduce/original/9981_neuron_locality.sh`; code: commit 3ee4adf (release archive `mxn-src-3ee4adf.tar.gz`). Rules commit: 3ee4adf. Verdict file: `9981_neuron_locality.verdict.json` (the job's `# VERDICT:` file, `artifacts/neuron/locality/run/loc_check.json`). Commit hashes and log entries refer to the private development repository; the text below is copied from it unchanged.

---

# Experiment log entries (rules and amendments committed before the run)

## 2026-11-03 01:30 UTC · RULES neuron locality placement (committed before 9981; code on branch loc, 3ee4adf)
- Design and rules: reports/design_neuron_locality.md on branch loc (neuron check 1–5 with 5 = 1: each KU chooses its
  own anchor, the target of its strongest pointer; distribution check: no placement table, hosts report counts only).
  Code kept on branch loc (worktree t24-loc) until the value-kind test 9979 is done, so 9977 / 9979 run unchanged
  code. Measured motivation: 9974 at rendezvous placement, 770 same-host vs 3,122 other-host hop forwards (19.8%).
- Protocol: LOCAL plan (anchor, wanted host = rendezvous of the anchor), quotas from host counts (cap 1.25 × mean),
  LOCAL copy (KUs within quota take the anchor as key, sent whole), MEMBERS add / drop / done, LOCAL keys (KEYS
  broadcast: pointers store their targets' keys; tasks route by key); a host notes where it sent each KU (forwarding
  during the move). Rebalancing, cold copies and replication place by the key (default the KU id).
- Job 9981: fresh pool, 5 hosts, neuron at the dev choices; P0, locality move live, P_L. Pass rules R1–R6 (identity
  with P0 in every cycle and P_L, K; 0 incomplete, not_here 0, every pointer target found; every KU once, no host
  above cap; forward_moved 0 in P_L; text 0; live). Reported, no bar: same-host share of hop forwards and cross-host
  forwards P0 vs P_L, moved KUs, loads, move time.
- Dry runs (tiny fresh pool, 188 KUs, 30 questions, 5 hosts, waves of 10): known-good all hold (moved 13; loads 35 /
  32 / 47 / 28 / 46, cap 47; same-host share 0.111 → 0.593; other-host forwards 24 → 11). Known-bad
  NHOST_LOCAL_NOKEY (sent, keys not set) → R4 fails; NHOST_LOCAL_NOTKEY (pointers not told the keys) → R4 fails;
  NHOST_SKIP_ADD → R1–R4 fail (first version crashed in the keys phase on missing targets; fixed to report them,
  rule-writing fix before commit). Regression with the loc code: rebalancing, cold-copy and replication smokes all
  pass their rules, rows P0 / P2 identical 30 / 30 to their earlier runs. Tests 91 pass.

