# Rules for job 9991g_cache_test
Headline result: §4b slot cache 10,452 / 10,452 identical. Job script as it ran: `reproduce/original/9991g_cache_test.sh`; code: commit 879ee8f (release archive `mxn-src-879ee8f.tar.gz`). Rules commit: 879ee8f. Verdict file: `9991g_cache_test.verdict.json` (the job's `# VERDICT:` file, `artifacts/neuron/slotcache/test_check.json`). Commit hashes and log entries refer to the private development repository; the text below is copied from it unchanged.

---

# Experiment log entries (rules and amendments committed before the run)

## 2026-11-12 09:56 UTC — RULES slot-value cache (9991e dry, 9991f dev identity + CPU, 9991g identity on 9991b test rows)
Design reports/design_slot_cache.md (12911ce), approved by the owner 2026-11-12: build as designed, proof plan as
written, plus a unit test of label normalization; forgetting benchmark audit (later) must confirm a forgotten KU's id
and labels are absent from every host's cache. **Rule change (future runs only):** the 2026-11-06 rules' "no value /
person / title index anywhere" is replaced by the **host-cache rule**: a host may keep a cache over its own KUs only,
which proposes candidates; each KU decides with its own forward pass; rebuildable from the KUs; never spans hosts.
9991b and earlier stay under the old rule.
Build (this commit): per host label → own KU ids (keys = each KU's own labels(), the table holds() reads; one
normalization), brought up to date with the KUs held at every REVERSE (arrived: added; left: removed),
cache_invalidate(k) for future EDIT, cache_rebuild() for checks; NHOST_CACHE_CHECK=1 asserts cache = rebuild at every
change and at rebalancing add / drop; NHOST_SLOT_CACHE=0 restores the full scan; NHOST_CACHE_DROP_ONE known-bad.
Counters: reverse_kus_evaluated = candidates, reverse_kus_owned = own KUs × items (old meaning).
Checks before rules (by hand, this commit's code): unit test tests/test_slot_cache.py on every slot label of the
test pool: pass (70 s). Dry-run set, neuron, 5 hosts, cache on + check: rows_A / rows_B 30 / 30 identical to
9991a1 neuron_h5 (no cache), K 30 / 30, audit 0; candidates 135 vs 5,280 owned. Rebalancing (2Wiki dry-run pool,
3 + 1 spare, check on at every add / drop): P2 40 / 40, P4 40 / 40, every live cycle 40 / 40, spare empty after
leave, audit 0, rebal_check pass.
Rules (one attempt each; answers only, nothing scored or tuned):
- 9991e (dry): unit test; arms neuron and all × 1 and 5 hosts, cache on vs off identical (rows A and B), K, complete;
  known-bad DROP_ONE changes ≥ 1 answer; rebalancing as above with rebal_check pass.
- 9991f (GATE 9991e): 9991d's 400 dev questions, 5 hosts and 1 host, identical to 9991d's rows; CPU split reported.
- 9991g (GATE 9991f): all 10,452 9991b test questions, neuron, 5 hosts: identical to 9991b's neuron rows A and B, K,
  complete, audit 0.
Neuron check: (1) knowledge in each KU's own parameters and slots; the cache holds only labels the KU owns, no
probabilities, no answers: pass; (2) every candidate decides with its own holds / forward pass / yes rule;
non-candidates have no matching slot (would say nothing): pass; (3) cache per host over own KUs only, nothing in
gateway / resultant / driver, never sent between hosts: pass; (4) training unchanged: pass; (5) unchanged.
Distribution check: one cache per host; REVERSE still broadcast; updated by the same events that move KUs; no central
state: pass. Architecture check: host index = cache over that host's KUs, rebuildable, never spanning hosts
(share-nothing rule 2026-10-24b §6; host-cache rule 2026-11-09): pass. Baseline (not neurons): none.


---

# `design_slot_cache.md` as committed in 879ee8f

## Per-host slot-value cache for reverse lookup (design)

Owner decisions 2026-11-09 (cache allowed under the host-cache rule; build only if slot-table scans are a significant
share of reverse-hop CPU). Status: **design, sent to the owner; nothing built**. Rule change: the 2026-11-06 rules'
"no value / person / title index anywhere" is replaced, for future runs only, by the host-cache rule; 9991b and
earlier results stay under the old rule.

### 1. Why (9991d, log 2026-11-11)

On 400 dev reverse questions every host checks every own KU's slot table for every reverse item (7,250,800 KU checks
for 6,810 forward passes). Real REVERSE CPU on 1 host: 158 s, of which **115 s (73%) is the slot-table scan**,
42 s forward passes, 0.7 s list checks; on 5 hosts 171 s, scan 118 s (69%). Waiting / spinning is negligible and the
old 5-vs-1-host gap was a metering artifact, so the scan is the cost worth removing.

### 2. What the cache is

- Per host, a dict `label → set of own KU ids`, where label = `normalize(slot.label)` for every slot of every KU this
  host holds (exactly the keys of each KU's own `_labels` table, `nku.holds`). Value kinds are **not** in the key:
  the KU still applies `fits(rel, kind)` itself.
- It covers this host's own KUs only; it is never sent, merged or consulted across hosts; the gateway, resultant and
  driver hold nothing new.
- It is a cache: rebuildable at any time from the KUs held (`rebuild()` = one pass over own KUs' slots).

### 3. Reverse lookup with the cache

`reverse_answer(items)`: for each item, candidates = union of `cache.get(normalize(v))` over the item's values,
taken in ascending KU id (the order of today's full scan). For each candidate the KU runs **exactly today's code**:
the exclude check, `u.holds(rel, v)` (own slot table, own kind rule), its own forward pass (`u.probs`, per-host
`pcache` as now) and `u.lists` (its own yes rule). A KU not in the candidate set has no slot with that label, so
`holds` would return nothing and today's code would skip it: the set of KUs that decide, the order of `yes`, and every
answer are unchanged by construction. The cache proposes; the KU decides.

Counters: `reverse_kus_evaluated` then counts candidates; a new `reverse_kus_owned` keeps the old meaning (own KUs ×
items) so cost tables stay comparable.

### 4. Keeping it right (every way the KU set or a KU's slots change)

| event | cache update |
|---|---|
| TRAIN (KU built) | add the KU's labels |
| MEMBERS add (rebalancing, MOVE), COLD restore, LOCAL move in | add the arriving KUs' labels |
| MEMBERS drop, LOCAL move out | remove the leaving KUs' ids |
| DROPTEXT | none (slots and labels stay; only raw text goes) |
| EDIT (benchmark pack, future) | remove the KU's old labels, add the new ones after its retrain |
| FORGET (benchmark pack, future) | remove the KU's id from every label; empty labels deleted |

Safety net: a host-local check (`NHOST_CACHE_CHECK=1`, dry runs only) recomputes the cache from the KUs after every
membership change and fails loudly on any difference. Turning the cache off (`NHOST_SLOT_CACHE=0`) restores the full
scan.

### 5. Checks

- **Neuron check:** (1) knowledge stays in each KU's parameters and slots; the cache holds only labels the KU already
  owns, as a lookup key, no probabilities, no answers; (2) each candidate KU decides "yes" with its own forward pass
  and own rule; non-candidates provably would have said nothing (no matching slot); (3) no index spans hosts, no index
  in gateway / resultant / driver; (4) training unchanged; (5) unchanged. Pass.
- **Distribution check:** one cache per host over its own KUs; REVERSE is still broadcast to every host and each host
  answers from its own cache; updated by the same events that move KUs; no central state. Pass.
- **Architecture check:** host index = a cache over that host's KUs, rebuildable from them, never spanning hosts
  (share-nothing rule, 2026-10-24b §6; host-cache rule 2026-11-09). Pass.
- **Text audit:** labels are slot labels (audit kind `slot_label`, counted apart from sentences); the audit must stay
  0 after DROPTEXT with the cache present (dry-run check).

### 6. Proof plan (one attempt; answers only, no scoring)

1. Dry runs (9990b set): cache on vs off, neuron and all arms, 1 and 5 hosts: identical rows; known-bad
   (`NHOST_CACHE_DROP_ONE`: the cache forgets one label) must change some answer or be caught by the cache check;
   rebalancing dry run (H → H+1 → H) with `NHOST_CACHE_CHECK=1`.
2. 9991d's 400 dev questions: cache on vs 9991d rows (identical) and CPU split (expected: scan ≈ 0, total ≈ forward
   + lists + lookups).
3. **Equivalence on the 9991b test rows** (owner requirement): neuron arm, 5 hosts, θ_f 0.7, the 10,452 test
   questions, cache on → answer, sorted answer set and complete identical to 9991b's neuron rows for every question.
   Test questions are used only for this identity check; nothing is scored or tuned.
4. Memory: cache size per host reported (expected tens of MB: ~207,612 name slots plus other kinds over all hosts).

Expected cost after the cache (9991d set, 1 host): REVERSE CPU ~43 s instead of 158 s (scan removed); measured, not
assumed.
