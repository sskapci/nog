# NOG results package (v1, publication freeze 2026-11-17)

Every number below carries its interval (Wilson 95% unless stated), its size n, the job that produced it, the commit
that records it and the experiment-log entry (heading date) that reports it. Numbers are
copied from the log, not recomputed. Baselines are labelled "baseline (not neurons)". Corpus A numbers are on
fictional companies (synthetic data generated for this demo; any resemblance to real companies or people is
unintended).

*Commit hashes, experiment-log entries and file:line citations (README.md, chapter*.md, design_*.md, experiment_log.md) refer to the private development repository; this public repository is a curated release snapshot. For every headline job, `evidence/` holds its rules as committed before the run and its verdict file, named by job id; the code exactly as it ran (its commit in `reproduce/jobs.tsv`) is attached to this snapshot's GitHub release as a source archive, with SHA-256 checksums in `reproduce/SHA256SUMS`; `reproduce/run_job.sh` downloads the archive, checks it, unpacks it and re-runs the original job script ([reproduce.md](reproduce.md)).*

## 0. Headline tiers at a glance (neurons; fresh tests, one attempt each)

| test | tier | neurons exact | control | n | job |
|---|---|---|---|---|---|
| 2Wiki test3 (value-kind) | **FLIES** | 39.70% [37.6, 41.9] | first slot 28.80% | 2,000 | 9979 |
| 2Wiki first e2e test | PARTLY | 22.85% [21.1, 24.7] | first slot 26.70% | 2,000 | 9974 |
| MetaQA 1-hop | PARTLY | 55.40% [53.2, 57.6] | first slot 40.25% | 2,000 | 9985b |
| MetaQA 2-hop (reverse) | **FLIES** | 49.34% [44.3, 54.4] | first slot 5.80% | 379 | 9987b |
| MetaQA 3-hop (reverse) | **FLIES** | 13.45% [11.9, 15.2] | first slot 7.96% | 1,621 | 9987b |
| MetaQA sets / person heads (6 cells) | **FLIES** (all 6) | 16.19% [15.5, 16.9] overall | all fitting slots 4.28% | 10,452 | 9991b |
| corpus A 1h_company | **FLIES** | 24.45% [22.6, 26.4] | all 0.0%, first 6.65% | 2,000 | 9993d |
| corpus A 2h_company / 2h_value | DOES NOT FLY | 0.0% | 0.0% | 409 / 320 | 9993d |
| MetaQA confidence table | FAIL (count rule) | 46 confident < 50 | — | 3,521 | 9989a |
| Forgetting (corpus A, 30 / 150 / 300) | PASS | leaks 0, audit 0, retain 100% | — | 12,938 q | 9994a4b |
| Updating (corpus A, 500 edits) | PASS | known facts 99.15% switched; neighbours 100% | — | 14,248 q | 9994a5 |
| Distribution (rebalance, cold, R = 2, locality, cache) | PASS each | identical answers | — | up to 10,452 | 9976–9981, 9991g |

## 1. Claim, principle, tiers (exact text)

- **Claim** (README.md:19–21; reports/design_ku_neurons.md:7): "KUs are neurons. Each KU holds its knowledge in its own
  trained parameters, learned only from its own article, and computes its own answers. … The system is distributed: KUs
  (neurons) are spread across many hosts, which divide both the KUs and the work, Apache Spark style."
- **Principle** (README.md:3–4): "KUs are the center of the system. Everything else is a supplement that exists only to
  let KUs do their own job." Production answers only through KU neurons; the chapter 2 system (stored text + shared
  extractor) is frozen as a baseline (README.md:11).
- **Tiers** (experiment_log.md:5507–5509; chapter4.md:101–103): FLIES = neurons beat the first-slot control (McNemar
  p < 0.05) on a fresh test with every choice frozen on a disjoint dev, with K (text deleted, answers identical), audit 0
  and 5 hosts = 1 host holding, and every barred cue-word type at its bar. PARTLY = some type has a significant per-type
  neuron win but FLIES fails. DOES NOT FLY = otherwise; a failed dev gate gives "DOES NOT FLY (dev-pick)". MetaQA
  variants: per relation (log 6353–6354), per hop count (6539–6541), per cell with "all fitting slots" as the set
  control (6816–6818). FLIES does not mean competitive accuracy, every type working, working confidence or generality
  (README.md:29–33).

## 2. 2WikiMultiHopQA

### 2a. Neurons
| result | number | interval | n | job | commit | log entry |
|---|---|---|---|---|---|---|
| feasibility: gold (KU, relation) lookups, vs random slot 25.2% (p 6.5e-5); text deleted 448 / 448 identical | 37.7% | [33.4, 42.3] | 448 | 9967 | rules 3c11c21 | 2026-11-01 · 9967 neuron feasibility TEST: PASS |
| first end-to-end test: tier PARTLY (first slot 26.70%, better; 235 / 312, p 0.0011) | 22.85% | [21.1, 24.7] | 2,000 (14,177 KUs) | 9974 | log 3ebb874 | 2026-11-02 17:49 — VERDICT 9974 |
| value-kind dev gate PASS (first 28.69%; 362 / 107, p 2.0e-33) | 38.93% | [37.0, 40.9] | 2,489 | 9977 | rules 68c00ce; log 6642a94 | 2026-11-03 15:24 — VERDICT 9977 |
| **test3: FLIES** (first 28.80% [26.9, 30.8]; 332 / 114, p 9.1e-26; 5 hosts = 1 host, 794 / 794 correct, identical per type) | **39.70%** | [37.6, 41.9] | 2,000 (14,114 KUs) | 9979 | rules 68c00ce; log 891acb4 | 2026-11-03 21:49 — VERDICT 9979 |
| test3 oracle ceiling (measurement only) | 62.35% | [60.2, 64.4] | 2,000 | 9979 | 891acb4 | same |
| test3 bridge_comparison (first 38.22; 144 / 53, p 6.6e-11) | 59.04% | [54.4, 63.6] | 437 | 9979 | 891acb4 | same |
| test3 comparison (first 79.26; 18 / 49, p 1.9e-4, **first wins**) | 72.90% | [68.8, 76.7] | 487 | 9979 | 891acb4 | same |
| test3 compositional (first 2.80; 137 / 12, p 4.9e-28; cue bar 19.78 >= 17) | 18.00% | [15.5, 20.8] | 822 | 9979 | 891acb4 | same |
| test3 inference (first 0.00; 33 / 0, p 2.3e-10; bar 12.99 >= 8) | 12.99% | [9.4, 17.7] | 254 | 9979 | 891acb4 | same |

### 2b. Text-reading system (chapter 2: stored text + shared extractor) — baseline (not neurons)
| result | number | interval | n | job | commit | log entry |
|---|---|---|---|---|---|---|
| same test3 questions (neurons vs baseline 116 / 495, p 1.2e-56: baseline better) | 58.65% | [56.5, 60.8] | 2,000 | 9982b | log 4e673a3 | 2026-11-04 10:13 — VERDICT 9982b |
| step 8 end to end (vs Qwen2.5-7B closed book 25.9% [24.0, 27.9]; 768 / 91, p 3e-134) | 59.75% | [57.6, 62.0] | 2,000 | step 8 | 1faecde; rules 48e9d4a | 2026-10-27 · step 8 end to end on test |
| our extractor vs Qwen2.5-1.5B reading the same KUs (57.40%; 91 / 78, p 0.36) | 60.0% | [55.6, 64.2] | 500 | B2 | rules 0b242bd | 2026-10-27 · B2 result |
| step 8 on paraphrases (60.33 → 19.00%) | 19.00% | — | 300 | step 8 | 5bcdd1f | 2026-10-27 · step 8 |
| HONEST: confident wrong with all supporting KUs detached | 0.8% (4 / 500) | [0.3, 2.0] | 500 | 9934 | 5d84b81 | 2026-10-29 · 9934 HONEST attempt 2 |
| UPDATABLE (baseline): FAIL, final | 85.86% | [81.5, 89.3] | — | step 9 | d3b8d6f | log line 3756 |

## 3. MetaQA (WikiMovies, 18,127 KUs, 5 hosts)

### 3a. 1-hop
| result | number | interval | n | job | commit | log entry |
|---|---|---|---|---|---|---|
| dev gate PASS (first 39.64; 569 / 206, p 4.0e-40) | 55.85% | [53.8, 57.9] | 2,240 | 9985a | rules 49efe13; log 35c4adf | 2026-11-05 03:55 — VERDICT 9985a |
| **test: tier PARTLY** (first 40.25% [38.1, 42.4]; 505 / 202, p 8.1e-31; oracle 78.25% [76.4, 80.0]) | **55.40%** | [53.2, 57.6] | 2,000 | 9985b | log 346f37b; correction 44bdfd8 | 2026-11-05 12:37 — VERDICT 9985b (+ 17:12 CORRECTION) |
| cue-worded relations (first 20.61) | 73.89% | [70.8, 76.7] | 854 | 9985b | 44bdfd8 | same |
| uncued relations (first 54.89 [52.0, 57.7]) | 41.62% | [38.8, 44.5] | 1,146 | 9985b | 44bdfd8 | same |
| directed_by / written_by / starred_actors (bar 48: missed by 4.04) | 88.25 / 56.70 / 43.96% | [85.1, 90.8] / [50.6, 62.6] / [34.2, 54.2] | 502 / 261 / 91 | 9985b | 44bdfd8 | same |
| release_year (layout; first 95.53) / in_language / has_genre / has_tags | 75.77 / 23.00 / 0.00 / 10.16% | [72.1, 79.1] / [15.8, 32.2] / [0, 1.1] / [6.0, 16.6] | 582 / 100 / 336 / 128 | 9985b | 44bdfd8 | same |

### 3b. 2- and 3-hop (one reverse hop; yes rule rel θ 0.1)
| result | number | interval | n | job | commit | log entry |
|---|---|---|---|---|---|---|
| **2-hop test: FLIES** (first 5.80 [3.9, 8.6]; 171 / 6, p 4.2e-43; oracle 74.67) | **49.34%** | [44.3, 54.4] | 379 | 9987b | rules f871f38; log bbc9e65 | 2026-11-06 13:52 — VERDICT 9987b |
| **3-hop test: FLIES** (first 7.96 [6.7, 9.4]; 190 / 101, p 2.0e-7; oracle 32.76) | **13.45%** | [11.9, 15.2] | 1,621 | 9987b | bbc9e65 | same |
| reverse cost per question | CPU 1.40 / 1.02 s; wall 18.0 / 13.4 s per 50-question batch | — | 2,000 | 9987b | bbc9e65 | same |

### 3c. Sets and person heads (exact set; θ_f 0.7; control = all fitting slots)
| result | number | interval | n | job | commit | log entry |
|---|---|---|---|---|---|---|
| dev: 8 of 10 cells pass the gate; 1h_person / 1h_person_sets fail (DOES NOT FLY, dev-pick) | 21.44% vs all 12.58% | — | 8,427 | 9991a | log 749824d | 2026-11-08 15:05 — VERDICT 9991a |
| **test: all 6 tested cells FLY**; overall (all 4.28 [3.9, 4.7]; 1345 / 100, p 6.9e-279) | **16.19%** | [15.5, 16.9] | 10,452 | 9991b | rules b24b1ad; log c504ff0 | 2026-11-09 20:02 — VERDICT 9991b |
| 1h_now / 1h_sets / 2h_person / 2h_person_sets / 2h_sets / 3h_sets | 52.24 / 8.04 / 23.65 / 3.35 / 17.95 / 1.70% | [49.5, 55.0] / [6.6, 9.7] / [21.8, 25.6] / [2.6, 4.2] / [16.3, 19.7] / [1.2, 2.4] | 1,271 / 1,181 / 2,000 ×4 | 9991b | c504ff0 | same |
| oracle ceiling overall (neurons reach 38% of it) | 42.64% | [41.7, 43.6] | 10,452 | 9991b | cf122e3 | 2026-11-09 22:29 |

## 4. Confidence and distribution mechanics

### 4a. Confidence
| result | number | interval | n | job | commit | log entry |
|---|---|---|---|---|---|---|
| 2Wiki table dev: PASS; precision 96.27% [91.6, 98.4] | 134 confident (5.38%) | [4.6, 6.3] | 2,489 | 9983a | log b13d7f9 | 2026-11-04 15:25 |
| 2Wiki table on test3; precision 96.47% [90.1, 98.8]; compositional / inference 0 confident | 85 (4.25%) | [3.5, 5.2] | 2,000 | 9983b | be8fa3d | 2026-11-04 18:23 |
| MetaQA table dev: **FAIL** (count rule: 46 < 50; precision 91.30% [79.7, 96.6] held) | 46 (1.31%) | [1.0, 1.7] | 3,521 | 9989a | log 13302e7 | 2026-11-06 15:20 |

### 4b. Distribution mechanics (pass = identical answers, 0 lost, text audit 0)
| result | identity | n | job | commit | log entry |
|---|---|---|---|---|---|
| live rebalancing 5 → 6 → 5 hosts; 2,335 KUs moved each way; join 23.5 s, leave 19.0 s | 2,000 / 2,000 ×4 | 2,000 | 9976 | log 90ca799 | 2026-11-02 20:41 |
| cold copies R = 1, two live kills; restored 2,929 / 3,620 KUs; takeover 5.4 / 8.0 s | 2,000 / 2,000 | 2,000 | 9978 | d765823 | 2026-11-03 01:25 |
| replication R = 2, two live kills; takeover 8.4 / 14.7 s | 1,000 / 1,000 | 1,000 | 9980 | 6642a94 | 2026-11-03 15:51 |
| locality placement: same-host forwards 19.8 → 64.2%; cross-host −55% | 2,000 / 2,000 | 2,000 | 9981 | 08b20c8 | 2026-11-03 22:44 |
| memory: host trim, mean RSS 4,130 → 2,688 MB (−35%) | 30 / 30; 1,200 / 1,200 | 1,200 | 9991c | d8489c7 | 2026-11-10 14:21 |
| slot-value cache on MetaQA test: KU checks 145.0M → 385,572; real REVERSE CPU 305.7 s (0.029 s / question) | **10,452 / 10,452** | 10,452 | 9991g | a5ef4d6 | 2026-11-12 19:29 |
| forget / edit on corpus A | see §7–8 | | | | |

## 5. Demo corpus A (fictional companies; neurons; trained planner)

Corpus a3: 3,000 company documents (one KU each), 32,265 questions; names from a checked pool (Wikidata name check v2:
exact, case-insensitive, edit distance <= 1, brand word), log 2026-11-13 03:20Z. Test questions use only held-out
wordings the planner never saw.

| result | number | interval | n | job | commit | log entry |
|---|---|---|---|---|---|---|
| planner hold-out exact (train wordings) | 100.0% | — | 1,932 | 9993a | 40f5007 | 2026-11-13 — VERDICT 9993a_biz_planner |
| dry runs | PASS 20 / 20 checks (attempt 2) | — | 60 questions | 9993b2 | 7930fed | 2026-11-13 — VERDICT 9993b2_biz_dryrun |
| dev theta_f | 0.7 | — | 2,598 dev-fit | 9993c | bd91471 | 2026-11-13 — VERDICT 9993c_biz_dev |
| dev-pick exact, neuron (all cells) | 72.19% | [70.4, 73.9] | 2,600 | 9993c | bd91471 | same |
| test 1h_company | **FLIES**: neuron 24.45% vs all 0.0%, first 6.65% | [22.6, 26.4] | 2,000 | 9993d | 1b18770 | 2026-11-14 — VERDICT 9993d_biz_test |
| test 2h_company | DOES NOT FLY: 0.0% | [0.0, 0.9] | 409 | 9993d | 1b18770 | same |
| test 2h_value | DOES NOT FLY: 0.0% | [0.0, 1.2] | 320 | 9993d | 1b18770 | same |
| cause of the test drop | planner mis-plans the held-out wordings (0 plans right for ceo, hq, product, parent, subsidiary, every 2-hop type); oracle also falls with those plans (31.2 / 0.2 / 0.0%) | — | 2,729 | 9993d | 1b18770 | same |

## 6. RAG baseline (baseline, not neurons; Qwen2.5 1.5B Q8 / 7B Q4 on CPU llama.cpp, bge-small + FAISS)

| result | number | interval | n | job | commit | log entry |
|---|---|---|---|---|---|---|
| exclusive timing, 30 dev q, k 5 | 7B 73.5 s, 1.5B 43.9 s wall / question | — | 30 | 9994a | 61789b8 | 2026-11-13 — VERDICT 9994a_rag_timing |
| MetaQA k (dev recall@k) | k = 3 (recall 68.98) | — | dev split | 9994b | 796d1f4 | 2026-11-16 — VERDICT 9994b_rag_mq_dev |
| MetaQA test exact, neurons (9991b rows) | 18.73% | [15.9, 22.0] | 630 | 9994c | ac31b6d | 2026-11-16 — VERDICT 9994c_rag_mq_test |
| MetaQA test exact, RAG 7B | 16.98% | [14.3, 20.1] | 630 | 9994c | ac31b6d | same |
| MetaQA test exact, RAG 1.5B | 11.43% | [9.2, 14.2] | 630 | 9994c | ac31b6d | same |
| McNemar neurons vs 7B | 51 vs 40, p = 0.29 (no difference) | — | 630 | 9994c | ac31b6d | same |
| McNemar neurons vs 1.5B | 67 vs 21, p = 9.2e-7 (neurons better) | — | 630 | 9994c | ac31b6d | same |
| where each wins | RAG: 1h_now (19 vs 12 / 6), 7B 1h_sets; neurons: 2h_person, 2h_sets (15 vs 0) | — | 105 / cell | 9994c | ac31b6d | same |
| speed | RAG 25.0 s (1.5B) / 38.1 s (7B) wall per question; neurons 715.6 s for 10,452 questions on 5 hosts (~0.07 s per question, batched) | — | 630 / 10,452 | 9994c, 9991b | ac31b6d | same |
| corpus A k | k = 10 (recall 71.8) | — | dev | 9994e / 9994f | 5b80fd5 / 4b1e683 | 2026-11-16 — 9994e running; 2026-11-17 — VERDICT 9994f |
| corpus A dev exact (report only) | 1.5B 28.0%, 7B 37.0% | [20.1, 37.5] / [28.2, 46.8] | 100 | 9994f | 4b1e683 | 2026-11-17 — VERDICT 9994f_rag_biz_dev |
| corpus A test exact, all 5 cells, RAG 7B / 1.5B | 38.89% / 19.05% | [35.2, 42.8] / [16.2, 22.3] | 630 | 9994g | fd6651a | 2026-11-18 — VERDICT 9994g_rag_biz_test |
| corpus A test, 3 cells the neurons ran (1h_company, 2h_company, 2h_value): neurons (9993d rows) / RAG 7B / RAG 1.5B | 8.20% / 31.22% / 23.28% | [5.8, 11.4] / [26.8, 36.1] / [19.3, 27.8] | 378 | 9994g | fd6651a | same |
| McNemar neurons vs 7B | 2 vs 89, p = 3.4e-24 (RAG better) | — | 378 | 9994g | fd6651a | same |
| McNemar neurons vs 1.5B | 4 vs 61, p = 3.9e-14 (RAG better) | — | 378 | 9994g | fd6651a | same |
| by cell, neurons / 7B / 1.5B | 1h_company 24.6 / 84.9 / 65.9%; 2h_company 0.0 / 4.0 / 2.4%; 2h_value 0.0 / 4.8 / 1.6% (7B vs neurons p = 0.031); 1h_value 7B 17.5%, 1.5B 13.5%; 1h_year 7B 83.3%, 1.5B 11.9% (neurons not run on these two) | — | 126 / cell | 9994g | fd6651a | same |
| speed | RAG 36.4 s (1.5B) / 65.5 s (7B) median wall per question (server CPU 152 / 287 s); neurons 34.8 s for 2,729 questions on 5 hosts (~0.013 s per question, batched) | — | 630 / 2,729 | 9994g, 9993d | fd6651a | same |

## 7. Forgetting (corpus A, 5 hosts, gold plans, cold copies; job 9994a4b, commit a2897be, log 2026-11-17 — VERDICT 9994a4b_bench_forget)

| forgotten | leaks (Wilson upper) | audit hits | FORGET s | audit clean s | retain rows identical | forget q answered before → after |
|---|---|---|---|---|---|---|
| 30 | 0 / 197 (1.9%) | 0 | 0.95 | 2.68 | 12,739 / 12,739 | 191 → 164 |
| 150 | 0 / 933 (0.4%) | 0 | 1.63 | 3.71 | 11,991 / 11,991 | 903 → 773 |
| 300 | 0 / 1,843 (0.2%) | 0 | 1.69 | 3.63 | 11,063 / 11,063 | 1,781 → 1,514 |
RAG on the same benchmark (300 set, 60 forget + 30 retain questions; 9994e, 5b80fd5): leaks 0 (both models); retain
answers identical 21 / 30 (1.5B), 30 / 30 (7B) — baseline (not neurons).
Note: the first attempt 9994a4 failed its retain rule (rule-writing: the retain set ignored rows where a forgotten KU
had answered wrongly; those rows lost only that wrong answer); commit 5799f50.

## 8. Updating (corpus A, 500 edits, gold plans; job 9994a5, commit af3e1bf, log 2026-11-17 — VERDICT 9994a5_bench_edit)

| measure | after (vs new gold) | before (vs old gold) | n |
|---|---|---|---|
| edits retrained by exactly one host | 500 / 500 | | 500 |
| EDIT sent → new value answered | 1.06 s mean (1.04 median) | | 500 |
| efficacy, non-year | 72.99% [68.6, 77.0] | 73.22% [68.8, 77.2] | 422 |
| efficacy, known facts (new listed, old gone) | 99.15% [97.5, 99.7] | | 351 |
| efficacy, founded_year | 100.0% [95.3, 100] | | 78 |
| generalization (other wordings) | 77.2% [74.5, 79.7] | 77.4% [74.7, 79.9] | 1,000 |
| multi-hop through the edit | 46.5% [41.9, 51.1] | 44.1% [39.6, 48.7] | 454 |
| specificity, own other relations | 80.4% [77.9, 82.7] | 81.4% [79.0, 83.7] | 1,040 |
| specificity, neighbours sharing the old value | 100% rows identical | | 1,970 |
RAG on the same benchmark (60 efficacy + 30 neighbour questions; 9994e): new value listed and old gone 45 / 60 (1.5B),
50 / 60 (7B); neighbours identical 30 / 30, 29 / 30 — baseline (not neurons).


## 9. Limits (stated plainly; sources)

- **Accuracy is below text-reading systems.** On 2Wiki test3 the frozen text-reading baseline (stored text + shared
  extractor) answers 58.65% vs neurons 39.70% (9982b, p 1.2e-56); neurons reach 64% of their own oracle ceiling there,
  38% on MetaQA sets (42.64% ceiling). No comparison with current multi-hop QA systems (chapter4.md:265–270).
- **Layout relations** (dates in parentheses, release year, country) are near 0 for neurons on 2Wiki; MetaQA
  release_year 75.77% vs first slot 95.53% (README.md:181–183; 9985b). (Corpus A founded_year edits: 100% — template
  text.)
- **Uncued relations** are a stated limit (chapter4.md:250–257); has_genre 0.00% (not a slot).
- **Planner wording**: planners trained on template wordings fail unseen wordings (2Wiki paraphrases 60.33 → 19.00%,
  baseline; corpus A test 2-hop cells 0.0%, 9993d). Template questions throughout.
- **Confidence is narrow**: 2Wiki neurons flag 1 of 2,000 confident (9979); the MetaQA table failed its count rule
  (9989a); confident answers exist only for 2-hop replies with two findings at f = 1.0.
- **Comparison type** below its first-slot control on 2Wiki (72.90 vs 79.26%).
- **Scope**: one machine (5 host processes, Azure VM); at most one reverse step; sets only for MetaQA; corpus A is
  synthetic.
- **Security**: frames are not authenticated or encrypted; trusted LAN only (README.md:148–149, 202;
  chapter4.md:288–289).
- **RAG comparison scope**: CPU-only small models (1.5B Q8, 7B Q4), k fixed on dev, 630-question seeded subsets; speed
  compares per-question latency (RAG) with batched throughput (neurons).

## 10. Open notes for the editor (not results)

- README change-log order: the 2026-11-12 slot-cache row sits above newer rows (fix in the README rewrite).
- README principle 6 says "at least two copies (R >= 2)": that is the baseline; the neuron default is R = 1 (README.md:259).
- KU counts differ by pool (14,128 baseline pool; 14,114 test3; 14,177 fresh): label each with its pool.
- Intervals not given in the log (reported as "—"): 2- / 3-hop oracle ceilings, test3 cue / gap groups, 9991a dev
  overall, 9982b per type, 9967 random-slot control.
- 9985b cue-worded n corrected 856 → 854 (log 2026-11-05 17:12, 44bdfd8).
