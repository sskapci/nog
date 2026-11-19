# Evidence for the headline results
For every headline job: its rules as committed before the run (`<job>.rules.md`: the experiment-log entries and, where used, the design document of the rules commit, copied unchanged) and its verdict file as the job wrote it (`<job>.verdict.json`). Commit hashes and log entries refer to the private development repository. The code each job ran is in this snapshot's release (`reproduce/`, `docs/reproduce.md`).

A verdict file's `pass` is the job's own check. Test jobs pass only at tier FLIES, so 9985b (tier PARTLY) records `pass: false`; its tier is the result (docs/results.md §3a).

| job | headline result (docs/results.md) | rules commit | logged verdict |
|---|---|---|---|
| 9976_neuron_rebalance | §4b live rebalancing 5→6→5, answers identical | 84ca908 | 2026-11-02 20:41 UTC — VERDICT 9976 neuron_rebalance: **PASS** (rules 84ca908) |
| 9978_neuron_cold | §4b cold copies, two live kills | f8894b3 | 2026-11-03 01:25 UTC — VERDICT 9978 neuron_cold: **PASS** (rules f8894b3) |
| 9979_vk_test3 | §2a 2Wiki test3 FLIES 39.70% | 68c00ce | 2026-11-03 21:49 UTC — VERDICT 9979 vk_test3: **PASS, tier FLIES** (one attempt; rules 68c00ce; dev choices frozen from 9977) |
| 9980_neuron_repl | §4b replication R = 2 | 3928753 | 2026-11-03 15:51 UTC — VERDICT 9980 neuron_repl: **PASS** (rules 3928753) |
| 9981_neuron_locality | §4b locality placement | 3ee4adf | 2026-11-03 22:44 UTC — VERDICT 9981 neuron_locality: **PASS** (rules eed753e; code branch loc 3ee4adf) |
| 9982b_baseline_vktest | §2b text-reading baseline on test3 58.65% (baseline, not neurons) | 63e18da | 2026-11-04 10:13 UTC — VERDICT 9982b baseline_vktest: PASS (baseline, not neurons; report only; no effect on 9979) |
| 9985a_mq_dev | §3a MetaQA 1-hop dev gate | 49efe13 | 2026-11-05 03:55 UTC — VERDICT 9985a mq_dev: PASS (MetaQA 1-hop dev; dev-pick gate holds) |
| 9985b_mq_test | §3a MetaQA 1-hop test PARTLY 55.40% | 49efe13 | 2026-11-05 12:37 UTC — VERDICT 9985b mq_test: tier PARTLY (MetaQA 1-hop test; one attempt) |
| 9987a_mqmh_dev | §3b MetaQA 2/3-hop dev | f871f38 | 2026-11-06 08:18 UTC — VERDICT 9987a mqmh_dev: PASS (MetaQA 2- / 3-hop dev; both hop counts pass the dev-pick gate) |
| 9987b_mqmh_test | §3b MetaQA 2-hop 49.34% / 3-hop 13.45% FLIES | f871f38 | 2026-11-06 13:52 UTC — VERDICT 9987b MetaQA 2-/3-hop test: PASS (2-hop FLIES, 3-hop FLIES); VERDICT 9988a PASS |
| 9990a_mqs_planner_train | §3c sets planner | 8af9fb8 | 2026-11-06 23:25 UTC — VERDICTS 9990a mqs_planner_train PASS, 9990b mqs_dryrun PASS; idle gap |
| 9991a_mqs_dev | §3c sets dev | 368ce83 | 2026-11-08 15:05 UTC — VERDICT 9991a mqs_dev PASS (8 of 10 cells pass the dev-pick gate); 9991b released |
| 9991b_mqs_test | §3c sets test, 6 cells FLY, 16.19% | 368ce83 | 2026-11-09 20:02 UTC — VERDICT 9991b mqs_test: all 6 tested cells FLIES (oracle still running) |
| 9991g_cache_test | §4b slot cache 10,452 / 10,452 identical | 879ee8f | 2026-11-12 19:29Z — VERDICT 9991g_cache_test: PASS (slot cache identity on all 10,452 9991b test questions) |
| 9993a_biz_planner | §5 corpus A planner | 6bc7678 | 2026-11-13 15:18Z — VERDICT 9993a_biz_planner: PASS (hold-out exact 100.0) |
| 9993b2_biz_dryrun | §5 corpus A dry runs | 099e99d | 2026-11-13 18:30Z — VERDICT 9993b2_biz_dryrun (attempt 2): PASS (20 / 20) |
| 9993c_biz_dev | §5 corpus A dev | 099e99d | 2026-11-13 20:41Z — VERDICT 9993c_biz_dev: PASS — theta_f 0.7; cells to test 1h_company, 2h_company, 2h_value |
| 9993d_biz_test | §5 corpus A test (1h_company FLIES) | 099e99d | 2026-11-14 23:52Z — VERDICT 9993d_biz_test: PASS (1h_company FLIES; 2h_company, 2h_value do not) — limit: planner on unseen wordings |
| 9994a3b_bench_dry | §7–8 forget / edit dry runs | 0519f4a | 2026-11-16 15:52Z — VERDICT 9994a3b_bench_dry (attempt 2): PASS (7 / 7) |
| 9994a4b_bench_forget | §7 forgetting | 5799f50 | 2026-11-17 03:35Z — VERDICT 9994a4b_bench_forget (attempt 2): PASS — forgetting benchmark on corpus A (gold plans) |
| 9994a5_bench_edit | §8 updating | 0519f4a | 2026-11-17 03:15Z — VERDICT 9994a5_bench_edit: PASS — updating benchmark on corpus A (gold plans) |
| 9994b_rag_mq_dev | §6 RAG MetaQA dev (baseline, not neurons) | c05e28d | 2026-11-16 02:26Z — VERDICT 9994b_rag_mq_dev: PASS — baseline (not neurons); k = 3; dev 100 |
| 9994c_rag_mq_test | §6 RAG MetaQA test (baseline, not neurons) | c05e28d | 2026-11-16 14:42Z — VERDICT 9994c_rag_mq_test: PASS — RAG on MetaQA test 630 — baseline (not neurons) |
| 9994e_rag_bench | §7–8 RAG forget / edit arms (baseline, not neurons) | 9a759b0 | 2026-11-17 02:30Z — VERDICT 9994e_rag_bench: PASS — RAG forget / edit arms on corpus A — baseline (not neurons) |
| 9994f_rag_biz_dev | §6 RAG corpus A dev (baseline, not neurons) | 2cc957e | 2026-11-17 06:51Z — VERDICT 9994f_rag_biz_dev: PASS — RAG on demo corpus A, dev 100 — baseline (not neurons) |
| 9994g_rag_biz_test | §6 RAG corpus A test (baseline, not neurons) | 2cc957e | 2026-11-18 02:43Z — VERDICT 9994g_rag_biz_test: PASS — RAG on demo corpus A, test 630 — baseline (not neurons) |
