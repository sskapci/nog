# Job 9982b: code as it ran

Job 9982b is the text-reading baseline (labelled "baseline (not neurons)") on the 2Wiki test3 questions:
58.65% [56.5, 60.8] on 2,000 questions (`docs/results.md` §2b; private development log, entry "2026-11-04 10:13 — VERDICT 9982b").

It ran from a commit on an experiment branch of the development repository. That commit changed two files only, and
this directory holds them exactly as they were:

- `experiments/expert/conn_run.py`: adds the `vktest` pool (test3 of the value-kind attempt, `pool_vktest.pkl` from
  `experiments/neuron/vk_prep.py`; base mode only, the baseline, not neurons).
- `experiments/expert/audit_hashes.py`: adds the `vktest` audit hash sets and an optional POOL argument.

The job script as it ran is `reproduce/original/9982b_baseline_vktest.sh`.
