# Reproducing NOG's headline results

Every headline number in [results.md](results.md) comes from one job. `reproduce/` holds each job's **original script,
as it ran** (`reproduce/original/`; the only change: the author's home directory is written `/home/USER/`; commit
hashes in comments refer to the private development repository), the **commit** it ran at (`reproduce/jobs.tsv`), and a
wrapper that re-runs it with the code of that commit. Each job's rules and verdict file are in `evidence/`.

```
reproduce/run_job.sh JOB DATA_DIR VENV_DIR [WORK_DIR]      # REPRODUCE_DRY=1 to set up without running
```

The code of each job commit is a source archive attached to this snapshot's GitHub release (`snapshot-2026-11-19`;
SHA-256 checksums in `reproduce/SHA256SUMS`; `REPRODUCE_ARCHIVES=DIR` uses archives already downloaded). The wrapper downloads
the job's archive, checks its checksum, unpacks it into `WORK_DIR/JOB`, links `artifacts/` to your prepared inputs and `.venv/` to your
Python environment, points the script's working-directory line at that directory, and runs it. The job writes its
outputs and its verdict file (named in the script's `# VERDICT:` line) under `artifacts/`. Compare them with the
numbers in [results.md](results.md) (section given in `jobs.tsv`) and with the job's verdict file in `evidence/`.

## Requirements

- Linux, CPU only; 8 cores and ~32 GB RAM were used (the 2Wiki and MetaQA runs keep 5 host processes in memory).
- Python with the packages in `pyproject.toml` (`pip install -e .[rag]` for the RAG baseline: faiss-cpu 1.9.0,
  sentence-transformers 6.1.0),
  llama.cpp build b11433 (`llama-server`) and the Qwen2.5-1.5B-Instruct Q8_0 and Qwen2.5-7B-Instruct Q4_K_M GGUF files.
- Ports 7400–7409, 7499 (neuron hosts) and 7410 (llama server) free on 127.0.0.1. **Frames are not authenticated or
  encrypted: run only on a trusted machine or LAN.**

## Inputs (prepared once, into DATA_DIR)

Datasets are not redistributed. Build each input from the public source with the scripts in this repository:

| input (under artifacts/) | source | built by |
|---|---|---|
| `data/2wiki/extracted`, `expert/` (encoder, tokenizer, planner) | 2WikiMultiHopQA (fixed segmentation) | "Building the 2Wiki inputs" below |
| `expert_mq*`, `data/metaqa_*`, `reports/neuron/mq*_ids.json` | MetaQA + WikiMovies | `experiments/dataset/metaqa_prep.py`, `metaqa_prep_mh.py`, `metaqa_prep_sets.py` (usage in each file; the id files are also in each job's archive) |
| `demo/a3`, `expert_biz`, `data/biz`, `bench_biz` | own generator (fictional companies) + Wikidata labels (CC0) for the name check | `experiments/dataset/`: `demo_gen.py --make-pool`, `demo_wdlabels.py`, `demo_namecheck2.py`, `demo_gen.py --pool`, `demo_prep.py`, `bench_prep.py` |
| `neuron/dev/opt_pick.json` | 2Wiki dev | the neuron settings pick (hidden 128, 600 epochs, lr 0.03, rich signal) |
| `gguf/`, `llamacpp/b11433/` | Qwen2.5 GGUF (Apache-2.0), llama.cpp (MIT) | downloaded |

The question-id files and seeded subsets the jobs read are in each job's source archive (`reports/neuron/`,
`reports/rag/`). MetaQA question text is not (ids in `reproduce/rag/`): for 9994b and 9994c, `run_job.sh` rebuilds `reports/rag/mqs_{test630,dev100}.json` in the
job directory from `DATA_DIR/data/metaqa_sets` with `experiments/baselines/rag_mq_text.py` and stops if a hash differs.

## Building the 2Wiki inputs

Setup (Python 3.11, CPU-only torch):

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch
uv pip install --python .venv/bin/python -e '.[dev]'
export PYTHONPATH=$PWD
pytest                                       # unit tests
```

Data: 2WikiMultiHopQA (version with fixed sentence segmentation and evidence ids) unpacked in
`artifacts/data/2wiki/extracted`; models and KU builds go to `artifacts/expert` (both untracked). Below,
`D=artifacts/data/2wiki/extracted` and `E=artifacts/expert`.

Build the expert (each script prints its own usage):

```bash
python experiments/expert/prep.py $D $E                                    # pools (KU texts) and training pairs
python experiments/expert/tokenizer.py $D $E                               # our tokenizer (16,000 pieces)
python experiments/expert/train_encoder.py $E $E/enc --hours=8             # our encoder (resumable: --resume=CK.pt)
python experiments/expert/build_kus.py $E $E/enc $E/kus --pool=pool_test.pkl   # KU store (mxn/ku); --limit=N smoke
python experiments/expert/planner_run.py $E $E/enc $E/planner2 --stage=train --plans=plans2   # then --stage=test
python experiments/expert/extract2_run.py $E $E/enc $E/extract2 --stage=train  # then --stage=threshold, test
```

The country-equivalence table is built by `experiments/expert/country_table.py` (needs Simple Wikipedia page and
redirect SQL dumps).

A single host is `python -m mxn.expert.replica --package PKG --name NAME --hostlist HOSTS.json --R 2 --expert $E`.
Code layout: `mxn/neuron` (KU neurons: KU, host, gateway), `mxn/expert` (supplements: encoder, planner, extractor,
operations, confidence; and the frozen text-reading baseline), `mxn/ku` (KU store), `mxn/common` (shared pieces),
`mxn/baselines` (RAG and Qwen baselines, never part of the system), `experiments/` (input preparation), `tests/`.

## Order

Gated jobs read their gate's outputs: run `9985a` before `9985b`, `9987a` before `9987b`, `9990a` → `9991a` → `9991b`
→ `9991g`, `9993a` → `9993b2` → `9993c` → `9993d`, `9994a3b` → `9994a4b` / `9994a5`, `9994b` → `9994c`,
`9994f` → `9994g`.

## Commits

The commits in `reproduce/jobs.tsv` are in the private development repository; their code is public as the release archives.
9982b's commit (63e18da) is from an experiment branch; the two files it changed are also in `reproduce/9982b/`.

## Exactness

Runs are deterministic on the same machine: repeated runs gave identical rows (noise-floor checks in every dry run;
9994a4b reproduced 9994a4's rows exactly). Wall-clock and CPU times depend on the machine. Four early jobs (9976,
9978, 9979, 9980) ran from the working branch, not a pinned worktree; `jobs.tsv` gives the branch head at their start
time and their rules commit; uncommitted edits at that moment cannot be ruled out.

## Tests

`pytest` runs the unit tests (94 pass; 3 skip unless the built inputs are present under `artifacts/`).
