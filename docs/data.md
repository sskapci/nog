# Data and attribution

NOG's own code and text are licensed under the GNU Affero General Public License v3.0 only (`LICENSE`; SPDX
`AGPL-3.0-only`). Third-party data and text derived from it are not covered by that license and keep their own
licenses. Datasets, model weights and built networks are not redistributed; `docs/reproduce.md` rebuilds every input
from the public sources.

**Models trained on MetaQA are research artifacts.** The MetaQA planner and the MetaQA KUs are trained on data partly
based on MovieLens and OMDb, whose terms forbid commercial use without permission. They are not covered by the
commercial license; commercial use needs models trained on data without such restrictions (for example the
fictional-company corpus, which is our own).

| Source | Used for | License | Files in this repository with derived text |
|---|---|---|---|
| **Simple English Wikipedia** (`wikimedia/wikipedia`, config `20231101.simple`, via Hugging Face), © Wikipedia contributors | Phase 1–2 knowledge units; source sentences of the template and model-written questions | CC BY-SA 3.0 and GFDL (as listed on the dataset card; current Wikipedia text is CC BY-SA 4.0) | none (earlier phases; not in this snapshot); short examples in `docs/*.md` |
| **2WikiMultiHopQA** (Ho et al., COLING 2020, [Alab-NII/2wikimultihop](https://github.com/Alab-NII/2wikimultihop)) | Questions, answers, evidence triples; its context paragraphs (from English Wikipedia, © Wikipedia contributors) are the expert system's knowledge units | Apache-2.0 (repository); the Wikipedia paragraphs are CC BY-SA | none; short examples in `docs/*.md` and `evidence/*.md` |
| **Wikimedia pageviews** (Wikimedia Analytics) | Popularity weights for query streams | CC0 1.0 | none (not in this snapshot) |
| **Qwen-generated text** (Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct, Apache-2.0 models, run locally) | Held-out questions (1.5B, from Simple Wikipedia sentences); 2Wiki paraphrases (7B); B2 answers (1.5B) | The models are released under Apache-2.0 (model cards and LICENSE file, checked 2026-11-18); that license has no terms on model outputs. We claim no rights in these outputs and treat each as carrying the license of the input text it was generated from (CC BY-SA for Wikipedia-based questions; Apache-2.0 / CC BY-SA for 2Wiki paraphrases). This is our reading, not legal advice. | none (not in this snapshot) |
| **MetaQA** (Zhang et al., AAAI 2018, [yuyuz/MetaQA](https://github.com/yuyuz/MetaQA); © the MetaQA authors) and **WikiMovies** (Miller et al., EMNLP 2016; © 2016-present Facebook, Inc.), both partly based on **MovieLens** (GroupLens Research, University of Minnesota; Harper and Konstan, 2015) and **OMDb** data | Movie documents as KUs (WikiMovies knowledge source); 1/2/3-hop questions | MetaQA and WikiMovies: CC BY 3.0 Unported. The underlying MovieLens terms allow research use only, forbid redistribution without permission and commercial use without permission from GroupLens; OMDb content is CC BY-NC 4.0 (non-commercial). These files are research data, **not covered by the repository's commercial license**. Data used: the Hugging Face mirror camazlucas/MetaQA, byte-identical to the official copy linked from the MetaQA README (KB and all 9 vanilla question files, sha256, checked 2026-11-18) | ids only: `reproduce/rag/mqs_test630_ids.json`, `mqs_dev100_ids.json` (+ `mqs_subsets.json`: sizes, seeds, hashes). No MetaQA question text is in this repository; `experiments/baselines/rag_mq_text.py` rebuilds the RAG question files from the public copy and checks their hashes |
| **Wikidata** (labels and aliases, via the QLever endpoint) | Name check of the fictional corpus (generated names must not match real entities) | CC0 1.0 | `experiments/dataset/demo_hits.json` (generated names that matched a label) |
| **2WikiMultiHopQA**, further derived files | Error analyses and extractor triggers | as above | none (not in this snapshot) |
| **Software and models used, not redistributed** | llama.cpp b11433 (MIT); BAAI/bge-small-en-v1.5 (MIT); FAISS (MIT); sentence-transformers (Apache-2.0); Qwen2.5 1.5B / 7B Instruct GGUF (Apache-2.0) | as stated | none |
| **Corpus A** (own generator) | Fictional companies, people, cities, products. Fictional companies (synthetic data generated for this demo). Any resemblance to real companies or people is unintended. | AGPL-3.0-only (own) | examples in `docs/*.md` and `evidence/*.md` |

Fictional-company numbers are on synthetic data generated for this demo; any resemblance to real companies or people
is unintended.
