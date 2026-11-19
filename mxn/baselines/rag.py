"""RAG baseline — baseline (not neurons): what a customer would otherwise run on the same CPU hardware
(reports/design_benchmark_pack.md §1; owner decisions D1, D5, 2026-11-09).

Index: one chunk per document (WikiMovies movie documents: title + its sentences), embedded with BAAI/bge-small-en-v1.5
on CPU, exact inner-product search (FAISS IndexFlatIP over normalized vectors). Reader: a local Qwen2.5 GGUF on our own
llama.cpp server (CPU, 127.0.0.1, greedy; mxn/baselines/llama_server.py), one fixed prompt. Never part of the system.

usage: rag.py build POOL.pkl OUT_DIR
       rag.py answer OUT_DIR QUESTIONS.json ROWS.jsonl --server=PATH --model=GGUF [--k=5] [--threads=8] [--port=7410]
              [--gold=GOLD.json (dry runs: question id → gold document titles, used instead of retrieval)]
              [--empty (dry runs, known-bad: no passages)]
→ build: OUT_DIR/{index.faiss, docs.json, build.json}; answer: one row per question
  {id, answer, answer_set, conf, complete, retrieved, cpu_s, wall_s, prompt_tokens, gen_tokens} (resumable).
"""
from __future__ import annotations

import json
import pickle
import resource
import sys
import time
from pathlib import Path

EMB = "BAAI/bge-small-en-v1.5"
QPREFIX = "Represent this sentence for searching relevant passages: "      # bge query instruction
PROMPT = ("Answer the question using only the passages below. List every answer, separated by \" | \". "
          "If the passages do not contain the answer, reply \"unknown\". Reply with the answer(s) only.\n\n"
          "{passages}\n\nQuestion: {question}\nAnswer:")


def cpu() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def proc_cpu(pid: int) -> float:
    """CPU seconds of another process (the llama.cpp server) from /proc."""
    import os
    f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
    return (int(f[11]) + int(f[12])) / os.sysconf("SC_CLK_TCK")


def docs_of(pool: str) -> list[dict]:
    M = pickle.load(open(pool, "rb"))
    sents = {k: [] for k in range(len(M["titles"]))}
    for s, o in zip(M["sentences"], M["owner"]):
        sents[o].append(s)
    return [{"title": t, "text": " ".join(sents[k])} for k, t in enumerate(M["titles"])]


def build(pool: str, out: Path) -> None:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    out.mkdir(parents=True, exist_ok=True)
    t0, c0 = time.time(), cpu()
    docs = docs_of(pool)
    m = SentenceTransformer(EMB, device="cpu")
    v = m.encode([f"{d['title']}. {d['text']}" for d in docs], batch_size=64, normalize_embeddings=True,
                 show_progress_bar=False).astype(np.float32)
    ix = faiss.IndexFlatIP(v.shape[1]); ix.add(v)
    faiss.write_index(ix, str(out / "index.faiss"))
    json.dump(docs, open(out / "docs.json", "w"))
    json.dump({"pool": pool, "embedder": EMB, "docs": len(docs), "dim": int(v.shape[1]),
               "wall_s": round(time.time() - t0, 1), "cpu_s": round(cpu() - c0, 1)}, open(out / "build.json", "w"), indent=1)


def parse(text: str) -> list[str]:
    """Parse v2 (post hoc from 9992a, reports/design_rag_rules.md §2): first line, split on "|", strip quotes and one
    trailing period, drop "unknown" items (7B appends " | unknown"), dedupe case-insensitively."""
    t = text.strip().split("\n")[0].strip()
    seen, out = set(), []
    for x in t.split("|"):
        x = x.strip().strip('"').strip()
        x = x[:-1] if x.endswith(".") else x
        x = x.strip().strip('"').strip()
        if x and x.lower() != "unknown" and x.lower() not in seen:
            seen.add(x.lower()); out.append(x)
    return out


def answer(d: Path, qfile: str, rows_out: str, opts: dict) -> None:
    import faiss
    from sentence_transformers import SentenceTransformer
    from mxn.baselines.llama_server import ServerLLM
    k = int(opts.get("k", 5))
    docs = json.load(open(d / "docs.json")); ix = faiss.read_index(str(d / "index.faiss"))
    by_title = {x["title"]: x for x in docs}
    gold = json.load(open(opts["gold"])) if opts.get("gold") else None
    done = set()
    if Path(rows_out).exists():
        done = {json.loads(l)["id"] for l in open(rows_out)}
    qs = [q for q in json.load(open(qfile)) if q[0] not in done]
    m = SentenceTransformer(EMB, device="cpu")
    llm = ServerLLM(opts["server"], opts["model"], port=int(opts.get("port", 7410)), threads=int(opts.get("threads", 8)),
                    ctx=8192, log=str(Path(rows_out).with_suffix(".server.log")))
    fh = open(rows_out, "a")
    try:
        for n, (qid, q) in enumerate(qs, 1):
            t0, c0, s0 = time.time(), cpu(), proc_cpu(llm.proc.pid)
            if "empty" in opts:
                hits = []
            elif gold is not None:
                hits = [by_title[t] for t in gold.get(qid, []) if t in by_title]
            else:
                qv = m.encode([QPREFIX + q], normalize_embeddings=True).astype("float32")
                _, I = ix.search(qv, k); hits = [docs[i] for i in I[0] if i >= 0]
            passages = "\n\n".join(f"[{i + 1}] {h['title']}: {h['text']}" for i, h in enumerate(hits)) or "(no passages)"
            out = llm.generate(PROMPT.format(passages=passages, question=q), max_new_tokens=96)
            s = parse(out)
            fh.write(json.dumps({"id": qid, "answer": " | ".join(s) or None, "answer_set": s, "conf": 1.0 if s else 0.0,
                                 "complete": True, "raw": out, "retrieved": [h["title"] for h in hits],
                                 "wall_s": round(time.time() - t0, 3), "cpu_s_client": round(cpu() - c0, 3),
                                 "cpu_s_server": round(proc_cpu(llm.proc.pid) - s0, 3),
                                 **llm.last}) + "\n")
            fh.flush()
            if n % 50 == 0:
                print(time.strftime("%H:%M:%S"), f"rag {n}/{len(qs)}", flush=True)
    finally:
        fh.close(); llm.close()


if __name__ == "__main__":
    opts = dict((a[2:].split("=", 1) + [""])[:2] for a in sys.argv if a.startswith("--"))
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args[0] == "build":
        build(args[1], Path(args[2]))
    else:
        answer(Path(args[1]), args[2], args[3], opts)
