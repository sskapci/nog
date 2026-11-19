"""Rebuild the MetaQA RAG question files from question ids and the public MetaQA copy — baseline (not neurons).

The repository keeps only question ids for the MetaQA RAG subsets (MetaQA / WikiMovies are partly based on MovieLens
and OMDb data, whose terms do not allow redistribution). This script
writes the question files the RAG jobs read (9994b, 9994c):
  python experiments/baselines/rag_mq_text.py METAQA_SETS_DIR [OUT_DIR]
  METAQA_SETS_DIR: artifacts/data/metaqa_sets, built from the official MetaQA copy by experiments/dataset/metaqa_prep*.py
  OUT_DIR (default reports/rag): writes mqs_test630.json and mqs_dev100.json as [[qid, question], ...], in the order
  of reproduce/rag/mqs_{test630,dev100}_ids.json, and checks each file's sha256 against reproduce/rag/mqs_subsets.json
  (the hashes of the files the jobs ran with). Exit 1 on any mismatch or missing id.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RAG = REPO / "reproduce" / "rag"


def main(data: Path, out: Path) -> int:
    q = {r["_id"]: r["question"] for r in json.load(open(data / "dev.json"))}
    meta = json.load(open(RAG / "mqs_subsets.json"))["files"]
    out.mkdir(parents=True, exist_ok=True)
    bad = 0
    for name in ("mqs_test630.json", "mqs_dev100.json"):
        ids = json.load(open(RAG / name.replace(".json", "_ids.json")))
        missing = [i for i in ids if i not in q]
        if missing:
            print(f"{name}: {len(missing)} ids not in {data / 'dev.json'} (first {missing[0]})")
            bad += 1
            continue
        f = out / name
        json.dump([[i, q[i]] for i in ids], open(f, "w"))  # same format as rag_prep.py subsets
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        ok = h == meta[name]["sha256"]
        print(f"{name}: {len(ids)} questions, sha256 {'OK' if ok else 'MISMATCH ' + h}")
        bad += not ok
    return 1 if bad else 0


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.exit(main(Path(a[0]), Path(a[1]) if len(a) > 1 else REPO / "reports" / "rag"))
