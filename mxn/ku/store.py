"""The KU store: one directory per KU set (e.g. artifacts/expert/kus/net), the format the chapter 1 network used.

  units.jsonl          one KU per line: {"id", "title", "sentences", "n", "near", "long"} (older stores: experts.jsonl);
                       "near" / "long" are the retired walk's links, written empty and ignored here
  chunks.npy           float16 sentence vectors of all KUs, in line order (n rows per KU)
  meta.json            {"hubs", "embedding_model"}

Moved out of mxn/graph/network.py (Network.save / Network.load) in the cleanup of 2026-10-30. KU calibration (the
walk's O1 unit confidence, p_KU) is retired (decision 2026-10-30): stores built before carry questions.jsonl,
question_vecs.npy and meta "confidence_weights", an unused field that is not read; new stores do not write it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class KU:
    """One knowledge unit as stored: its title, its sentences and their vectors."""

    def __init__(self, unit_id: str, title: str, sentences: list[str], chunk_vecs: np.ndarray):
        if len(sentences) != len(chunk_vecs) or len(sentences) == 0:
            raise ValueError("unit needs one vector per sentence and at least one sentence")
        self.id, self.title, self.sentences = unit_id, title, sentences
        self.chunk_vecs = np.ascontiguousarray(chunk_vecs, dtype=np.float32)


def load_store(path: str | Path, expect_model: str | None = None) -> list[KU]:
    """The store's KUs in file order. expect_model: the encoder name the caller embeds with; a store built with
    another one is refused (as Network.load did against the config's embedding model)."""
    path = Path(path)
    vecs = np.load(path / "chunks.npy").astype(np.float32)
    meta = json.loads((path / "meta.json").read_text())
    built = meta.get("embedding_model")
    if built and expect_model and built.split("/")[-1] != expect_model.split("/")[-1]:
        raise ValueError(f"{path} was built with {built}, config uses {expect_model}; "
                         f"queries would land in a different concept space. Rebuild or change the config.")
    kus, pos = [], 0
    rows = path / "units.jsonl"
    with open(rows if rows.exists() else path / "experts.jsonl") as f:
        for line in f:
            row = json.loads(line)
            n = row["n"]
            kus.append(KU(row["id"], row["title"], row["sentences"], vecs[pos: pos + n].copy()))
            pos += n
    return kus


def write_store(path: str | Path, kus: list[KU], embedding_model: str) -> None:
    """Write KUs in the store format (no walk links: "near" / "long" empty, no hubs; no calibration)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "chunks.npy", np.concatenate([u.chunk_vecs for u in kus]).astype(np.float16))
    with open(path / "units.jsonl", "w") as f:
        for u in kus:
            f.write(json.dumps({"id": u.id, "title": u.title, "sentences": u.sentences,
                                "n": len(u.sentences), "near": [], "long": []}) + "\n")
    (path / "meta.json").write_text(json.dumps({"hubs": [], "embedding_model": embedding_model}))
