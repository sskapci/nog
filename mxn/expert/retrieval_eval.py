"""Supporting-fact recall at KU level for exact central search (as experiments/phase3/p3_retrieval.py):
rank sentences by score, take the first k distinct KUs; recall = share of a question's supporting
facts whose title is cited, averaged per question; "all" = every supporting title cited."""

from __future__ import annotations

import numpy as np


def first_k_titles(scores: np.ndarray, owner: np.ndarray, k: int, depth: int = 400) -> list[int]:
    top = np.argpartition(-scores, min(depth, len(scores) - 1))[:depth]
    top = top[np.argsort(-scores[top])]
    out = []
    for s in top:
        o = int(owner[s])
        if o not in out:
            out.append(o)
            if len(out) == k:
                break
    return out


def recall(cited: list[list[int]], gold: list[list[int]]) -> dict:
    r = np.array([np.mean([g in set(c) for g in gs]) for c, gs in zip(cited, gold)])
    a = np.array([all(g in set(c) for g in gs) for c, gs in zip(cited, gold)])
    return {"recall": float(100 * r.mean()), "all": float(100 * a.mean()), "per_q": r.tolist(), "all_q": a.tolist()}
