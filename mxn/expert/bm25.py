"""Okapi BM25 over passages (an algorithm: no learned weights). Sparse term-passage matrix, so a
query scores every passage with one sparse product."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import scipy.sparse as sp

WORD = re.compile(r"\w+", re.UNICODE)


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


class BM25:
    def __init__(self, passages: list[str], tokenize=words, k1: float = 1.2, b: float = 0.75):
        self.tok, self.k1, self.b = tokenize, k1, b
        self.vocab: dict[str, int] = {}
        rows, cols, vals = [], [], []
        lens = np.zeros(len(passages), np.float32)
        for j, p in enumerate(passages):
            c = Counter(self.tok(p))
            lens[j] = sum(c.values())
            for t, n in c.items():
                rows.append(self.vocab.setdefault(t, len(self.vocab))); cols.append(j); vals.append(n)
        tf = sp.csr_matrix((np.array(vals, np.float32), (rows, cols)), shape=(len(self.vocab), len(passages)))
        df = np.diff(tf.indptr).astype(np.float32)               # passages containing each term
        n = len(passages)
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        norm = k1 * (1 - b + b * lens / max(lens.mean(), 1e-9))
        tf = tf.tocoo()
        w = tf.data * (k1 + 1) / (tf.data + norm[tf.col])
        self.W = sp.csr_matrix((w.astype(np.float32), (tf.row, tf.col)), shape=tf.shape)   # term x passage

    def query_vector(self, text: str) -> sp.csr_matrix:
        c = Counter(t for t in self.tok(text) if t in self.vocab)
        idx = np.array([self.vocab[t] for t in c], np.int64)
        val = np.array([self.idf[self.vocab[t]] * n for t, n in c.items()], np.float32)   # repeated terms count
        return sp.csr_matrix((val, (np.zeros(len(idx), np.int64), idx)), shape=(1, len(self.vocab)))

    def scores(self, texts: list[str]) -> np.ndarray:
        """Dense (queries x passages) BM25 scores."""
        Q = sp.vstack([self.query_vector(t) for t in texts]) if texts else sp.csr_matrix((0, len(self.vocab)))
        return (Q @ self.W).toarray()
