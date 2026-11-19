"""B2 baseline prompt and answer helpers (Qwen2.5-1.5B reading one KU): copied unchanged from mxn/reasoner.py, the
retired Qwen-as-reasoner pipeline (cleanup 2026-10-30; README History §8.2)."""

from __future__ import annotations

import re
import string


ANSWER = """Facts:
{facts}

Answer the question using only the facts above. Reply with a short phrase only (a name, a date, a
place, or yes/no). If the facts do not contain the answer, reply "unknown".
Question: {q}
Answer:"""


def normalize(s: str) -> str:
    """The official 2Wiki answer normalization."""
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def short(a: str) -> str:
    a = a.strip().split("\n")[0].strip().strip('"').rstrip(".")
    return a


def is_unknown(a: str) -> bool:
    n = normalize(a)
    return not n or n in ("unknown", "not known", "cannot be determined", "no answer") or n.startswith("unknown")
