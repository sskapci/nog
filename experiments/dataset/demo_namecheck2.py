"""Wikidata name check v2 for demo corpus A (owner request 2026-11-12, on top of demo_namecheck.py's exact check).

Fictional companies (synthetic data generated for this demo). Any resemblance to real companies or people is
unintended.

A generated name is a hit (dropped) when some Wikidata item has an English or "mul" label or alias that
  1. equals it case-insensitively (casefold): "exact-ci";
  2. is within Levenshtein distance 1 of it on the full name, case-insensitively: "edit1";
  3. (company and product names) contains its brand word, the first word, as a whole word (casefold), and the item
     is a company or brand (direct P31 in demo_wdlabels.BRAND_CLASSES; no subclass walk: limitation): "brand".
Runs locally on the label export of demo_wdlabels.py (CC0); nothing is sent anywhere. Edit distance <= 1 by
symmetric deletion (a key set of each name and its one-character deletions; each label and its deletions are
looked up), then confirmed by an exact Levenshtein check.
An API version (Wikimedia search per name) was tried first: correct on the dry run, but HTTP 429 at ~20 s per name.

usage: python demo_namecheck2.py NAMES.json WD_DIR HITS.json [REPORT.json]
-> HITS.json gets every hit name added; REPORT.json: per hit name, the rule, the matching label and its file.
"""
from __future__ import annotations

import gzip
import json
import re
import sys
import time
from pathlib import Path


def literal(line: str) -> str:
    s = line.rstrip("\n")
    if s.startswith("<"):                                       # brand.tsv: "?i\t?l"
        s = s.split("\t", 1)[1]
    j = s.rfind('"')
    return s[1:j].replace('\\"', '"').replace("\\\\", "\\") if s.startswith('"') and j > 0 else s


def lev1(a: str, b: str) -> bool:
    """Levenshtein distance <= 1."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i + (len(a) == len(b)):] == b[i + 1:]


def dels(s: str):
    return (s[:i] + s[i + 1:] for i in range(len(s)))


def main(names_f: str, wd: str, hits_f: str, report_f: str | None) -> None:
    N = json.load(open(names_f))
    kinds = {n: k for k, v in N.items() for n in v}
    key: dict[str, set[str]] = {}
    for n in kinds:
        c = n.casefold()
        for d in (c, *dels(c)):
            key.setdefault(d, set()).add(n)
    lens = {len(n.casefold()) for n in kinds}
    lo, hi = min(lens) - 1, max(lens) + 1
    hit: dict[str, list] = {}
    files = sorted(Path(wd).glob("all_*.tsv.gz"))
    t0, rows = time.time(), 0
    for f in files:
        with gzip.open(f, "rt", encoding="utf-8") as g:
            for line in g:
                rows += 1
                l = literal(line)
                c = l.casefold()
                if not lo <= len(c) <= hi:
                    continue
                cand = set(key.get(c, ()))
                for d in dels(c):
                    if d in key:
                        cand |= key[d]
                for n in cand:
                    nc = n.casefold()
                    if n not in hit and lev1(c, nc):
                        hit[n] = ["exact-ci" if c == nc else "edit1", l, f.name]
        print(time.strftime("%H:%M:%S"), f.name, f"rows {rows}, hits {len(hit)}, {time.time() - t0:.0f} s", flush=True)
    brand_words: dict[str, set[str]] = {}
    for n in kinds:
        if kinds[n] in ("companies", "products"):
            brand_words.setdefault(n.split()[0].casefold(), set()).add(n)
    bf = Path(wd) / "brand.tsv.gz"
    with gzip.open(bf, "rt", encoding="utf-8") as g:
        next(g)
        for line in g:
            l = literal(line)
            for w in re.split(r"[^\w]+", l.casefold()):
                for n in brand_words.get(w, ()):
                    if n not in hit:
                        hit[n] = ["brand", l, bf.name]
    by: dict[str, int] = {}
    for n, h in hit.items():
        by[h[0]] = by.get(h[0], 0) + 1
    old = set(json.load(open(hits_f))) if Path(hits_f).exists() else set()
    json.dump(sorted(old | set(hit)), open(hits_f, "w"), indent=0)
    if report_f:
        json.dump({"names": len(kinds), "label_rows": rows, "hits": len(hit), "by_rule": by,
                   "by_kind": {k: sum(1 for n in hit if kinds[n] == k) for k in N}, "detail": hit},
                  open(report_f, "w"), indent=1)
    print(f"names {len(kinds)}, label rows {rows}, hits in this set {len(hit)} {by}, drop list {len(old | set(hit))}")


if __name__ == "__main__":
    main(*sys.argv[1:4], sys.argv[4] if len(sys.argv) > 4 else None)
