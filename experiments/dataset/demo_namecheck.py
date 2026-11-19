"""Wikidata name check for demo corpus A (reports/design_demo_corpus.md §6; owner decision D3, 2026-11-09).

Every generated proper name (companies, people, cities, products; names.json from demo_gen.py) is looked up on the
Wikidata query service as an exact English label or alias of any item (stricter than the design's "company and
person labels": any item type counts). Hits are added to the drop list; demo_gen.py then never draws them.
Wikidata structured data is CC0 (license page fetched 2026-11-12T16:41Z, artifacts/demo/license, sha256 0939c49d...).
Only generated fictional names are sent; nothing else. Exact match is case-sensitive (labels and our names are both
title case); a case variant is not checked (limitation, stated).
Resumable: names already checked are cached in CACHE.json.

usage: python demo_namecheck.py NAMES.json CACHE.json HITS.json
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

URL = "https://query.wikidata.org/sparql"
UA = "t24-demo-namecheck/0.1 (fictional name collision check)"
BATCH = 200


def lookup(names: list[str]) -> set[str]:
    vals = " ".join(json.dumps(n) + "@en" for n in names)
    q = (f"SELECT DISTINCT ?l WHERE {{ VALUES ?l {{ {vals} }} "
         f"{{ ?i rdfs:label ?l }} UNION {{ ?i skos:altLabel ?l }} }}")
    body = urllib.parse.urlencode({"query": q}).encode()
    req = urllib.request.Request(URL, data=body, headers={"User-Agent": UA, "Accept": "application/sparql-results+json",
                                                          "Content-Type": "application/x-www-form-urlencoded"})
    for attempt in range(5):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=120))
            return {b["l"]["value"] for b in r["results"]["bindings"]}
        except Exception as e:                                   # 429 / 5xx / timeout: back off and retry
            print(time.strftime("%H:%M:%S"), "retry", attempt, repr(e)[:120], flush=True)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("Wikidata query failed 5 times")


def main(names_f: str, cache_f: str, hits_f: str) -> None:
    N = json.load(open(names_f))
    names = sorted({n for v in N.values() for n in v})
    cache = json.load(open(cache_f)) if Path(cache_f).exists() else {}
    todo = [n for n in names if n not in cache]
    for i in range(0, len(todo), BATCH):
        part = todo[i:i + BATCH]
        hit = lookup(part)
        cache.update({n: n in hit for n in part})
        json.dump(cache, open(cache_f, "w"))
        print(time.strftime("%H:%M:%S"), f"checked {min(i + BATCH, len(todo))}/{len(todo)}, hits so far "
              f"{sum(cache.values())}", flush=True)
        time.sleep(1)
    old = set(json.load(open(hits_f))) if Path(hits_f).exists() else set()
    hits = sorted(old | {n for n, h in cache.items() if h})
    json.dump(hits, open(hits_f, "w"), indent=0)
    print(f"names {len(names)}, new checked {len(todo)}, hits in this set {sum(cache.get(n, False) for n in names)}, "
          f"drop list {len(hits)}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
