"""Export Wikidata labels for the demo corpus A name check v2 (owner request 2026-11-12).

Fictional companies (synthetic data generated for this demo). Any resemblance to real companies or people is
unintended.

Downloads, from the QLever Wikidata SPARQL endpoint (qlever.dev; Wikidata structured data is CC0, license page
fetched 2026-11-12T16:41Z, artifacts/demo/license), every English and "mul" label and alias of every item (one
literal per line), plus the labels and aliases of items that are a company or brand (direct P31 in BRAND_CLASSES; one query per
class, merged and deduplicated locally, after the single query got HTTP 429 for 2 h).
Nothing is sent but these fixed queries. Chunked by LIMIT / OFFSET (2M rows); each set's COUNT first; a non-final chunk must have exactly 2M rows and the
chunks must sum to the COUNT (a first version stopped at a silently cut 9.4M-row chunk of 93.9M); resumable.
The Wikimedia API was tried first and answered HTTP 429 / cirrussearch-too-busy at ~20 s per name (log 2026-11-12).

usage: python demo_wdlabels.py OUT_DIR
-> OUT_DIR/{all_<set>_<k>.tsv.gz, brand_<Q>.tsv.gz, brand.tsv.gz, export.json (row counts, sha256, time)}
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

URL = "https://qlever.dev/api/wikidata"
UA = "t24-demo-namecheck/0.2 (fictional name collision check)"
CHUNK = 2_000_000
PAUSE = 20                                                  # s between requests (QLever answered 429)
BRAND_PAUSE = 120                                           # s between the per-class brand queries
PFX = ("PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX skos: <http://www.w3.org/2004/02/skos/core#> "
       "PREFIX wdt: <http://www.wikidata.org/prop/direct/> PREFIX wd: <http://www.wikidata.org/entity/> ")
SETS = {"en_label": "?i @en@rdfs:label ?l", "mul_label": "?i @mul@rdfs:label ?l",
        "en_alias": "?i @en@skos:altLabel ?l", "mul_alias": "?i @mul@skos:altLabel ?l"}
BRAND_CLASSES = ["Q4830453", "Q891723", "Q6881511", "Q431289", "Q783794", "Q167037", "Q5621421", "Q658255",
                 "Q778575", "Q20631656", "Q1589009"]


def count(pat: str) -> int:
    q = PFX + f"SELECT (COUNT(?l) AS ?n) WHERE {{ {pat} }}"
    req = urllib.request.Request(URL + "?" + urllib.parse.urlencode({"query": q}),
                                 headers={"User-Agent": UA, "Accept": "application/sparql-results+json"})
    for attempt in range(8):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=300))
            return int(r["results"]["bindings"][0]["n"]["value"])
        except Exception as e:
            print(time.strftime("%H:%M:%S"), "retry count", attempt, repr(e)[:150], flush=True)
            time.sleep(60 * (attempt + 1))
    raise RuntimeError("count failed")


def fetch(query: str, out: Path, expect: int | None = None) -> int:
    body = urllib.parse.urlencode({"query": PFX + query}).encode()
    req = urllib.request.Request(URL, data=body, headers={"User-Agent": UA, "Accept": "text/tab-separated-values",
                                                          "Content-Type": "application/x-www-form-urlencoded"})
    for attempt in range(8):
        try:
            n = 0
            tmp = out.with_suffix(".part")
            with urllib.request.urlopen(req, timeout=900) as r, gzip.open(tmp, "wt", encoding="utf-8") as g:
                head = r.readline()
                if not head.startswith(b"?"):
                    raise RuntimeError(head[:200])
                for line in r:
                    g.write(line.decode("utf-8")); n += 1
            if expect is not None and n != expect:                 # stream cut short: retry
                raise RuntimeError(f"rows {n} != expected {expect}")
            tmp.rename(out)
            time.sleep(PAUSE)
            return n
        except Exception as e:
            print(time.strftime("%H:%M:%S"), "retry", attempt, repr(e)[:150], flush=True)
            time.sleep(60 * (attempt + 1))
    raise RuntimeError(f"export failed: {out.name}")


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    meta_f = out / "export.json"
    meta = json.load(open(meta_f)) if meta_f.exists() else {"files": {}}
    meta.setdefault("started", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    totals = meta.setdefault("totals", {})
    for s, pat in SETS.items():
        if s not in totals:
            totals[s] = count(pat)
            json.dump(meta, open(meta_f, "w"), indent=1)
            print(time.strftime("%H:%M:%S"), "count", s, totals[s], flush=True)
            time.sleep(PAUSE)
        for k in range(-(-totals[s] // CHUNK)):
            name = f"all_{s}_{k:03d}.tsv.gz"
            if name in meta["files"]:
                continue
            last = k == -(-totals[s] // CHUNK) - 1
            n = fetch(f"SELECT ?l WHERE {{ {pat} }} LIMIT {CHUNK} OFFSET {k * CHUNK}", out / name,
                      None if last else CHUNK)
            meta["files"][name] = {"rows": n, "sha256": hashlib.sha256((out / name).read_bytes()).hexdigest()}
            json.dump(meta, open(meta_f, "w"), indent=1)
            print(time.strftime("%H:%M:%S"), name, n, flush=True)
        got = sum(v["rows"] for f, v in meta["files"].items() if f.startswith(f"all_{s}_"))
        meta.setdefault("drift", {})[s] = got - totals[s]           # live endpoint: the index changes during export
        json.dump(meta, open(meta_f, "w"), indent=1)
        if abs(got - totals[s]) > 1e-5 * totals[s]:
            raise RuntimeError(f"{s}: rows {got} != count {totals[s]}")
    # brand set: one query per class (owner decision 2026-11-12: the single VALUES query got HTTP 429 for 2 h),
    # BRAND_PAUSE s between requests; per-class files kept (resumable), then merged with duplicates removed.
    # Same rows as the single SELECT DISTINCT query: union over classes of DISTINCT (?i, ?l), deduplicated here.
    if "brand.tsv.gz" not in meta["files"]:
        for q in BRAND_CLASSES:
            name = f"brand_{q}.tsv.gz"
            if name in meta["files"]:
                continue
            n = fetch(f"SELECT DISTINCT ?i ?l WHERE {{ ?i wdt:P31 wd:{q} . "
                      f"{{ ?i rdfs:label ?l }} UNION {{ ?i skos:altLabel ?l }} "
                      f"FILTER(LANG(?l) = \"en\" || LANG(?l) = \"mul\") }}", out / name)
            meta["files"][name] = {"rows": n, "sha256": hashlib.sha256((out / name).read_bytes()).hexdigest()}
            json.dump(meta, open(meta_f, "w"), indent=1)
            print(time.strftime("%H:%M:%S"), name, n, flush=True)
            time.sleep(BRAND_PAUSE)
        rows: set[str] = set()
        for q in BRAND_CLASSES:
            with gzip.open(out / f"brand_{q}.tsv.gz", "rt", encoding="utf-8") as g:
                rows.update(line for line in g)
        with gzip.open(out / "brand.tsv.gz", "wt", encoding="utf-8") as g:
            g.writelines(sorted(rows))
        n = len(rows)
        meta["files"]["brand.tsv.gz"] = {"rows": n, "per_class_sum": sum(meta["files"][f"brand_{q}.tsv.gz"]["rows"]
                                                                          for q in BRAND_CLASSES),
                                         "method": "per-class queries, merged, duplicates removed",
                                         "sha256": hashlib.sha256((out / "brand.tsv.gz").read_bytes()).hexdigest()}
        json.dump(meta, open(meta_f, "w"), indent=1)
        print(time.strftime("%H:%M:%S"), "brand.tsv.gz", n, flush=True)
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(meta, open(meta_f, "w"), indent=1)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
