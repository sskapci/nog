"""The two audits (pre-result amendment 2026-10-29 to the connections rules 22a544e; replaces the single text audit).

1. KU-content audit (gateway and driver process): the process holds no KU instance (KU, network, calibration,
   extractor) and no KU line that is not a title or name line of some KU. Answers, edit values and questions are
   names or inputs: a string equal to a KU title (or a title without its disambiguator) is not counted.
2. Index audit (gateway only): nothing reachable from the gateway object is a title index — no Resolver /
   HostTitleIndex / KUTitle instance, no KU title string, no title-vector matrix, no key table (≥ 1,000 title keys, or string / word-tuple keys → KU ids) and
   no placement table (≥ 1,000 KU ids → host names); no attribute named like one ("resolver", "titles", "placement",
   "TV", "exact", "nodis", "connections", "gold"). The country table of the operations (gateway.equiv) is a fixed
   input of the operations, not an index: it is skipped, and title strings inside it are reported, not counted.
Hash sets are made outside the audited process (experiments/expert/audit_hashes.py, from the hosts' packages).
"""
from __future__ import annotations

import gc
import hashlib
import re

BANNED_ATTR = {"resolver", "titles", "placement", "TV", "exact", "nodis", "connections", "gold"}
_strip = lambda t: re.sub(r"\s*\([^)]*\)$", "", t)


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


def name_lines(kus) -> set[str]:
    """Every KU title and every title without its disambiguator. kus: [(title, sentences)]."""
    return {x for t, _ in kus for x in (t, _strip(t))}


def content_hashes(kus) -> set[str]:
    """sha1 of every KU line that is not a title or name line of some KU (audit 1)."""
    names = name_lines(kus)
    return {sha1(s) for _, ss in kus for s in ss if s not in names}


def title_hashes(kus) -> set[str]:
    return {sha1(x) for x in name_lines(kus)}


def audit_content(content: set[str]) -> dict:
    """Audit 1 over the whole process (every tracked container, as the original rule)."""
    from mxn.expert.extractor import Extractor, ReadOnlyExtractor
    from mxn.ku.store import KU

    gc.collect()
    banned = (KU, Extractor, ReadOnlyExtractor)        # KU-holding classes (walk classes retired to legacy/, cleanup)
    objs, lines = [], set()
    for o in gc.get_objects():
        if isinstance(o, banned):
            objs.append(type(o).__name__)
        elif isinstance(o, (list, tuple, set, frozenset, dict)):
            for x in (o.values() if isinstance(o, dict) else o):
                if isinstance(x, str) and len(x) > 20 and sha1(x) in content:
                    lines.add(sha1(x))
            if isinstance(o, dict):
                for x in o:
                    if isinstance(x, str) and len(x) > 20 and sha1(x) in content:
                        lines.add(sha1(x))
    return {"ku_objects": sorted(set(objs)), "ku_lines_found": len(lines), "clean": not objs and not lines}


def audit_index(gateway, titles: set[str]) -> dict:
    """Audit 2: objects reachable from the gateway object (its client included), the country table skipped."""
    import types

    import numpy as np

    from mxn.expert.hostindex import HostTitleIndex, KUTitle
    from mxn.expert.resolve import Resolver

    try:
        import torch
        skip_types = (torch.Tensor, torch.nn.Parameter)
    except ImportError:                                         # pragma: no cover
        skip_types = ()
    hosts = set(getattr(getattr(gateway, "client", None), "hosts", {}) or {})
    equiv = getattr(gateway, "equiv", None)
    seen, stack = {id(equiv)}, [gateway]
    idx, found, tv, keys, place = [], set(), [], [], []
    while stack:
        o = stack.pop()
        if id(o) in seen:
            continue
        seen.add(id(o))
        if isinstance(o, (types.ModuleType, type, types.FunctionType, types.BuiltinFunctionType)) or isinstance(o, skip_types):
            continue
        if isinstance(o, (Resolver, HostTitleIndex, KUTitle)):
            idx.append(type(o).__name__)
        if isinstance(o, str):
            if sha1(o) in titles:
                found.add(sha1(o))
            continue
        if isinstance(o, np.ndarray):
            if o.ndim == 2 and o.shape[0] >= 100 and o.shape[1] in (128, 256, 384, 512, 768):
                tv.append(list(o.shape))
            continue
        if isinstance(o, dict) and len(o) >= 1000:
            ks = list(o)[:2000]
            vs0 = list(o.values())[:2000]
            ids = lambda v: isinstance(v, int) or (isinstance(v, (list, tuple, set)) and v and all(isinstance(x, int) for x in v))
            if (sum(isinstance(k, str) and sha1(k) in titles for k in ks) >= len(ks) // 2
                    or (all(isinstance(k, (str, tuple)) for k in ks) and sum(ids(v) for v in vs0) >= len(vs0) // 2)):
                keys.append(len(o))                         # title keys, or string / word-tuple keys → KU ids
            vs = list(o.values())[:2000]
            if hosts and sum((isinstance(v, str) and v in hosts) or (isinstance(v, (list, tuple)) and any(x in hosts for x in v if isinstance(x, str))) for v in vs) >= len(vs) // 2:
                place.append(len(o))
        stack.extend(gc.get_referents(o))
    cl = getattr(gateway, "client", None)
    attrs = set(vars(gateway)) | (set(vars(cl)) if hasattr(cl, "__dict__") else set())
    eq_titles = 0
    if isinstance(equiv, dict):
        eq_titles = sum(1 for k, v in equiv.items() for x in [k, *(v if isinstance(v, (list, tuple)) else [v])]
                        if isinstance(x, str) and sha1(x) in titles)
    r = {"index_objects": sorted(set(idx)), "titles_found": len(found), "title_vector_matrices": tv, "key_tables": keys,
         "placement_tables": place, "banned_attributes": sorted(BANNED_ATTR & attrs), "objects_scanned": len(seen),
         "country_table_title_strings_skipped": eq_titles}
    r["clean"] = not (idx or found or tv or keys or place or r["banned_attributes"])
    return r
