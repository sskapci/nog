"""Country-equivalence table (post hoc, decisions 2026-10-26 §2; rules: reports/experiment_log.md,
"country-equivalence table rules"). Sources: 2Wiki train questions and Simple Wikipedia redirects only.

usage: country_table.py DATA_DIR EXPERT_DIR PAGE_SQL_GZ REDIRECT_SQL_GZ
"""
import gzip, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

from mxn.expert.ops import normalize

data, ex, page_gz, redir_gz = [Path(a) for a in sys.argv[1:5]]
RELS = {"country of citizenship", "country", "country of origin"}
ids = set()
for f in ("fit_ids.txt", "mdev_ids.txt", "tune_ids.txt"):
    ids |= set(open(ex / f).read().split())
train = [r for r in json.load(open(data / "train.json")) if r["_id"] in ids]
print(f"{len(train)} train questions", flush=True)

vocab = Counter(normalize(o) for r in train for s, rel, o in r["evidences"] if rel in RELS)
vocab.pop("", None)
print(f"vocabulary {len(vocab)} strings", flush=True)


def compared(r):
    """The two compared single values of a yes/no question on a country relation, else None."""
    by = defaultdict(set)
    for s, rel, o in r["evidences"]:
        if rel in RELS:
            by[s].add(normalize(o))
    if len(by) != 2 or any(len(v) != 1 for v in by.values()):
        return None
    a, b = (next(iter(v)) for v in by.values())
    return (a, b) if a != b else None


yes, no = Counter(), Counter()
for r in train:
    if r["type"] in ("comparison", "bridge_comparison") and r["answer"].lower() in ("yes", "no"):
        p = compared(r)
        if p:
            (yes if r["answer"].lower() == "yes" else no)[frozenset(p)] += 1
A = [p for p in yes if p not in no]
print(f"type A: {len(yes)} yes pairs, {len(A)} kept ({len(yes) - len(A)} also seen as no)", flush=True)


def tuples(path, pat):
    rx = re.compile(pat)
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:                    # values follow "INSERT INTO ... VALUES" on their own lines
            if not line.startswith(("--", "/*", "CREATE", "  ", ")", "DROP", "LOCK", "UNLOCK", "SET")):
                yield from rx.findall(line)


unesc = lambda s: re.sub(r"\\(.)", r"\1", s).replace("_", " ")
pages, is_redirect = {}, {}
for pid, ns, title, red in tuples(page_gz, r"\((\d+),(\d+),'((?:[^'\\]|\\.)*)',(\d)"):
    if ns == "0":
        pages[int(pid)] = unesc(title); is_redirect[int(pid)] = red == "1"
target = {}
for pid, ns, title in tuples(redir_gz, r"\((\d+),(\d+),'((?:[^'\\]|\\.)*)'"):
    if ns == "0" and int(pid) in pages:
        target[normalize(pages[int(pid)])] = normalize(unesc(title))
print(f"simplewiki: {len(pages)} main pages, {len(target)} redirects", flush=True)
B = defaultdict(set)                      # page → vocabulary strings that are it or redirect to it
for s in vocab:
    B[target.get(s, s)].add(s)
B_edges = [(min(g), x) for g in B.values() if len(g) > 1 for x in sorted(g) if x != min(g)]
print(f"type B: {len(B_edges)} edges", flush=True)

parent = {}


def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x


for a, b in [tuple(p) for p in A] + B_edges:
    parent[find(a)] = find(b)
comp = defaultdict(set)
for x in list(parent):
    comp[find(x)].add(x)
groups = sorted((sorted(g) for g in comp.values() if len(g) > 1), key=len, reverse=True)
table = {x: i for i, g in enumerate(groups) for x in g}
conflicts = [sorted(p) for p in no if len(p) == 2 and all(x in table for x in p) and len({table[x] for x in p}) == 1]
res = {"train_questions": len(train), "vocabulary": len(vocab), "yes_pairs": len(yes), "edges_A": len(A),
       "edges_B": len(B_edges), "components": len(groups), "strings_in_table": len(table), "conflicts": len(conflicts),
       "conflict_pairs": conflicts[:50], "largest": groups[:15]}
print(json.dumps({k: v for k, v in res.items() if k not in ("largest", "conflict_pairs")}))
for g in groups[:15]:
    print(len(g), g[:25])
json.dump({"table": table, "groups": groups}, open(ex / "country_table.json", "w"), indent=1)
json.dump(res, open(ex / "country_table_report.json", "w"), indent=1)
