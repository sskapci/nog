"""MetaQA scope for set answers and person heads (design input; user priority 2, 2026-11-06). Analysis only.

Template path, found in the KB over at most PER_TPL train questions per template (any head, any answer count): a path
is a sequence of steps, "f:r" (forward: movie -> its values for r) or "r:r" (reverse: value -> movies listing it for
r), alternating from the head's kind (movie head starts forward, value head starts reverse); the head is excluded after
every step (as the reverse hop of design_reverse_lookup.md, which excludes the head movie before hop 3). A question matches a path when the path's result equals its gold set (normalized). Template path = the
path matching the most labelled questions if it matches >= 90% of those with any match and is unique at the top
(as metaqa_paths.py, generalized to value heads and set answers).

Per split and hop count, each question falls in the first category that fits:
  no_path          template without a path
  now              movie head naming exactly one document, one gold answer, at most one reverse step (current scope)
  sets             as now but several gold answers
  person           value (person, year, ...) head, one gold answer, exactly one reverse step
  person_sets      as person but several gold answers
  multi_reverse    two or more reverse steps (priority 4)
  ambiguous_head   movie head naming several documents (or none)
Ceilings on dev only: exact set match using only triples stated as slots in the movie's own document (build_slots).
usage: python metaqa_scope.py WIKIMOVIES_DIR METAQA_DIR OUT.json
"""
import collections, json, re, sys
from pathlib import Path
sys.path.insert(0, __import__("os").path.abspath(__file__).rsplit("/experiments/", 1)[0])
from mxn.neuron.slots import build_slots  # noqa: E402

wm, mq, outp = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
PER_TPL, MAJ, CAP = 20, 0.90, 3000
norm = lambda s: re.sub(r"\s+", " ", s.lower()).strip()
strip = lambda t: re.sub(r"\s*\([^)]*\)$", "", t)

titles, sents, cur = [], [], None
for line in open(wm / "knowledge_source" / "wiki.txt", encoding="utf-8"):
    line = line.rstrip("\n")
    if not line.strip():
        cur = None; continue
    n, _, text = line.partition(" ")
    if n == "1":
        titles.append(text); sents.append([]); cur = sents[-1]
    elif cur is not None and text.strip():
        cur.append(text.strip())
by_name = collections.defaultdict(list)
for k, t in enumerate(titles):
    by_name[norm(strip(t))].append(k)

fwd = collections.defaultdict(lambda: collections.defaultdict(set))
rev = collections.defaultdict(lambda: collections.defaultdict(set))
kb = [l.rstrip("\n").split("|") for l in open(mq / "kb" / "kb.txt", encoding="utf-8") if l.count("|") == 2]
for h, r, v in kb:
    fwd[norm(h)][r].add(norm(v)); rev[norm(v)][r].add(norm(h))
RELS = sorted({r for _, r, _ in kb})


def step(F, R, S, s):
    d, r = s.split(":")
    T = F if d == "f" else R
    return set().union(*[T[x].get(r, set()) for x in S]) if S else set()


def run(F, R, head, path):
    S = {head}
    for s in path:
        S = step(F, R, S, s) - {head}
        if len(S) > CAP:
            return None
    return S


def paths_for(head, gold, n):
    out = []

    def go(S, path):
        if len(path) == n:
            if S == gold:
                out.append(tuple(path))
            return
        d = ("f" if head in fwd else "r") if not path else ("r" if path[-1][0] == "f" else "f")
        for r in RELS:
            T = step(fwd, rev, S, f"{d}:{r}") - {head}
            if T and len(T) <= CAP:
                go(T, path + [f"{d}:{r}"])
    for start in (["f"] if head in fwd else []) + (["r"] if head in rev else []):
        for r in RELS:
            T = step(fwd, rev, {head}, f"{start}:{r}") - {head}
            if T and len(T) <= CAP:
                go(T, [f"{start}:{r}"])
    return out


def read(hop, split):
    for k, line in enumerate(open(mq / hop / "vanilla" / f"qa_{split}.txt", encoding="utf-8")):
        q, _, a = line.rstrip("\n").partition("\t")
        m = re.search(r"\[([^\]]+)\]", q)
        yield k, norm(m.group(1)), q[:m.start()] + "X" + q[m.end():], {norm(x) for x in a.split("|") if x}


tpl_path = {}
for hop, n in (("1-hop", 1), ("2-hop", 2), ("3-hop", 3)):
    lab = collections.defaultdict(collections.Counter)
    for k, head, t, gold in read(hop, "train"):
        if lab[t]["_q"] >= PER_TPL:
            continue
        ps = paths_for(head, gold, n)
        lab[t]["_q"] += 1; lab[t]["_n"] += bool(ps)
        for p in set(ps):
            lab[t][json.dumps(p)] += 1
    tp = {}
    for t, c in lab.items():
        lc = sorted(((v, k) for k, v in c.items() if k[0] == "["), reverse=True)
        if lc and lc[0][0] >= MAJ * c["_n"] and (len(lc) == 1 or lc[1][0] < lc[0][0]):
            tp[t] = json.loads(lc[0][1])
    tpl_path[hop] = tp
    print(hop, "templates", len(lab), "with path", len(tp), flush=True)

# stated triples (dev ceilings): value is a slot of the movie's own document; movie named by exactly one document
sf = collections.defaultdict(lambda: collections.defaultdict(set)); sr = collections.defaultdict(lambda: collections.defaultdict(set))
slot_cache = {}
for h, r, v in kb:
    h, v = norm(h), norm(v)
    ks = by_name.get(h, [])
    if len(ks) != 1:
        continue
    if h not in slot_cache:
        slot_cache[h] = {norm(s.label) for s in build_slots(titles[ks[0]], sents[ks[0]])}
    if v in slot_cache[h]:
        sf[h][r].add(v); sr[v][r].add(h)
print("stated triples built", flush=True)

out = {"template_paths": {h: {t: p for t, p in tp.items()} for h, tp in tpl_path.items()}, "splits": {}}
CATS = ("now", "sets", "person", "person_sets", "multi_reverse", "ambiguous_head", "no_path")
for hop in ("1-hop", "2-hop", "3-hop"):
    for split in ("train", "dev", "test"):
        C = collections.Counter(); ceil = collections.Counter(); shapes = collections.Counter()
        for k, head, t, gold in read(hop, split):
            C["total"] += 1
            p = tpl_path[hop].get(t)
            if p is None:
                C["no_path"] += 1; continue
            nrev = sum(s[0] == "r" for s in p); movie_head = p[0][0] == "f"
            if nrev >= 2:
                cat = "multi_reverse"
            elif movie_head and len(by_name.get(head, [])) != 1:
                cat = "ambiguous_head"
            elif movie_head:
                cat = "now" if len(gold) == 1 else "sets"
            else:
                cat = "person" if len(gold) == 1 else "person_sets"
            C[cat] += 1; shapes[f"{cat} {' '.join(s[0] for s in p)}"] += 1
            if split == "dev" and cat in CATS[:4]:
                ceil[f"{cat}_n"] += 1
                ceil[cat] += run(sf, sr, head, p) == gold
        res = {"total": C["total"], **{c: C[c] for c in CATS},
               **{f"{c}_pct": round(100 * C[c] / C["total"], 2) for c in CATS}, "shapes": dict(shapes)}
        if split == "dev":
            res["stated_ceiling_pct"] = {c: round(100 * ceil[c] / ceil[f"{c}_n"], 2) if ceil[f"{c}_n"] else None
                                         for c in CATS[:4]}
        out["splits"][f"{hop}/{split}"] = res
        print(hop, split, {c: res[f"{c}_pct"] for c in CATS}, res.get("stated_ceiling_pct", ""), flush=True)
json.dump(out, open(outp, "w"), indent=1)
