"""MetaQA 2- / 3-hop movie-head questions (train split only): relation path per question template, found in the KB.

Shape (only movies have KUs): hop 1 forward head -r1-> V; hop 2 reverse: movies M with (M, r2, v) for v in V;
3-hop: hop 3 forward M -r3-> answers. A question's candidate paths are the (r1, r2[, r3]) whose result set equals the
gold answer set, with the head movie excluded from M (first labelling: every template with a unique path used "excl",
so only "excl" is kept). Template path = the path that matches the most of its labelled train questions, if it
matches >= 90% of those with any matching path (a question can match several paths, e.g. director = writer); at
most PER_TPL questions per template are labelled (cost).
usage: python metaqa_paths.py METAQA_DIR OUT.json
"""
import collections, json, re, sys
from pathlib import Path

mq, outp = Path(sys.argv[1]), Path(sys.argv[2])
norm = lambda s: re.sub(r"\s+", " ", s.lower()).strip()
fwd = collections.defaultdict(lambda: collections.defaultdict(set))   # movie -> rel -> values
rev = collections.defaultdict(lambda: collections.defaultdict(set))   # value -> rel -> movies
movies = set()
for l in open(mq / "kb" / "kb.txt", encoding="utf-8"):
    p = l.rstrip("\n").split("|")
    if len(p) != 3:
        continue
    h, r, v = norm(p[0]), p[1], norm(p[2])
    fwd[h][r].add(v); rev[v][r].add(h); movies.add(h)
RELS = sorted({r for m in fwd.values() for r in m})
REV = ["directed_by", "written_by", "starred_actors", "release_year", "in_language", "has_genre", "has_tags"]


PER_TPL = 20


def paths_for(head, gold, n, excl):
    out = []
    for r1, V in fwd[head].items():
        for r2 in REV:
            M = set().union(*[rev[v].get(r2, set()) for v in V])
            if excl:
                M = M - {head}
            if n == 2:
                if M == gold:
                    out.append((r1, r2))
                continue
            if not M or len(M) > 2000:
                continue
            for r3 in RELS:
                if set().union(*[fwd[m].get(r3, set()) for m in M]) == gold:
                    out.append((r1, r2, r3))
    return out


out = {}
for hop, n in (("2-hop", 2), ("3-hop", 3)):
    lab = collections.defaultdict(collections.Counter); var = collections.Counter(); tot = 0
    for line in open(mq / hop / "vanilla" / "qa_train.txt", encoding="utf-8"):
        q, _, a = line.rstrip("\n").partition("\t")
        m = re.search(r"\[([^\]]+)\]", q); head = norm(m.group(1))
        if head not in movies:
            continue
        gold = {norm(x) for x in a.split("|") if x}
        tpl = q[:m.start()] + "X" + q[m.end():]
        if lab[tpl]["_q"] >= PER_TPL:
            continue
        tot += 1
        ps = paths_for(head, gold, n, True)
        lab[tpl]["_q"] += 1; lab[tpl]["_n"] += bool(ps)
        for p_ in ps:
            lab[tpl][json.dumps(p_)] += 1
        var["any_path"] += bool(ps)
    tp, dropped = {}, 0
    for t, c in lab.items():
        lc = sorted(((v, k) for k, v in c.items() if k not in ("_n", "_q")), reverse=True)
        if lc and c["_n"] and lc[0][0] >= 0.9 * c["_n"] and (len(lc) == 1 or lc[1][0] < lc[0][0]):
            tp[t] = lc[0][1]
        else:
            dropped += 1
    out[hop] = {"questions_labelled": tot, "unique_path_by_variant": dict(var), "templates": len(lab),
                "templates_with_path": len(tp), "templates_dropped": dropped, "template_path": tp,
                "path_counts": collections.Counter(tp.values()).most_common()}
    print(hop, tot, dict(var), len(lab), len(tp), dropped)
json.dump(out, open(outp, "w"), indent=1)
