"""Demo corpus A: fictional companies (reports/design_demo_corpus.md; owner decision D3, 2026-11-09).

Fictional companies (synthetic data generated for this demo). Any resemblance to real companies or people is
unintended.

Seeded generator: syllable names (companies, people, cities, products; industries from a fixed generic list), a
knowledge base over the eight relations of mxn/neuron/vocab/biz.json, one templated document per company (one KU per
company), and questions (1-hop forward / reverse, 2-hop) with gold answer sets. Names listed in --drop (Wikidata label
hits, experiments/dataset/demo_namecheck.py) are never drawn; a drop changes later draws, deterministically.
Checked pool (owner decision 2026-11-13, replaces regenerate-until-clean: a drop reshuffled later draws, so most names
changed each round and the check could not converge): --make-pool writes a candidate pool per kind (FACTOR x the
corpus's needs; products at FACTOR x 3 per company, the maximum) from its own seed; the name check runs once over it;
--pool then draws every name, in pool order, from the pool minus --drop, and the corpus rng no longer draws names.
The run fails if a kind's pool runs out, and checks that every corpus name is in the pool and not dropped.
10% of parent links (seeded) are stated only in the child's document: the parent's document omits that subsidiary,
so the stated ceiling is below 100% (fact flag "own": the value is in the head company's own document).
Splits: companies train / dev / test 60 / 20 / 20 (seeded); a question goes to its head's split (value heads: split by
seeded hash of the value); test questions use only the held-out wording of each relation (ask[r][-1]).

usage: python demo_gen.py OUT_DIR [--drop=hits.json] [--companies=3000] [--pool=pool.json]
       python demo_gen.py --make-pool=pool.json [--factor=3] [--companies=3000]
-> OUT_DIR/{kb.json, pool.pkl (titles, sentences, owner, questions), names.json, gen.json, NOTICE.txt}
"""
from __future__ import annotations

import hashlib
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np

SEED = 20261112
NOTICE = ("Fictional companies (synthetic data generated for this demo). "
          "Any resemblance to real companies or people is unintended.")
VOCAB = json.load(open(Path(__file__).resolve().parents[2] / "mxn" / "neuron" / "vocab" / "biz.json"))
INDUSTRIES = ["Logistics", "Biotechnology", "Agriculture", "Aerospace", "Insurance", "Retail Banking", "Robotics",
              "Shipbuilding", "Textiles", "Mining", "Pharmaceuticals", "Renewable Energy", "Telecommunications",
              "Food Processing", "Semiconductors", "Construction", "Publishing", "Software", "Packaging",
              "Water Treatment", "Medical Devices", "Furniture", "Chemicals", "Rail Transport", "Cosmetics",
              "Video Games", "Steel", "Tourism", "Consumer Electronics", "Cybersecurity"]
ON = ["b", "br", "c", "d", "dr", "f", "g", "gr", "h", "k", "kl", "l", "m", "n", "p", "pr", "r", "s", "st", "t", "tr",
      "v", "z", "sh", "th", ""]
VO = ["a", "e", "i", "o", "u", "ae", "ei", "ou", "y"]
CO = ["", "", "", "n", "r", "l", "s", "x", "m", "nd", "rk", "lt", "st"]
CO_SUFFIX = ["", "", "", "Systems", "Labs", "Group", "Works", "Dynamics", "Holdings", "Industries", "Partners"]
CITY_SUFFIX = ["", "", "haven", "burg", "ford", "mouth", "dale", "port", "stad", "field"]
LINE = ["", "", "Pro", "One", "Max", "Lite", "Core", "Air", "Flex", "Prime"]


class Names:
    def __init__(self, rng: random.Random, drop: set[str]):
        self.rng, self.drop, self.used = rng, {d.lower() for d in drop}, {i.lower() for i in INDUSTRIES}

    def word(self, lo: int, hi: int) -> str:
        r = self.rng
        return "".join(r.choice(ON) + r.choice(VO) + r.choice(CO) for _ in range(r.randint(lo, hi))).capitalize()

    def take(self, make) -> str:
        while True:
            n = " ".join(make().split())
            if len(n) >= 4 and n.lower() not in self.used and n.lower() not in self.drop:
                self.used.add(n.lower())
                return n

    def company(self):
        return self.take(lambda: f"{self.word(2, 3)} {self.rng.choice(CO_SUFFIX)}")

    def person(self):
        return self.take(lambda: f"{self.word(2, 2)} {self.word(2, 3)}")

    def city(self):
        return self.take(lambda: self.word(1, 2) + self.rng.choice(CITY_SUFFIX))

    def product(self):
        return self.take(lambda: f"{self.word(2, 2)} {self.rng.choice(LINE)}")


class PoolNames:
    """Names drawn in order from a checked pool (kind -> list), skipping dropped and used names."""
    KIND = {"company": "companies", "person": "people", "city": "cities", "product": "products"}

    def __init__(self, pool: dict[str, list[str]], drop: set[str]):
        drop_l = {d.lower() for d in drop}
        self.left = {k: iter([n for n in v if n.lower() not in drop_l]) for k, v in pool.items()}
        self.used = {i.lower() for i in INDUSTRIES}

    def take(self, kind: str) -> str:
        for n in self.left[kind]:
            if n.lower() not in self.used:
                self.used.add(n.lower())
                return n
        raise RuntimeError(f"checked pool for {kind} ran out")

    def __getattr__(self, m: str):
        if m in self.KIND:
            return lambda: self.take(self.KIND[m])
        raise AttributeError(m)


POOL_SEED = SEED + 10


def make_pool(n_comp: int, factor: int) -> dict[str, list[str]]:
    """Candidate names per kind, unique across kinds; own seed, so the corpus draws do not move the pool."""
    nm = Names(random.Random(POOL_SEED), set())
    need = {"companies": n_comp, "people": n_comp // 2, "cities": 200, "products": 3 * n_comp}
    make = {"companies": nm.company, "people": nm.person, "cities": nm.city, "products": nm.product}
    return {k: [make[k]() for _ in range(factor * need[k])] for k in need}


def listing(xs: list[str]) -> str:
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def kb_of(n_comp: int, drop: set[str], pool: dict | None = None) -> dict:
    rng = random.Random(SEED)
    nm = PoolNames(pool, drop) if pool is not None else Names(rng, drop)
    comps = [nm.company() for _ in range(n_comp)]
    people = [nm.person() for _ in range(n_comp // 2)]
    cities = [nm.city() for _ in range(200)]
    kb = {c: {r: [] for r in VOCAB["relations"]} for c in comps}
    own = {}                                                   # (company, rel, value) -> stated in own document
    top = set(rng.sample(comps, n_comp // 5))                  # parents come only from these: no cycles
    for c in comps:
        f = kb[c]
        f["founded_by"] = rng.sample(people, rng.choice([1, 1, 2, 2, 3]))
        f["ceo"] = [rng.choice(f["founded_by"]) if rng.random() < 0.3 else rng.choice(people)]
        f["headquartered_in"] = [rng.choice(cities)]
        f["industry"] = [rng.choice(INDUSTRIES)]
        f["product"] = [nm.product() for _ in range(rng.choice([1, 2, 2, 3]))]
        f["founded_year"] = [str(rng.randint(1890, 2022))]
        if c not in top and rng.random() < 0.375:              # ~30% of all companies have a parent
            p = rng.choice(sorted(top))
            f["parent"] = [p]
            kb[p]["subsidiary"].append(c)
            own[(p, "subsidiary", c)] = rng.random() >= 0.10   # 10%: stated only in the child's document
    return {"companies": comps, "people": people, "cities": cities, "kb": kb, "own": own}


def document(c: str, f: dict, own: dict, rng: random.Random) -> list[str]:
    y, fb = f["founded_year"][0], listing(f["founded_by"])
    s = [rng.choice([f"{c} was founded in {y} by {fb}.", f"{fb} established {c} in {y}.",
                     f"{c} was started by {fb} in {y}."])]
    s.append(rng.choice([f"{c} operates in the {f['industry'][0]} sector.",
                         f"The company is active in the {f['industry'][0]} industry."]))
    s.append(rng.choice([f"It is headquartered in {f['headquartered_in'][0]}.",
                         f"{c} has its head office in {f['headquartered_in'][0]}.",
                         f"The company is based in {f['headquartered_in'][0]}."]))
    s.append(rng.choice([f"{c} is led by chief executive {f['ceo'][0]}.",
                         f"{f['ceo'][0]} runs the company as chief executive.",
                         f"The chief executive of the company is {f['ceo'][0]}."]))
    pr = listing(f["product"])
    s.append(rng.choice([f"Its products include {pr}.", f"{c} makes {pr}.", f"The company sells {pr}."]))
    if f["parent"]:
        s.append(rng.choice([f"{c} is a subsidiary of {f['parent'][0]}.", f"It is part of {f['parent'][0]}.",
                             f"The company is owned by {f['parent'][0]}."]))
    subs = [x for x in f["subsidiary"] if own[(c, "subsidiary", x)]]
    if subs:
        s.append(rng.choice([f"Its subsidiaries are {listing(subs)}.", f"{c} owns {listing(subs)}.",
                             f"The company has acquired {listing(subs)}."]))
    if rng.random() < 0.5:
        s.append(rng.choice([f"The company employs about {rng.randint(20, 950)} people.",
                             "It has offices in several countries.", "The company is privately held."]))
    return s


def split_of(key: str, comp_split: dict) -> str:
    if key in comp_split:
        return comp_split[key]
    h = int(hashlib.sha256(f"{SEED}:{key}".encode()).hexdigest(), 16) % 10
    return "train" if h < 6 else "dev" if h < 8 else "test"


REV = {   # value-head (reverse) 1-hop wordings; last = held out (test only)
    "founded_by": ["Which companies did {V} found?", "What companies were founded by {V}?",
                   "Which firms did {V} start?"],
    "ceo": ["Which company does {V} run?", "Which companies have {V} as chief executive?",
            "Where is {V} the chief executive?"],
    "headquartered_in": ["Which companies are headquartered in {V}?", "What companies are based in {V}?",
                         "Which firms have their head office in {V}?"],
    "product": ["Which company makes {V}?", "Who sells {V}?", "Which firm offers {V}?"],
    "industry": ["Which companies are in the {V} industry?", "What companies operate in the {V} sector?",
                 "Which firms are active in {V}?"],
}
TWO = {   # 2-hop: (r1 forward company->company, r2 forward) and (r1 reverse value->companies, r2 forward)
    ("f:parent", "f:ceo"): ["Who is the chief executive of the parent company of {D}?",
                            "Who runs the company that owns {D}?", "Who leads the owner of {D}?"],
    ("f:parent", "f:headquartered_in"): ["Where is the parent company of {D} headquartered?",
                                         "Where is the owner of {D} based?",
                                         "In which city is the head office of the company that owns {D}?"],
    ("f:subsidiary", "f:product"): ["What products do the subsidiaries of {D} make?",
                                    "What do the companies owned by {D} sell?",
                                    "Which products are offered by companies belonging to {D}?"],
    ("r:founded_by", "f:headquartered_in"): ["Where are the companies founded by {V} headquartered?",
                                             "In which cities are the firms started by {V} based?",
                                             "Where do the companies that {V} founded have their head office?"],
    ("r:headquartered_in", "f:ceo"): ["Who runs the companies headquartered in {V}?",
                                      "Who are the chief executives of companies based in {V}?",
                                      "Who leads the firms with a head office in {V}?"],
}


def questions(G: dict, comp_split: dict) -> list[dict]:
    kb, rels = G["kb"], VOCAB["relations"]
    inv = {r: {} for r in rels}
    for c, f in kb.items():
        for r in rels:
            for v in f[r]:
                inv[r].setdefault(v, []).append(c)
    rng = random.Random(SEED + 1)
    out = []

    def add(kind: str, head: str, path: str, ans: list[str], wordings: list[str], slot: str):
        sp = split_of(head, comp_split)
        w = wordings[-1] if sp == "test" else rng.choice(wordings[:-1])
        out.append({"id": f"demo-{sp}-{len(out)}", "split": sp, "head": head, "type": f"{kind}:{path}",
                    "question": w.replace(slot, head), "answers": sorted(set(ans))})

    for c, f in kb.items():
        for r in rels:
            if f[r]:
                add("company", c, f"1hop:f:{r}", f[r], VOCAB["ask"][r], "{D}")
        for (a, b), ws in TWO.items():
            if a.startswith("f:") and f[a[2:]]:
                ans = [v for m in f[a[2:]] for v in kb[m][b[2:]]]
                add("company", c, f"2hop:{a}:{b}", ans, ws, "{D}")
    for r, ws in REV.items():
        for v, cs in inv[r].items():
            add("value", v, f"1hop:r:{r}", cs, ws, "{V}")
    for (a, b), ws in TWO.items():
        if a.startswith("r:"):
            for v, cs in inv[a[2:]].items():
                add("value", v, f"2hop:{a}:{b}", [x for m in cs for x in kb[m][b[2:]]], ws, "{V}")
    return out


def main(out: Path, drop: set[str], n_comp: int, pool_f: str | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pool = json.load(open(pool_f)) if pool_f else None
    G = kb_of(n_comp, drop, pool)
    rng = random.Random(SEED + 2)
    order = G["companies"][:]
    rng.shuffle(order)
    comp_split = {c: "train" if i < 0.6 * n_comp else "dev" if i < 0.8 * n_comp else "test" for i, c in enumerate(order)}
    rng = random.Random(SEED + 3)
    titles, sents, owner = [], [], []
    for k, c in enumerate(G["companies"]):
        titles.append(c)
        for s in document(c, G["kb"][c], G["own"], rng):
            sents.append(s); owner.append(k)
    qs = questions(G, comp_split)
    pickle.dump({"titles": titles, "sentences": sents, "owner": np.array(owner), "questions": qs},
                open(out / "pool.pkl", "wb"))
    json.dump({"notice": NOTICE, "kb": G["kb"], "split": comp_split,
               "not_in_own_doc": [list(k) for k, v in G["own"].items() if not v]}, open(out / "kb.json", "w"))
    names = {"companies": G["companies"], "people": G["people"], "cities": G["cities"],
             "products": sorted({p for f in G["kb"].values() for p in f["product"]})}
    json.dump(names, open(out / "names.json", "w"))
    check = None
    if pool is not None:                                       # every corpus name is a checked, not dropped, pool name
        drop_l = {d.lower() for d in drop}
        bad = {k: [n for n in v if n not in set(pool[k]) or n.lower() in drop_l] for k, v in names.items()}
        check = {"pool_sizes": {k: len(v) for k, v in pool.items()},
                 "pool_passed": {k: sum(1 for n in v if n.lower() not in drop_l) for k, v in pool.items()},
                 "drawn": {k: len(v) for k, v in names.items()}, "not_checked_or_dropped": {k: len(v) for k, v in bad.items()},
                 "pool_sha256": hashlib.sha256(Path(pool_f).read_bytes()).hexdigest()}
        if any(bad.values()):
            raise RuntimeError(f"corpus names outside the checked pool or dropped: {check}")
    (out / "NOTICE.txt").write_text(NOTICE + "\n")
    cnt = {}
    for q in qs:
        k = (q["split"], q["type"].split(":")[1])
        cnt[f"{k[0]}:{k[1]}"] = cnt.get(f"{k[0]}:{k[1]}", 0) + 1
    gen = {"notice": NOTICE, "seed": SEED, "companies": n_comp, "documents": len(titles), "sentences": len(sents),
           "people": len(G["people"]), "cities": len(G["cities"]), "products": len(names["products"]),
           "industries": len(INDUSTRIES), "parent_links": sum(1 for f in G["kb"].values() if f["parent"]),
           "not_in_own_doc": sum(1 for v in G["own"].values() if not v), "dropped_names": len(drop),
           "questions": len(qs), "questions_by_split_hop": dict(sorted(cnt.items())), "name_pool": check,
           "sha256_pool": hashlib.sha256((out / "pool.pkl").read_bytes()).hexdigest()}
    json.dump(gen, open(out / "gen.json", "w"), indent=1)
    print(json.dumps(gen, indent=1))


if __name__ == "__main__":
    opts = dict((a[2:].split("=", 1) + [""])[:2] for a in sys.argv if a.startswith("--"))
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if opts.get("make-pool"):
        P = make_pool(int(opts.get("companies", 3000)), int(opts.get("factor", 3)))
        json.dump(P, open(opts["make-pool"], "w"), indent=0)
        print({k: len(v) for k, v in P.items()})
        sys.exit(0)
    drop = set(json.load(open(opts["drop"]))) if opts.get("drop") else set()
    main(Path(args[0]), drop, int(opts.get("companies", 3000)), opts.get("pool"))
