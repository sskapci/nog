"""DRAFT (decision 2026-10-27 §4): hand-written paraphrase grammar for planner training-data augmentation.

Written from the TRAIN question forms only (plans2_fit: 169 plan signatures, 2–3 wordings each). Given a plan
(template, r1, r2, op) and the entity strings, it writes a new question with known entity spans; the plan label is
unchanged by construction. Not used by any run yet.

Coverage: every relation and op of the plan language; relations without a phrase bank fall back to
"the <relation> of X". The wordings are deliberately different from 2Wiki's own templates (which stay in the
training data as they are).
"""

from __future__ import annotations

import random

# what kind of thing E is, from the first relation (2Wiki's "film" / "song" words); None = a person or place
KIND = {"director": ["film", "movie"], "performer": ["song", "track"], "composer": ["film", "work"],
        "producer": ["film", "work"], "publication date": ["film", "work"], "country of origin": ["film", "work"],
        "editor": ["film"], "creator": ["work"], "presenter": ["show"]}

# noun phrases "the N of X" / "X's N" for person-valued relations
NOUN = {
    "director": ["director", "filmmaker"],
    "performer": ["performer", "singer"],
    "composer": ["composer", "music writer"],
    "producer": ["producer"],
    "father": ["father", "dad"],
    "mother": ["mother", "mom"],
    "spouse": ["spouse", "husband or wife", "partner in marriage"],
    "child": ["child"],
    "sibling": ["sibling", "brother or sister"],
    "founded by": ["founder"],
    "editor": ["editor"],
    "creator": ["creator"],
    "presenter": ["presenter", "host"],
    "employer": ["employer"],
}
# a short verb form "the person who V X" for the first hop of chains
VERB = {"director": "directed", "performer": "performed", "composer": "composed the music for",
        "producer": "produced", "founded by": "founded", "editor": "edited", "creator": "created"}

# question frames for the last relation of a chain, about a description D (a noun phrase)
ASK = {
    "date of birth": ["On what date was {D} born?", "What was the birth date of {D}?", "{D} was born on which date?",
                      "Give the date {D} was born."],
    "date of death": ["On what date did {D} pass away?", "What was the date of death of {D}?",
                      "{D} died on which date?", "When exactly did {D} die?"],
    "place of birth": ["In which place was {D} born?", "What is the birthplace of {D}?", "{D} was born where?",
                       "Where did {D} come into the world?"],
    "place of death": ["In which place did {D} die?", "Where did {D} pass away?", "{D} died where?",
                       "What was the place where {D} died?"],
    "country of citizenship": ["Which country is {D} a citizen of?", "What country does {D} come from?",
                               "{D} holds citizenship of which country?", "Of which nation is {D} a citizen?"],
    "place of burial": ["Where is {D} buried?", "In which place was {D} laid to rest?"],
    "cause of death": ["What did {D} die of?", "What caused the death of {D}?"],
    "educated at": ["Where did {D} study?", "Which school did {D} attend?"],
    "award received": ["Which award did {D} receive?", "What prize was given to {D}?"],
    "employer": ["Who did {D} work for?", "Which organization employed {D}?"],
    "occupation": ["What did {D} do for a living?"],
}
PERSON_ASK = ["Who is {N}?", "Who was {N}?", "Name {N}.", "Can you tell me who {N} is?"]

# comparison words per (relation, op)
CMP = {
    ("date of birth", "earlier"): ["was born earlier", "is the older one", "has the earlier birth date", "came into the world first"],
    ("date of birth", "later"): ["was born later", "is the younger one", "has the later birth date"],
    ("date of death", "earlier"): ["died earlier", "passed away first", "has the earlier date of death"],
    ("date of death", "later"): ["died later", "passed away last", "lived longer into the future"],
    ("publication date", "earlier"): ["was released earlier", "came out sooner", "premiered first", "is the older release"],
    ("publication date", "later"): ["was released later", "came out more recently", "is the newer release"],
    ("inception", "earlier"): ["was founded earlier", "is older", "was established first"],
    ("inception", "later"): ["was founded later", "is newer", "was established more recently"],
}
SAME = {
    "country of citizenship": ["share a nationality", "hold the same citizenship", "come from the same country"],
    "country": ["lie in the same country", "belong to the same country", "share a country"],
    "country of origin": ["come from the same country", "share a country of origin", "originate in the same country"],
}


def _noun(rel: str, rng: random.Random) -> str:
    return rng.choice(NOUN.get(rel, [rel]))


def _desc(rel: str, ent: str, rng: random.Random, spans: list, text_so_far: str) -> tuple[str, list]:
    """A description "the N of [kind] E" / "E's N" / "the person who V E", with E's span relative to the phrase."""
    kind = rng.choice(KIND.get(rel, [None]))
    forms = [("the {n} of " + (kind + " " if kind else ""), ""), ("", "'s {n}")]
    if rel in VERB:
        forms.append((f"the person who {VERB[rel]} " + ("the " + kind + " " if kind else ""), ""))
    pre, post = rng.choice(forms)
    n = _noun(rel, rng)
    pre, post = pre.format(n=n), post.format(n=n)
    return pre + ent + post, [len(pre), len(pre) + len(ent)]


def chain(r1: str, r2: str | None, e1: str, rng: random.Random) -> tuple[str, list]:
    """Chain question with E1's span. r2 None = a one-hop question (not in the plan language; unused)."""
    d, (a, b) = _desc(r1, e1, rng, [], "")
    if r2 in ASK:
        frame = rng.choice(ASK[r2])
    else:
        frame = rng.choice(PERSON_ASK).replace("{N}", "the " + _noun(r2, rng) + " of {D}") if r2 in NOUN else \
            f"What is the {r2} of {{D}}?"
    i = frame.index("{D}")
    if i == 0 and a > 0:                                   # never change the entity's own spelling
        d = d[0].upper() + d[1:]
    q = frame.replace("{D}", d)
    return q, [[i + a, i + b]]


def compare(rel: str, op: str, e1: str, e2: str, rng: random.Random, via: str | None = None) -> tuple[str, list]:
    """Compare (via None) or bridge (via = r1: the comparison is about the r1 of each entity)."""
    if op in ("same", "different"):
        verb = rng.choice(SAME.get(rel, [f"have the same {rel}"]))
        subj = (lambda e: e) if via is None else (lambda e: f"the {_noun(via, rng)} of {e}")
        frames = ["Do {A} and {B} " + verb + "?", "Is it true that {A} and {B} " + verb + "?",
                  "Would you say {A} and {B} " + verb + "?"]
        if op == "different":
            frames = ["Do {A} and {B} not " + verb + "?", "Is it false that {A} and {B} " + verb + "?"]
        A, B = subj("\0"), subj("\1")
    else:
        what = rng.choice(CMP.get((rel, op), [f"has the {op} {rel}"]))
        if via is None:
            A, B = "\0", "\1"
            frames = ["Between {A} and {B}, which " + what + "?", "Of {A} and {B}, which one " + what + "?",
                      "{A} or {B}: which " + what + "?", "Which " + what + ": {A} or {B}?"]
        else:
            n = _noun(via, rng)
            what = what.replace("is the older one", "is older").replace("is the younger one", "is younger")
            A, B = "\0", "\1"
            frames = ["Between {A} and {B}, which one's " + n + " " + what + "?",
                      "Of {A} and {B}, whose " + n + " " + what + "?",
                      "{A} or {B}: which has the " + n + " who " + what + "?"]
    q = rng.choice(frames).replace("{A}", A).replace("{B}", B)
    i = q.index("\0"); q = q.replace("\0", e1, 1)
    j = q.index("\1"); q = q.replace("\1", e2, 1)
    return q, [[i, i + len(e1)], [j, j + len(e2)]]


def paraphrase(plan: dict, ents: list[str], rng: random.Random) -> tuple[str, list] | None:
    """New question text and spans for a training plan; None if the template is unknown."""
    if plan["template"] == "chain":
        return chain(plan["r1"], plan["r2"], ents[0], rng)
    if plan["template"] == "compare":
        return compare(plan["r1"], plan["op"], ents[0], ents[1], rng)
    if plan["template"] == "bridge":
        return compare(plan["r2"], plan["op"], ents[0], ents[1], rng, via=plan["r1"])
    return None
