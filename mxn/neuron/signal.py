"""Signal format shared by the gateway and every KU (allowed as shared: ruling N2).

RELATIONS: the planner's relation labels (the 34 relations of plans2_fit, as experiments/expert/planner_run.py
derives them). CUES: words that name a relation in English: the relation's own words plus the planner paraphrase
grammar's nouns and verbs (mxn/expert/para_grammar.py NOUN / VERB) and their plain inflections. They are language
(how a relation is said), not knowledge about any entity; a KU uses them only on its own sentences.
FRAMES: query wordings for a relation about a subject, from the same grammar (the forms the gateway's sub-questions
take: "the director of X", "X's director", ASK frames).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from mxn.expert.para_grammar import ASK, NOUN, VERB

RELATIONS = ['award received', 'cause of death', 'child', 'composer', 'country', 'country of citizenship',
             'country of origin', 'creator', 'date of birth', 'date of death', 'director', 'doctoral advisor', 'editor',
             'educated at', 'employer', 'father', 'founded by', 'has part', 'inception', 'manufacturer', 'mother',
             'occupation', 'performer', 'place of birth', 'place of burial', 'place of death', 'place of detention',
             'presenter', 'producer', 'publication date', 'publisher', 'sibling', 'spouse', 'student of']
R_INDEX = {r: i for i, r in enumerate(RELATIONS)}

# cue words (lower case, matched as whole words in the KU's own sentence)
CUES = {
    "award received": ["award", "awarded", "won", "prize"],
    "cause of death": ["died of", "cause of death", "death from"],
    "child": ["child", "children", "son", "daughter"],
    "composer": ["composer", "composed", "music by", "score"],
    "country": ["country"],
    "country of citizenship": ["citizen", "citizenship", "nationality"],
    "country of origin": ["country of origin", "produced in"],
    "creator": ["creator", "created"],
    "date of birth": ["born", "birth"],
    "date of death": ["died", "death"],
    "director": ["director", "directed", "filmmaker"],
    "doctoral advisor": ["doctoral advisor", "advisor", "supervised"],
    "editor": ["editor", "edited"],
    "educated at": ["educated", "studied", "graduated", "attended", "alumnus"],
    "employer": ["employer", "employed", "worked for", "worked at"],
    "father": ["father", "son of", "daughter of"],
    "founded by": ["founder", "founded by", "co-founded"],
    "has part": ["consists of", "includes"],
    "inception": ["founded", "established", "formed"],
    "manufacturer": ["manufacturer", "manufactured", "made by"],
    "mother": ["mother", "son of", "daughter of"],
    "occupation": ["occupation", "profession"],
    "performer": ["performer", "performed", "singer", "sung", "recorded by"],
    "place of birth": ["born", "birthplace"],
    "place of burial": ["buried", "burial", "interred"],
    "place of death": ["died", "death"],
    "place of detention": ["imprisoned", "detained", "prison"],
    "presenter": ["presenter", "presented", "host", "hosted"],
    "producer": ["producer", "produced"],
    "publication date": ["released", "published", "premiered", "release", "publication"],
    "publisher": ["publisher", "published by"],
    "sibling": ["sibling", "brother", "sister"],
    "spouse": ["spouse", "wife", "husband", "married"],
    "student of": ["student of", "studied under", "pupil of"],
}
for _r, _ns in NOUN.items():                     # the grammar's nouns / verbs are cues too
    CUES[_r] = sorted(set(CUES[_r]) | {n for n in _ns if " " not in n})
for _r, _v in VERB.items():
    CUES[_r] = sorted(set(CUES[_r]) | {_v.split()[0]})

# relation-vocabulary synonyms (options round, user 2026-11-01; N4: words that name the relation, no layout or
# convention rules, no entity knowledge such as demonyms)
SYNONYMS = {
    "date of birth": ["born on", "birthday", "date of birth"],
    "date of death": ["passed away", "date of death", "killed", "executed", "murdered", "assassinated"],
    "place of birth": ["born in", "native of", "hometown", "place of birth"],
    "place of death": ["died in", "died at", "passed away in", "place of death"],
    "publication date": ["aired", "debuted", "launched", "issued", "first shown", "first performed", "broadcast"],
    "inception": ["incorporated", "created", "opened", "inaugurated", "launched", "built", "started"],
    "country": ["located in", "situated in", "nation"],
    "country of origin": ["made in", "originated in", "origin"],
    "country of citizenship": ["national", "citizens", "naturalized"],
    "spouse": ["wed", "widow", "widower", "partner", "consort", "marriage"],
    "father": ["born to", "child of", "his father", "her father"],
    "mother": ["born to", "child of", "his mother", "her mother"],
    "child": ["sons", "daughters", "father of", "mother of"],
    "sibling": ["siblings", "brothers", "sisters"],
    "director": ["helmed", "direction"],
    "composer": ["soundtrack", "music", "songwriter", "written by"],
    "producer": ["production", "produced by"],
    "performer": ["sang", "sings", "performed by", "band", "artist"],
    "educated at": ["alumna", "student at", "studied at", "graduate of"],
    "employer": ["joined", "hired", "worked"],
    "award received": ["received", "honored", "nominated", "winner"],
    "cause of death": ["died from", "succumbed"],
    "place of burial": ["grave", "cemetery", "resting place"],
    "founded by": ["founded by", "established by", "created by"],
    "publisher": ["publishing"],
    "manufacturer": ["built by", "produced by"],
}
for _r, _ws in SYNONYMS.items():
    CUES[_r] = sorted(set(CUES[_r]) | set(_ws))

# expected value kind of each relation (relation vocabulary, signal format: N4; reports/design_value_kind.md). The
# relation code carries it; a KU maps it to its own slot kinds (SLOT_KINDS) and answers only from those slots.
# "number": in the vocabulary, but none of the 34 relations takes a number and slots have no number kind.
KIND = {r: "entity" for r in RELATIONS}
KIND.update({"date of birth": "date", "date of death": "date", "publication date": "date", "inception": "date"})
SLOT_KINDS = {"date": ("date", "year"), "number": (), "entity": ("name",)}

# Relation vocabulary of another dataset (reports/design_metaqa.md): MXN_VOCAB names a file in mxn/neuron/vocab; it
# replaces the vocabulary in place (other modules hold these same objects). Unset: the 2Wiki vocabulary, unchanged.
VOCAB = os.environ.get("MXN_VOCAB") or None
GAPS: list[str] | None = None
if VOCAB:
    _v = json.load(open(Path(__file__).parent / "vocab" / f"{VOCAB}.json"))
    RELATIONS[:] = _v["relations"]
    R_INDEX.clear(); R_INDEX.update({r: i for i, r in enumerate(RELATIONS)})
    CUES.clear(); CUES.update({r: sorted(set(_v["cues"].get(r, []))) for r in RELATIONS})
    KIND.clear(); KIND.update({r: _v["kind"].get(r, "entity") for r in RELATIONS})
    NOUN, ASK = _v["nouns"], _v["ask"]
    GAPS = list(_v["gaps"])


def fits(rel: str, slot_kind: str) -> bool:
    """Whether a slot of this kind can answer rel (by the relation's expected value kind)."""
    return slot_kind in SLOT_KINDS[KIND[rel]]


def frames(rel: str, subject: str) -> list[str]:
    """Query wordings for (rel, subject), as the gateway's sub-questions are written."""
    nouns = NOUN.get(rel, [rel])
    out = [f"the {n} of {subject}" for n in nouns] + [f"{subject}'s {n}" for n in nouns]
    out += [a.format(D=subject) for a in ASK.get(rel, [])]
    out.append(f"{rel} of {subject}")
    return out


def rel_code(rel: str | None) -> np.ndarray:
    v = np.zeros(len(RELATIONS), np.float32)
    if rel is not None:
        v[R_INDEX[rel]] = 1.0
    return v


def signal_text(text: str, rel: str | None, rich: bool) -> str:
    """Option 2 (richer query signal): the gateway embeds the relation vocabulary with the sub-question, e.g.
    "the director of X ; director: directed, director, filmmaker". Signal format only (N4: relation vocabulary);
    the same function writes the KU's training inputs and the gateway's queries."""
    if not rich or rel is None:
        return text
    return f"{text} ; {rel}: {', '.join(sorted(CUES[rel])[:6])}"
