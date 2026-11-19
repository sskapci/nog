"""Our expert end to end (steps 7–8): question → plan → entity KUs → FIND inside KUs → operations → answer,
with a combined confidence (step 7 rules).

FIND(KU, r): the KU's read-only extractor copy on its own sentences (date rules first for date relations, else
the v2 decoding at the dev threshold) → (value, extraction confidence: span score, 1.0 for a rule date); the
KU's calibrated confidence for the query "<r> of <title>". A variable is resolved to a KU by the step 6
resolver (the question as tie context).
Confidence = plan confidence x product over FINDs of (extraction confidence x KU confidence).
"""

from __future__ import annotations

from mxn.expert.extractor import dates_rule, decode, piece_scores
from mxn.expert.ops import answer as template
from mxn.expert.planner import predict as plan_questions
from mxn.common.textkeys import _DISAMBIG


class Expert:
    def __init__(self, planner, extractor, sp, enc, resolver, units: list, t: float, equiv: dict | None,
                 entity_form: str = "title"):
        """units[k]: the KU of resolver.titles[k] (title, sentences); entity_form: how a winning
        entity is written ("title": KU title without its "(...)"; "span": the question's mention)."""
        self.planner, self.extractor, self.sp, self.enc, self.resolver = planner, extractor, sp, enc, resolver
        self.units, self.t, self.equiv, self.entity_form = units, t, equiv, entity_form

    def find(self, k: int, relation: str) -> dict:
        u = self.units[k]
        value = dates_rule(relation, u.sentences)
        if value is not None:
            p_ext = 1.0
        else:
            value, p_ext = decode(piece_scores(self.extractor, self.sp, relation, u.title, u.sentences), u.sentences, self.t)
        # p_KU = 1: the KU's own O1 confidence is retired (decision 2026-10-30; c2 never used it), as in every host run
        return {"ku": k, "title": u.title, "relation": relation, "value": value, "p_ext": p_ext, "p_ku": 1.0}

    def hop(self, value: str, question: str):
        return self.resolver(value, question) if value else (None, None)

    def run(self, questions: list[str]) -> list[dict]:
        out = []
        for q, plan in zip(questions, plan_questions(self.planner, self.sp, questions)):
            out.append(self.answer(q, plan))
        return out

    def answer(self, q: str, plan: dict) -> dict:
        rec = {"plan": plan, "finds": [], "entities": [], "answer": None, "conf": 0.0}
        spans = plan["spans"]
        need = 1 if plan["template"] == "chain" else 2
        if len(spans) < need or any(s is None for s in spans[:need]):
            return rec
        ents = [self.resolver(q[a:b], q) for a, b in spans[:need]]
        rec["entities"] = [[self.resolver.titles[k], rule] for k, rule in ents]
        f1 = [self.find(k, plan["r1"]) for k, _ in ents]
        rec["finds"] += f1
        if plan["template"] == "chain":
            k2, rule = self.hop(f1[0]["value"], q)
            if k2 is None:
                return rec
            f2 = self.find(k2, plan["r2"]); f2["resolved_by"] = rule; rec["finds"].append(f2)
            values = {"v1": [f1[0]["value"]], "v2": [f2["value"]] if f2["value"] else []}
        else:
            values = {"v1": [f1[0]["value"]] if f1[0]["value"] else [], "v2": [f1[1]["value"]] if f1[1]["value"] else []}
            if plan["template"] == "bridge":
                for side, var in ((0, "v3"), (1, "v4")):
                    k2, rule = self.hop(f1[side]["value"], q)
                    if k2 is None:
                        values[var] = []; continue
                    f2 = self.find(k2, plan["r2"]); f2["resolved_by"] = rule; rec["finds"].append(f2)
                    values[var] = [f2["value"]] if f2["value"] else []
        forms = {"title": [_DISAMBIG.sub("", self.resolver.titles[k]) for k, _ in ents],
                 "span": [q[a:b] for a, b in spans[:need]]}
        rec["answer_forms"] = {f: template(plan, values, names, self.equiv) for f, names in forms.items()}
        rec["answer"] = rec["answer_forms"][self.entity_form]
        c = plan["conf"]
        for f in rec["finds"]:
            c *= f["p_ext"] * f["p_ku"]
        rec["conf"] = c if rec["answer"] is not None else 0.0
        return rec
