"""One KU neuron as its host holds it (reports/design_neuron_e2e.md §1–2): its own network, slot table, weighted
connections and title vector. Everything here is built from the KU's own article (plus its own RESOLVE replies for
pointers) and used only by that KU.

Connections: per name slot that names another KU, {"target": KU id or None, "rels": {relation: weight}, "weight": w}.
Initial weights from the KU's own article (design §2.2): rels[r] = number of occurrences of the slot cued for r
(cloze.slot_relations); weight = number of occurrences of the slot. No later weight changes (no outside corrections,
decision 2026-11-01).

Answering (design §2.3): the KU's own forward pass on the query signal → fire strength f and a slot. f < F_MIN →
"don't know". Arm "first" (control, not the system): the first slot of the fitting kind, f = 1. Arm "all" (set
control, reports/design_metaqa_sets_persons.md): every slot of the fitting kind, f = 1.
Set mode (answer_set; same design): the KU lists every own slot of the kind that passes its own relative rule
p(slot) >= theta_f * p(its top slot); theta_f = 1.0 lists only its top slot (the single answer).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np
import torch

from mxn.neuron.cloze import examples, slot_relations
from mxn.neuron.kunet import KUNet, train
from mxn.neuron.signal import KIND, VOCAB, fits, frames, rel_code, signal_text
from mxn.neuron.slots import Slot, build_slots

F_MIN = 0.5


@dataclass
class Settings:
    hidden: int = 64
    epochs: int = 60
    lr: float = 0.01
    dense: bool = False
    rich: bool = False
    kind: bool = False               # value-kind restriction when answering (design_value_kind.md); training unchanged


@dataclass
class NeuronKU:
    k: int
    title: str
    slots: list
    net: KUNet
    conns: dict                      # slot index → {"target", "rels", "weight"}
    rich: bool
    text: list | None = None         # raw text: training and audit only; dropped before answering
    stats: dict = field(default_factory=dict)
    kind_restrict: bool = False      # answer only from slots of the relation's value kind (design_value_kind.md)
    key: int | None = None           # placement key (locality, design_neuron_locality.md); None = its own id

    @property
    def subject(self) -> str:
        return re.sub(r"\s*\([^)]*\)\s*$", "", self.title)            # as cloze.examples

    def query_text(self, rel: str) -> str:
        """The sub-question the language expert encodes for this KU and relation (feasibility: frames(...)[0])."""
        return signal_text(frames(rel, self.subject)[0], rel, self.rich)

    def answer(self, qv: np.ndarray, rel: str, arm: str = "neuron", f_min: float = F_MIN) -> tuple[float, int | None]:
        """(fire strength, slot index or None). f_min: below it the KU says it does not know (picked on dev-fit)."""
        if rel not in KIND or not self.slots:
            return 0.0, None
        fit = [j for j, s in enumerate(self.slots) if fits(rel, s.kind)]
        if arm == "first":
            return (1.0, fit[0]) if fit else (0.0, None)
        with torch.inference_mode():
            z = self.net(torch.from_numpy(qv[None]), torch.from_numpy(rel_code(rel)[None]))[0]
        if self.kind_restrict and not os.environ.get("NKU_NO_KIND"):   # value kind (design_value_kind.md); the env
            if not fit:                                                 # switch is the dry runs' known-bad only
                return 0.0, None
            keep = torch.full_like(z, float("-inf")); keep[fit] = 0.0; keep[-1] = 0.0
            z = z + keep                     # the KU answers only from its own slots of the asked kind (or none)
        p = torch.softmax(z, 0)
        f, j = float(1 - p[-1]), int(p[:-1].argmax())
        return (f, j) if f >= f_min else (f, None)

    def answer_set(self, qv: np.ndarray | None, rel: str, arm: str, f_min: float, theta_f: float) -> tuple[float, list]:
        """Set mode: (fire strength, [(slot index, p), ...] best first). neuron: answer()'s top slot, then every other
        fitting slot with p >= theta_f * p(top); nothing when the top is below f_min. first: its first fitting slot;
        all: every fitting slot (p 1.0)."""
        if rel not in KIND or not self.slots:
            return 0.0, []
        fit = [j for j, s in enumerate(self.slots) if fits(rel, s.kind)]
        if arm == "first":
            return (1.0, [(fit[0], 1.0)]) if fit else (0.0, [])
        if arm == "all":
            return (1.0, [(j, 1.0) for j in fit]) if fit else (0.0, [])
        f, j = self.answer(qv, rel, arm, f_min)
        if j is None:
            return f, []
        p = self.probs(qv, rel)
        top = float(p[j])
        rest = sorted(((float(p[i]), i) for i in (fit if self.kind_restrict else range(len(self.slots)))
                       if i != j and float(p[i]) >= theta_f * top and float(p[i]) > 0), reverse=True)
        return f, [(j, top)] + [(i, q) for q, i in rest]

    def labels(self) -> dict:
        """The KU's own normalized slot labels → slot indices (built once from its slots). The host's slot-value cache
        (reports/design_slot_cache.md) keys on exactly these labels, so both use one normalization."""
        from mxn.expert.ops import normalize
        lab = self.__dict__.get("_labels")
        if lab is None:
            lab = {}
            for j, s in enumerate(self.slots):
                lab.setdefault(normalize(s.label), []).append(j)
            self.__dict__["_labels"] = lab
        return lab

    def holds(self, rel: str, value: str) -> list[int]:
        """Reverse lookup, first step (the KU's own slot table): its slots of rel's value kind whose label is value."""
        from mxn.expert.ops import normalize
        lab = self.labels()
        v = normalize(value)
        return [j for j in lab.get(v, ()) if fits(rel, self.slots[j].kind)] if v else []

    def probs(self, qv: np.ndarray, rel: str) -> np.ndarray:
        """The KU's own distribution over its slots and "none" for rel (value-kind restricted, as answer)."""
        with torch.inference_mode():
            z = self.net(torch.from_numpy(qv[None]), torch.from_numpy(rel_code(rel)[None]))[0]
        if self.kind_restrict and not os.environ.get("NKU_NO_KIND"):
            fit = [j for j, s in enumerate(self.slots) if fits(rel, s.kind)]
            keep = torch.full_like(z, float("-inf")); keep[fit] = 0.0; keep[-1] = 0.0
            z = z + keep
        return torch.softmax(z, 0).numpy()

    def lists(self, p: np.ndarray | None, rel: str, value, rule: str, theta: float, arm: str = "neuron") -> float | None:
        """Reverse lookup (reports/design_reverse_lookup.md): does this KU list value for rel? p = its own probs(...)
        (cached by its host; None for the first arm). Yes iff a slot labelled value passes the rule: "rel": p(slot) >=
        theta * its own top slot's p; "abs": p(slot) >= theta. Arm first (control): yes iff value is its first slot of
        the kind. Returns p(slot) (1.0 for first) or None. value may be a list (set mode: yes iff it lists any of them).
        Arm all (set control): yes iff some slot of the kind is labelled value."""
        js = sorted({j for v in value for j in self.holds(rel, v)}) if isinstance(value, list) else self.holds(rel, value)
        if not js:
            return None
        if arm == "all":
            return 1.0
        if arm == "first":
            fit = [j for j, s in enumerate(self.slots) if fits(rel, s.kind)]
            return 1.0 if fit and fit[0] in js else None
        top = float(p[:-1].max()); pj = max(float(p[j]) for j in js)
        ok = pj >= theta * top if rule == "rel" else pj >= theta
        return pj if ok else None


def build(k: int, title: str, sentences: list[str], enc, st: Settings, gold: list | None = None,
          equiv: dict | None = None) -> NeuronKU:
    """Train one KU alone from its own article (same steps as experiments/neuron/train_kus.py, arm S). gold (arm O,
    the oracle ceiling, measurement only, never the system): the KU's gold triples label its slots instead of cues."""
    slots = build_slots(title, sentences)
    if gold is None:
        ex = examples(title, sentences, slots, seed=k, dense=st.dense)
    else:
        from mxn.neuron.arms import oracle_labels
        ex = examples(title, sentences, slots, seed=k, dense=st.dense, extra=oracle_labels(gold, slots, equiv),
                      cues=False)
    X = enc.encode([signal_text(t, r, st.rich) for t, r, _ in ex])
    R = np.stack([rel_code(r) for _, r, _ in ex]); y = np.array([t for _, _, t in ex])
    net, losses = train(X, R, y, len(slots), st.hidden, st.epochs, lr=st.lr, seed=k)
    cued = slot_relations(sentences, slots)
    conns = {}
    for j, s in enumerate(slots):
        if s.kind != "name":
            continue
        rels = {}
        for o in range(len(s.occ)):
            for r in cued.get((j, o), ()):
                rels[r] = rels.get(r, 0) + 1
        conns[j] = {"target": None, "rels": rels, "weight": len(s.occ)}
    return NeuronKU(k, title, slots, net, conns, st.rich, list(sentences),
                    {"slots": len(slots), "examples": len(ex), "params": net.n_params(), "loss_last": losses[-1]},
                    st.kind)


def anchor(u: NeuronKU) -> int:
    """The KU's own choice of placement anchor (locality): the target of its strongest pointer (weight = how often its
    own article links that slot to a relation; ties: higher weight, then lower target id); itself without pointers."""
    best = [(-sum(c["rels"].values()), c["target"]) for c in u.conns.values() if c["target"] is not None]
    return min(best)[1] if best else u.k


def package(u: NeuronKU) -> dict:
    """The KU's state as stored and moved (no raw text)."""
    return {"k": u.k, "title": u.title, "slots": [vars(s) for s in u.slots], "hidden": u.net.net[0].out_features,
            "d_query": u.net.net[0].in_features - len(rel_code(None)), "state": u.net.state_dict(),
            "conns": u.conns, "rich": u.rich, "stats": u.stats, "kind": u.kind_restrict,
            **({"key": u.key} if u.key is not None else {}), **({"vocab": VOCAB} if VOCAB else {})}


def unpackage(d: dict) -> NeuronKU:
    assert d.get("vocab") == VOCAB, f"KU vocabulary {d.get('vocab')} != this process's {VOCAB}"
    net = KUNet(d["d_query"], len(d["slots"]), d["hidden"]); net.load_state_dict(d["state"]); net.eval()
    return NeuronKU(d["k"], d["title"], [Slot(**s) for s in d["slots"]], net, d["conns"], d["rich"], None, d["stats"],
                    d.get("kind", False), d.get("key"))
