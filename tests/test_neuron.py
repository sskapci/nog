"""KU neurons: slots and examples come from the KU's own article only; training one KU touches no other KU; the
saved KU holds no raw text (neuron check, reports/design_ku_neurons.md §6)."""

import numpy as np
import torch

from mxn.neuron.cloze import MASK, examples, slot_relations
from mxn.neuron.kunet import answer, train
from mxn.neuron.signal import RELATIONS, frames, rel_code
from mxn.neuron.slots import build_slots

TITLE = "Tomorrow at Seven"
SENTS = ["Tomorrow at Seven is a 1933 American pre-Code film directed by Ray Enright.",
         "It was released on 5 August 1933 and stars Chester Morris and Vivienne Osborne."]


def test_slots_from_own_article():
    labels = {s.label: s.kind for s in build_slots(TITLE, SENTS)}
    assert labels["Ray Enright"] == "name" and labels["5 August 1933"] == "date" and labels["1933"] == "year"
    assert "Code" not in labels and TITLE not in labels and "Tomorrow" not in labels
    assert "Chester Morris" in labels and "Vivienne Osborne" in labels      # "and" does not join names


def test_cues_precede_slot_and_fit_kind():
    slots = build_slots(TITLE, SENTS)
    rel = {slots[k].label: r for (k, _), r in slot_relations(SENTS, slots).items()}
    assert rel["Ray Enright"] == {"director"}
    assert rel["American"] == set()                     # the cue comes after it
    assert rel["5 August 1933"] == {"publication date"}


def test_examples_use_only_own_text():
    slots = build_slots(TITLE, SENTS)
    ex = examples(TITLE, SENTS, slots)
    allowed = set(" ".join(SENTS + [MASK, TITLE]).lower().split())
    for r in RELATIONS:
        allowed |= set(" ".join(frames(r, TITLE)).lower().split())
    for text, r, t in ex:
        assert set(text.lower().split()) <= allowed
        assert r is None or r in RELATIONS and -1 <= t < len(slots)
    assert any(t == -1 for _, _, t in ex) and any(t >= 0 for _, _, t in ex)


def _fake_xy(ex, d=16, seed=0):
    rng = np.random.default_rng(seed)
    vec = {}
    X = np.stack([vec.setdefault(t, rng.standard_normal(d).astype(np.float32)) for t, _, _ in ex])
    return X, np.stack([rel_code(r) for _, r, _ in ex]), np.array([t for _, _, t in ex])


def test_training_is_local_and_deterministic():
    """Same KU, same examples → identical parameters, whatever other KUs exist or were trained before."""
    slots = build_slots(TITLE, SENTS); X, R, y = _fake_xy(examples(TITLE, SENTS, slots))
    a, _ = train(X, R, y, len(slots), hidden=8, epochs=5, seed=3)
    other = build_slots("Edwin Carewe", ["His father, Frank Fox, was white."])
    train(*_fake_xy(examples("Edwin Carewe", ["His father, Frank Fox, was white."], other), seed=1), len(other),
          hidden=8, epochs=5, seed=4)
    b, _ = train(X, R, y, len(slots), hidden=8, epochs=5, seed=3)
    assert all(torch.equal(p, q) for p, q in zip(a.state_dict().values(), b.state_dict().values()))


def test_answer_needs_no_text():
    slots = build_slots(TITLE, SENTS); X, R, y = _fake_xy(examples(TITLE, SENTS, slots))
    net, _ = train(X, R, y, len(slots), hidden=8, epochs=5)
    f, k = answer(net, X[0], R[0])
    assert 0.0 <= f <= 1.0 and 0 <= k < len(slots)


LIFE = ["Edwin Carewe (March 5, 1883 – January 22, 1940) was an American film director.",
        "Tomorrow at Seven is a 1933 American pre-Code film directed by Ray Enright."]


def test_layout_arm_measurement_labels():
    from mxn.neuron.arms import layout_labels
    slots = build_slots("Edwin Carewe", LIFE)
    lab = {slots[k].label: v for k, v in layout_labels(LIFE, slots).items()}
    assert lab["March 5, 1883"] == {"date of birth"} and lab["January 22, 1940"] == {"date of death"}
    assert lab["1933"] == {"publication date"} and "country of citizenship" in lab["American"]


def test_default_examples_unchanged_by_extra_option():
    slots = build_slots(TITLE, SENTS)
    assert examples(TITLE, SENTS, slots) == examples(TITLE, SENTS, slots, extra=None, cues=True)
    ex = examples(TITLE, SENTS, slots, extra={0: {"father"}}, cues=False)
    assert all(r in (None, "father") or t == -1 for _, r, t in ex)


def test_match_dates_and_names():
    from mxn.neuron.match import same
    assert same("March 5, 1883", "5 March 1883") and not same("1883", "5 March 1883")
    assert same("Ray Enright", "ray enright") and not same(None, "x")


def test_dense_examples_own_text_and_consistent():
    """Option 1: only the KU's own words; no input text with two different targets under the same relation."""
    slots = build_slots(TITLE, SENTS)
    ex = examples(TITLE, SENTS, slots, dense=True)
    allowed = set(" ".join(SENTS + [MASK, TITLE]).lower().split())
    for r in RELATIONS:
        allowed |= set(" ".join(frames(r, TITLE)).lower().split())
    seen = {}
    for text, r, t in ex:
        assert set(text.lower().split()) <= allowed
        if t >= 0:
            assert seen.setdefault((text, r), t) == t
    assert len(ex) > len(examples(TITLE, SENTS, slots))


def test_rich_signal_is_relation_vocabulary_only():
    from mxn.neuron.signal import CUES, signal_text
    t = signal_text("the director of X", "director", True)
    assert t.startswith("the director of X ; director: ")
    assert set(t.split(": ", 1)[1].split(", ")) <= set(CUES["director"])
    assert signal_text("x", None, True) == "x" and signal_text("x", "director", False) == "x"
