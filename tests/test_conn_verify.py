"""The verifier's message rule (experiments/expert/conn_verify.py): success, failure, correction, ambiguity."""
from experiments.expert.conn_verify import feedback_messages


def hop(src, rel, ku):
    return {"src": src, "src_relation": rel, "value_sha1": "v", "ku": ku}


def test_success_failure_correction_and_ambiguous():
    gold = {"q": [[1, "father", 2], [5, "director", 6], [5, "director", 7], [8, "mother", 9], [8, "spouse", 10]]}
    rows = [{"id": "q", "t": 3, "hops": [hop(1, "father", 2), hop(1, "father", 4), hop(5, "director", 4),
                                          hop(8, "mother", 4), hop(11, "father", 4)]}]
    m = [(x["k"], x["outcome"], x["target"]) for x in feedback_messages(rows, gold, 0)]
    assert m == [(1, "success", 2), (1, "failure", 4), (1, "correction", 2), (5, "failure", 4),
                 (5, "correction_ambiguous", None), (8, "failure", 4), (8, "correction", 9)]


def test_relation_mismatch_falls_back_to_the_sources_only_target():
    gold = {"q": [[1, "father", 2]]}
    rows = [{"id": "q", "t": 0, "hops": [hop(1, "parent", 4)]}]
    assert [(x["outcome"], x["target"]) for x in feedback_messages(rows, gold, 0)] == [("failure", 4), ("correction", 2)]
