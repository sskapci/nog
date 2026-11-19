from mxn.common.answer_check import AnswerJudge, vague


def test_vague_flags_text_references_and_missing_subject():
    s = "Dellert died on 5 March 2018 in Stockholm of respiratory failure, aged 92."
    assert vague("Which nearby municipalities are mentioned in the text?", "Lyon", s)
    assert vague("What does the article say happened next?", "Kjerstin Dellert", s)
    assert vague("When did she die?", "Kjerstin Dellert", s) == "no specific subject"
    assert vague("At what age did Kjerstin Dellert pass away?", "Kjerstin Dellert", s) is None


def test_judge_batches_and_keeps_order():
    class FakeLLM:
        def yes_prob_batch(self, prompts):
            return [1.0 if "Stockholm" in p.split("Sentence")[1] else 0.0 for p in prompts]

    j = AnswerJudge("unused", llm=FakeLLM())
    items = [("Where?", "In Stockholm.", "A"), ("Where?", "In Paris.", "B")] * 3
    assert j.scores(items, batch_size=4) == [1.0, 0.0] * 3
