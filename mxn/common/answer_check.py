"""Does a sentence answer a question?

Used twice, with the same code: to filter weak held-out questions (the source sentence must
answer its own question), and as an answer-level metric (the sentence routing returned
answers the question, whichever unit it came from).

- `vague(q, title, sentence)`: rule-based. The question points at "the text" instead of a
  subject, or shares no content word with the article title or the sentence.
- `AnswerJudge`: a small local instruct model (CPU) reads question + sentence and gives
  P(yes) from its first answer token.
"""

from __future__ import annotations

import re

from mxn.common.stopwords import _STOP, _WORD

_DEICTIC = re.compile(r"\b(the|this|that|given|above|provided)\s+(text|article|passage|sentence|paragraph|"
                      r"statement|excerpt|document)\b|\bmentioned\b", re.I)


def _content(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text) if w.lower() not in _STOP and len(w) > 2}


def vague(question: str, title: str, sentence: str) -> str | None:
    """Reason the question is too vague to be a fair test, or None."""
    if _DEICTIC.search(question):
        return "refers to the text"
    if not _content(question) & (_content(title) | _content(sentence)):
        return "no specific subject"
    return None


class AnswerJudge:
    def __init__(self, model_name: str, llm=None):
        if llm is None:
            from mxn.baselines.llm import LocalLLM

            llm = LocalLLM(model_name)
        self.llm = llm

    @staticmethod
    def prompt(question: str, sentence: str, title: str = "") -> str:
        src = f" (from the article '{title}')" if title else ""
        return (f"Question: {question}\nSentence{src}: {sentence}\n"
                "Does the sentence alone contain the answer to the question? Answer Yes or No.")

    def scores(self, items: list[tuple[str, str, str]], batch_size: int = 16, log=None) -> list[float]:
        """items: (question, sentence, title) -> P(the sentence answers the question)."""
        out: list[float] = []
        for b in range(0, len(items), batch_size):
            out += self.llm.yes_prob_batch([self.prompt(q, s, t) for q, s, t in items[b: b + batch_size]])
            if log:
                log(f"{len(out)}/{len(items)} judged")
        return out


class ExtractiveJudge:
    """SQuAD 2.0 extractive QA model: P(yes) = sigmoid(best answer span - no-answer score).

    A different model family from the question generator (no self-agreement), and trained to
    say when a passage has no answer. The article title is prefixed to the sentence, as the
    LLM judge also sees it.
    """

    def __init__(self, model_name: str = "deepset/roberta-base-squad2", max_span: int = 30):
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name, dtype=torch.float32).eval()
        self.max_span = max_span

    def scores(self, items: list[tuple[str, str, str]], batch_size: int = 32, log=None) -> list[float]:
        import torch

        out: list[float] = []
        for b in range(0, len(items), batch_size):
            chunk = items[b: b + batch_size]
            enc = self.tok([q for q, _, _ in chunk], [f"{t}: {s}" if t else s for _, s, t in chunk],
                           truncation="only_second", max_length=384, padding=True, return_tensors="pt")
            with torch.inference_mode():
                o = self.model(**enc)
            for n in range(len(chunk)):
                ctx = torch.tensor([sid == 1 for sid in enc.sequence_ids(n)])
                st, en = o.start_logits[n], o.end_logits[n]
                null = float(st[0] + en[0])
                pair = st[:, None] + en[None, :]
                i = torch.arange(len(st))
                ok = ctx[:, None] & ctx[None, :] & (i[None, :] >= i[:, None]) & (i[None, :] - i[:, None] < self.max_span)
                span = float(pair.masked_fill(~ok, -1e9).max())
                out.append(float(torch.sigmoid(torch.tensor(span - null))))
            if log:
                log(f"{len(out)}/{len(items)} judged")
        return out
