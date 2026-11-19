"""Step 1 (decisions 2026-10-24b): our subword tokenizer, trained from scratch.

SentencePiece unigram, 16,000 pieces, byte fallback (no unknown characters), case kept (names
matter), NFKC. Text: the context paragraphs of the 2Wiki fit split (deduplicated by title + text)
and every article of Simple English Wikipedia 20231101 (one line per paragraph). Held-out test
questions are never read. Sampling: 4,000,000 lines, shuffled, seed via sentencepiece's own.

usage: tokenizer.py DATA_DIR OUT_DIR
"""
import json, sys, time
from pathlib import Path

import sentencepiece as spm
from datasets import load_dataset

data, out = Path(sys.argv[1]), Path(sys.argv[2])
fit = set((out / "fit_ids.txt").read_text().split())
t0 = time.time()
lines, seen = 0, set()
corpus = out / "tok_corpus.txt"
with open(corpus, "w") as f:
    for r in json.load(open(data / "train.json")):
        if r["_id"] not in fit:
            continue
        for title, sents in r["context"]:
            key = (title, sents[0] if sents else "")
            if key in seen:
                continue
            seen.add(key)
            f.write(title + "\n" + " ".join(s.strip() for s in sents) + "\n"); lines += 2
    n2wiki = lines
    for a in load_dataset("wikimedia/wikipedia", "20231101.simple", split="train"):
        f.write(a["title"] + "\n"); lines += 1
        for p in a["text"].split("\n"):
            if len(p.strip()) > 1:
                f.write(p.strip() + "\n"); lines += 1
print(f"corpus: {lines} lines ({n2wiki} from 2Wiki fit, {len(seen)} paragraphs) in {time.time() - t0:.0f}s", flush=True)
spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(out / "tok16k"), vocab_size=16000,
                               model_type="unigram", character_coverage=0.9995, byte_fallback=True,
                               normalization_rule_name="nfkc", input_sentence_size=4_000_000,
                               shuffle_input_sentence=True, num_threads=8, max_sentence_length=8192,
                               pad_id=0, unk_id=1, bos_id=2, eos_id=3, user_defined_symbols=["[SEP]", "[MASK]"])
sp = spm.SentencePieceProcessor(model_file=str(out / "tok16k.model"))
test = [json.loads(l)["question"] for l in open("artifacts/2wiki/tune.jsonl")]
n = [len(sp.encode(q)) for q in test]
print(f"done in {time.time() - t0:.0f}s; vocab {sp.get_piece_size()}; tune questions: {sum(n) / len(n):.1f} pieces "
      f"({sum(len(q.split()) for q in test) / len(test):.1f} words)", flush=True)
print(sp.encode(test[0], out_type=str))
