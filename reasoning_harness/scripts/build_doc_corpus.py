"""Build the train/test text corpora for synthetic OCR lines from the repo's own text files.

Train text: ChatGPT app reviews + medical abstract sentences (NLP/train_sent.csv).
Test text:  LinkedIn app reviews + medical abstract sentences from the test split (NLP/test_sent.csv).
The two sides share no source rows, so benchmark lines are unseen text.
"""
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "archive"
OUT = Path(__file__).resolve().parents[1] / "docparse" / "data"
sys.path.insert(0, str(OUT.parents[1]))
from docparse.ocr.charset import normalize  # noqa: E402


def reviews(path):
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for key in ("title", "review"):
                yield row.get(key) or ""


def token_sentences(path):
    words = []
    for line in open(path, encoding="utf-8"):
        w = line.strip()
        if not w:
            if words:
                yield " ".join(words).replace(" ,", ",").replace(" .", ".").replace("( ", "(").replace(" )", ")").replace(" %", "%")
            words = []
        else:
            words.append(w)


def chunks(texts, limit_chars):
    out, total = [], 0
    for t in texts:
        t = " ".join(normalize(t).split())
        if len(t) < 8:
            continue
        out.append(t)
        total += len(t)
        if total > limit_chars:
            break
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = chunks(reviews(ROOT / "NLP/Chatgpt/chatgpt_reviews.csv"), 350_000) + \
        chunks(token_sentences(ROOT / "NLP/train_sent.csv"), 150_000)
    test = chunks(reviews(ROOT / "NLP/LinkedIn/linkedin_reviews.csv"), 60_000) + \
        chunks(token_sentences(ROOT / "NLP/test_sent.csv"), 30_000)
    random.Random(0).shuffle(train)
    random.Random(1).shuffle(test)
    (OUT / "corpus_train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (OUT / "corpus_test.txt").write_text("\n".join(test) + "\n", encoding="utf-8")
    print(len(train), "train paragraphs,", len(test), "test paragraphs")


if __name__ == "__main__":
    main()
