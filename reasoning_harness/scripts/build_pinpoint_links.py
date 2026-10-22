"""Count word links in WikiText-103 for the Pinpoint guesser (no pretrained model, only corpus counts).

Writes to artifacts/arena/pinpoint/:
  bigrams.npz   adjacent word pairs inside the 40k vocabulary ("poker face"), count >= 3
  hearst.npz    category/member pairs from "X such as A, B", "A and other X", "X including A",
                "A is a kind of X" ("planets such as mars")
  words.txt     every corpus word seen >= 5 times with its count (for compounds: "butter" + "fly")

usage: python scripts/build_pinpoint_links.py <dir with wikitext-103-raw-v1 train parquet files>
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from arena.games.pinpoint import TOKEN, sentences  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "artifacts/arena/pinpoint"
LIST = r"((?:[a-z]+, )*[a-z]+(?:,? (?:and|or) [a-z]+)?)"
PATTERNS = [  # (regex, group holding the category, group holding the members)
    (re.compile(r"([a-z]+),? such as " + LIST), 1, 2),
    (re.compile(r"([a-z]+),? including " + LIST), 1, 2),
    (re.compile(r"((?:[a-z]+, )*[a-z]+),? (?:and|or) other ([a-z]+)"), 2, 1),
    (re.compile(r"\b([a-z]+) (?:is|are|was|were) (?:a|an)? ?(?:type|kind|species|variety|form|genus|breed)s? of ([a-z]+)"), 2, 1),
    # "paris is a city in france", "the tibia is a bone of the leg"
    (re.compile(r"\b([a-z]+) (?:is|was) (?:a|an) ([a-z]+) (?:in|of|from|located|that|which|native|found|used)\b"), 2, 1),
]
SPLIT = re.compile(r",? (?:and|or) |, ")


def main(src):
    words = (OUT / "vocab.txt").read_text().split()
    index = {w: i for i, w in enumerate(words)}
    V = len(words)
    freq = Counter()
    hearst = Counter()
    chunks, buf, buffered = [], [], 0
    files = sorted(Path(src).glob("train-*.parquet"))
    import pyarrow.parquet as pq
    n_lines = 0
    for path in files:
        for text in pq.read_table(path, columns=["text"]).column("text").to_pylist():
            text = text.strip().lower()
            if not text or text.startswith("="):
                continue
            n_lines += 1
            toks = TOKEN.findall(text)
            freq.update(toks)
            ids = np.fromiter((index.get(t, -1) for t in toks), np.int64, len(toks))
            if len(ids) > 1:
                a, b = ids[:-1], ids[1:]
                keep = (a >= 0) & (b >= 0)
                buf.append(a[keep] * V + b[keep])
                buffered += len(buf[-1])
            for rx, gc, gm in PATTERNS:
                for m in rx.finditer(text):
                    cat = m.group(gc)
                    for mem in SPLIT.split(m.group(gm)):
                        hearst[(cat, mem)] += 1
            if buffered > 20_000_000:
                chunks.append(_count(np.concatenate(buf)))
                buf, buffered = [], 0
                print(f"{n_lines} lines", flush=True)
    chunks.append(_count(np.concatenate(buf)))
    codes = np.concatenate([c for c, _ in chunks])
    counts = np.concatenate([n for _, n in chunks])
    order = np.argsort(codes, kind="stable")
    codes, counts = codes[order], counts[order]
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    codes, counts = codes[starts], np.add.reduceat(counts, starts)
    keep = counts >= 3
    np.savez_compressed(OUT / "bigrams.npz", a=(codes[keep] // V).astype(np.int32), b=(codes[keep] % V).astype(np.int32),
                        n=counts[keep].astype(np.int32))
    # categories and members are kept as plain words; the guesser folds plurals at query time
    pairs = [(c, m, n) for (c, m), n in hearst.items() if c in index and m in index]
    np.savez_compressed(OUT / "hearst.npz", cat=np.array([p[0] for p in pairs]), mem=np.array([p[1] for p in pairs]),
                        n=np.array([p[2] for p in pairs], np.int32))
    kept = [(w, n) for w, n in freq.most_common() if n >= 5]
    (OUT / "words.txt").write_text("".join(f"{w} {n}\n" for w, n in kept))
    info = json.loads((OUT / "manifest.json").read_text())
    info["link_counts"] = {"lines": n_lines, "bigram_pairs": int(keep.sum()), "hearst_pairs": len(pairs),
                           "corpus_word_types": len(kept), "source": "WikiText-103 raw train (CC BY-SA 3.0)"}
    (OUT / "manifest.json").write_text(json.dumps(info, indent=2))
    print(info["link_counts"])


def _count(codes):
    u, n = np.unique(codes, return_counts=True)
    return u, n


if __name__ == "__main__":
    main(sys.argv[1])
