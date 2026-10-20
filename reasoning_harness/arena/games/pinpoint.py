"""Pinpoint (guess the category linking five clue words) with word vectors trained from scratch.

The vectors are a skip-gram word2vec model trained here on WikiText-103
(Wikipedia text, CC BY-SA). The guesser ranks candidate category words by
cosine similarity to the clues revealed so far. No pretrained language model,
dictionary of categories, or answer key is used at inference time.
"""
import json
import random
import re
from pathlib import Path

import numpy as np

TOKEN = re.compile(r"[a-z]+")


def sentences(parquet_paths):
    import pyarrow.parquet as pq
    for path in parquet_paths:
        table = pq.read_table(path, columns=["text"])
        for text in table.column("text").to_pylist():
            text = text.strip()
            if not text or text.startswith("="):
                continue
            toks = TOKEN.findall(text.lower())
            if len(toks) > 3:
                yield toks


class Corpus:
    def __init__(self, paths):
        self.paths = paths

    def __iter__(self):
        return sentences(self.paths)


def train_vectors(parquet_paths, out_dir, dim=200, epochs=3, workers=1, keep=40000, seed=0):
    from gensim.models import Word2Vec
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = Word2Vec(Corpus(parquet_paths), vector_size=dim, window=6, min_count=20, sg=1, negative=10,
                     sample=1e-4, epochs=epochs, workers=workers, seed=seed)
    words = model.wv.index_to_key[:keep]
    vecs = np.stack([model.wv[w] for w in words]).astype(np.float16)
    np.save(out_dir / "vectors.npy", vecs)
    (out_dir / "vocab.txt").write_text("\n".join(words) + "\n")
    info = {"kind": "pinpoint_word2vec", "dim": dim, "epochs": epochs, "corpus": "WikiText-103 raw train (CC BY-SA 3.0)",
            "corpus_words": int(model.corpus_total_words), "kept_vocab": len(words), "algorithm": "skip-gram, negative sampling",
            "initialisation": f"random (seed {seed})"}
    (out_dir / "manifest.json").write_text(json.dumps(info, indent=2))
    return info


class Guesser:
    def __init__(self, model_dir, candidates=None):
        model_dir = Path(model_dir)
        self.words = (model_dir / "vocab.txt").read_text().split()
        v = np.load(model_dir / "vectors.npy").astype(np.float32)
        self.vecs = v / np.linalg.norm(v, axis=1, keepdims=True).clip(min=1e-6)
        self.index = {w: i for i, w in enumerate(self.words)}
        cand = candidates if candidates is not None else self.words
        self.cand = [w for w in cand if w in self.index]
        self.cand_idx = np.array([self.index[w] for w in self.cand])
        self.head = None
        head_path = model_dir / "category_head.pt"
        if head_path.exists():
            import torch
            self.head = CategoryHead(self, torch.load(head_path, map_location="cpu", weights_only=True))

    def known(self, word):
        return word.lower() in self.index

    def rank(self, clues, k=5, use_head=True):
        ids = [self.index[c.lower()] for c in clues if c.lower() in self.index]
        if not ids:
            return []
        if self.head is not None and use_head:
            score = self.head.logits(ids)
        else:
            centroid = self.vecs[ids].mean(0)
            sims = self.vecs[self.cand_idx] @ centroid
            # a category word should relate to every clue, not just one: blend mean and minimum similarity
            per = self.vecs[self.cand_idx] @ self.vecs[ids].T
            score = 0.5 * sims / max(np.linalg.norm(centroid), 1e-6) + 0.5 * per.min(1)
        banned = {c.lower() for c in clues}
        order = np.argsort(-score)
        out = []
        for i in order:
            w = self.cand[i]
            if w in banned or any(w.rstrip("s") == b.rstrip("s") for b in banned):
                continue
            out.append((w, float(score[i])))
            if len(out) >= k:
                break
        return out


def random_guesser(candidates, rng):
    return lambda clues, k=5: [(w, 0.0) for w in rng.sample(candidates, k)]


def wordnet_puzzles(guesser, count=300, seed=0, nltk_dir=None):
    """Pinpoint-style puzzles from WordNet: a category noun and five of its kinds."""
    import os
    if nltk_dir:
        os.environ.setdefault("NLTK_DATA", nltk_dir)
    from nltk.corpus import wordnet as wn
    common = set(guesser.words[:20000])
    rng = random.Random(seed)
    puzzles, seen = [], set()
    for syn in wn.all_synsets("n"):
        head = syn.lemma_names()[0].lower()
        if "_" in head or head not in common or head in seen:
            continue
        kids = set()
        for h in syn.closure(lambda s: s.hyponyms(), depth=2):
            for lemma in h.lemma_names():
                lemma = lemma.lower()
                if "_" not in lemma and lemma in common and lemma != head:
                    kids.add(lemma)
        if len(kids) >= 6:
            seen.add(head)
            puzzles.append({"answer": head, "accept": [l.lower() for l in syn.lemma_names() if "_" not in l],
                            "clues": rng.sample(sorted(kids), 5)})
    rng.shuffle(puzzles)
    return puzzles[:count]


def evaluate(guess, puzzles):
    """Accuracy when the answer is the top guess after 1..5 clues, and in the top 5."""
    top1 = np.zeros(5)
    top5 = np.zeros(5)
    records = []
    for p in puzzles:
        accept = set(p.get("accept", [])) | {p["answer"]}
        row = {"answer": p["answer"], "clues": p["clues"], "guesses": []}
        for k in range(1, 6):
            g = [w for w, _ in guess(p["clues"][:k], 5)]
            row["guesses"].append(g)
            top1[k - 1] += bool(g) and g[0] in accept
            top5[k - 1] += bool(set(g) & accept)
        records.append(row)
    n = max(len(puzzles), 1)
    return {"cases": len(puzzles), "top1_after_k_clues": (100 * top1 / n).round(1).tolist(),
            "top5_after_k_clues": (100 * top5 / n).round(1).tolist()}, records


class CategoryHead:
    """Small network, trained from scratch, mapping the mean clue vector to the category word.

    Training pairs come from WordNet categories (kinds -> category). Every answer of
    the evaluation sets is excluded, so test categories are never seen in training.
    """

    def __init__(self, guesser, state=None, hidden=512):
        import torch
        from torch import nn
        self.g = guesser
        dim = guesser.vecs.shape[1]
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        if state is not None:
            self.net.load_state_dict(state)
        self.net.eval()
        self.cand = torch.from_numpy(guesser.vecs[guesser.cand_idx])

    def logits(self, ids):
        import torch
        x = torch.from_numpy(self.g.vecs[ids].mean(0, keepdims=True))
        with torch.no_grad():
            q = self.net(x) + x
        q = q / q.norm(dim=1, keepdim=True)
        return (q @ self.cand.T)[0].numpy()


def train_category_head(guesser, exclude, nltk_dir=None, steps=4000, seed=0, log=print):
    import os
    import torch
    import torch.nn.functional as F
    if nltk_dir:
        os.environ.setdefault("NLTK_DATA", nltk_dir)
    from nltk.corpus import wordnet as wn
    torch.manual_seed(seed)
    rng = random.Random(seed)
    cand_pos = {w: i for i, w in enumerate(guesser.cand)}
    exclude = set(exclude)
    groups = []
    for syn in wn.all_synsets("n"):
        heads = [l.lower() for l in syn.lemma_names() if "_" not in l]
        if not heads or any(h in exclude for h in heads) or heads[0] not in cand_pos:
            continue
        kids = set()
        for h in syn.closure(lambda s: s.hyponyms(), depth=2):
            for lemma in h.lemma_names():
                lemma = lemma.lower()
                if "_" not in lemma and lemma in guesser.index and lemma not in heads:
                    kids.add(guesser.index[lemma])
        if len(kids) >= 4:
            groups.append((cand_pos[heads[0]], sorted(kids)))
    head = CategoryHead(guesser)
    head.net.train()
    opt = torch.optim.AdamW(head.net.parameters(), lr=2e-3, weight_decay=1e-4)
    vecs = torch.from_numpy(guesser.vecs)
    for step in range(1, steps + 1):
        batch = [groups[rng.randrange(len(groups))] for _ in range(128)]
        xs = []
        for _, kids in batch:
            k = rng.randint(1, min(5, len(kids)))
            xs.append(vecs[rng.sample(kids, k)].mean(0))
        x = torch.stack(xs)
        q = head.net(x) + x
        q = q / q.norm(dim=1, keepdim=True)
        loss = F.cross_entropy(20 * q @ head.cand.T, torch.tensor([t for t, _ in batch]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            log(json.dumps({"step": step, "loss": round(loss.item(), 4)}))
    head.net.eval()
    return head, len(groups)
