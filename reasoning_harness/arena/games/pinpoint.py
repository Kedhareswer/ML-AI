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


class LinkGuesser:
    """Pinpoint guesser that also uses three kinds of links counted in the same corpus.

    For a candidate answer W and each clue c it looks for:
      * a category pattern: "planets such as mars", "mars and other planets"  (W is a category of c)
      * a two-word phrase: "poker face", "face value"                          (c W or W c)
      * a compound word: "butter" + "fly" = "butterfly"                        (cW or Wc)
    and falls back on word-vector similarity ("guitar" near "piano"). The answer must
    link to every clue, so the score is the mean link strength over the clues.
    Every count comes from WikiText-103; there is no pretrained model or answer list.
    """

    def __init__(self, model_dir, base):
        model_dir = Path(model_dir)
        self.g = base
        self.freq = {}
        for line in (model_dir / "words.txt").read_text().splitlines():
            w, n = line.split()
            self.freq[w] = int(n)
        self.total = sum(self.freq.values())
        bg = np.load(model_dir / "bigrams.npz")
        a, b, n = bg["a"], bg["b"], bg["n"].astype(np.float64)
        order = np.argsort(a, kind="stable")
        self.ra, self.rb, self.rn = a[order], b[order], n[order]        # right neighbours: c W
        order = np.argsort(b, kind="stable")
        self.la, self.lb, self.ln = b[order], a[order], n[order]        # left neighbours: W c
        h = np.load(model_dir / "hearst.npz")
        self.cats = {}
        for c, m, k in zip(h["cat"].tolist(), h["mem"].tolist(), h["n"].tolist()):
            d = self.cats.setdefault(self.sing(m), {})
            d[self.sing(c)] = d.get(self.sing(c), 0) + k
        self.stop = set(base.words[:150])
        self.vfreq = np.array([self.freq.get(w, 1) for w in base.words], np.float64)
        # answers are ordinary words: not stopwords, not suffix fragments ("ing"), not rare tokens
        suffix = {"ing", "ers", "ion", "ness", "less", "ful", "ish", "est", "ism", "ist", "ment"}
        self.ok = np.array([len(w) >= 3 and w not in self.stop and w not in suffix and w.isalpha()
                            and self.freq.get(w, 0) >= 300 for w in base.words])

    def sing(self, w):
        for suf, rep in (("ies", "y"), ("es", ""), ("s", "")):
            if w.endswith(suf) and len(w) >= len(suf) + 2 and self.freq.get(w[:-len(suf)] + rep, 0) >= 20:
                return w[:-len(suf)] + rep
        return w

    def _links(self, clue):
        """Per-vocabulary-word link strengths in [0, 1] for one clue, and the reason for each."""
        V = len(self.g.words)
        strength = np.zeros(V)
        kind = np.zeros(V, np.int8)  # 1 category, 2 phrase "c W", 3 phrase "W c", 4 compound "cW", 5 compound "Wc"
        c = self.sing(clue)
        for form in {clue, c}:
            i = self.g.index.get(form)
            if i is None:
                continue
            fc = self.vfreq[i]
            for keys, idx, cnt, k in ((self.ra, self.rb, self.rn, 2), (self.la, self.lb, self.ln, 3)):
                lo, hi = np.searchsorted(keys, [i, i + 1])
                w, n = idx[lo:hi], cnt[lo:hi]
                good = n >= 5
                w, n = w[good], n[good]
                pmi = np.log(n * self.total / (fc * self.vfreq[w]))
                s = np.clip(pmi / 10, 0, 1) * np.clip(np.log(n) / np.log(50), 0, 1)
                better = s > strength[w]
                strength[w[better]], kind[w[better]] = s[better], k
        for cat, n in self.cats.get(c, {}).items():
            j = self.g.index.get(cat)
            if j is not None:
                s = min(1.0, np.log1p(n) / np.log(20)) * 1.2
                if s > strength[j]:
                    strength[j], kind[j] = s, 1
        return strength, kind

    def _compounds(self, clue, cand):
        out = {}
        for j in cand:
            w = self.g.words[j]
            for word, k in ((clue + w, 4), (w + clue, 5)):
                n = self.freq.get(word, 0)
                if n >= 20:
                    s = min(1.0, np.log1p(n) / np.log(300))
                    if s > out.get(j, (0, 0))[0]:
                        out[j] = (s, k)
        return out

    def score(self, clues):
        clues = [c.lower() for c in clues if c.strip()]
        ids = [self.g.index[c] for c in clues if c in self.g.index]
        V = len(self.g.words)
        links = [self._links(c) for c in clues]
        total = np.zeros(V)
        for s, _ in links:
            total += s
        cand = np.flatnonzero(self.ok & (total > 0))
        cand = cand[np.argsort(-total[cand])][:3000]
        # compounds: check the best phrase partners plus the 20k most frequent words
        pool = np.union1d(cand, np.flatnonzero(self.ok[:20000]))
        reasons = []
        for c, (s, k) in zip(clues, links):
            for j, (cs, ck) in self._compounds(c, pool).items():
                if cs > s[j]:
                    s[j], k[j] = cs, ck
            reasons.append(k)
        strengths = np.array([s for s, _ in links])
        link = strengths.mean(0)
        cover = (strengths > 0.1).mean(0)  # share of clues this answer links to
        sim = np.zeros(V)
        if ids:
            per = self.g.vecs @ self.g.vecs[ids].T
            sim = np.clip(0.5 * per.mean(1) + 0.5 * per.min(1), 0, 1)
        # a linked answer must connect to (almost) every clue; otherwise fall back on meaning similarity
        need = 1.0 if len(clues) <= 2 else 0.75
        return np.where(cover >= need, 1 + link, sim), links, clues

    def rank(self, clues, k=5):
        total, links, clues = self.score(clues)
        banned = {c.lower() for c in clues} | {self.sing(c.lower()) for c in clues}
        out = []
        for j in np.argsort(-total):
            w = self.g.words[j]
            if not self.ok[j] or w in banned or self.sing(w) in banned or any(b in w or w in b for b in banned):
                continue
            if self.sing(w) in {x for x, _ in out}:  # "storm" and "storms" are one guess
                continue
            out.append((self.sing(w), float(total[j])))
            if len(out) >= k:
                break
        return out

    def explain(self, clues, word):
        """Why `word` fits each clue, in plain words."""
        _, links, clues = self.score(clues)
        j = self.g.index.get(word)
        if j is None:
            j = next((i for i, w in enumerate(self.g.words) if self.sing(w) == word), None)
        out = []
        for c, (s, k) in zip(clues, links):
            if j is None or s[j] <= 0:
                i = self.g.index.get(c)
                close = j is not None and i is not None and float(self.g.vecs[i] @ self.g.vecs[j]) > 0.3
                out.append("used in similar contexts (word vectors)" if close else None)
                continue
            w = self.g.words[j]
            out.append({1: f"{c} is a kind of {word}", 2: f"“{c} {w}”", 3: f"“{w} {c}”",
                        4: f"“{c}{w}”", 5: f"“{w}{c}”"}.get(int(k[j]), f"{c} ~ {word}"))
        return out
