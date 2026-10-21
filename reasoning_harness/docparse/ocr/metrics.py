"""Character and word error rates (edit distance / reference length)."""


def edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred, ref):
    return edit_distance(pred, ref) / max(1, len(ref))


def wer(pred, ref):
    return edit_distance(pred.split(), ref.split()) / max(1, len(ref.split()))


def summary(preds, refs):
    chars = sum(edit_distance(p, r) for p, r in zip(preds, refs))
    words = sum(edit_distance(p.split(), r.split()) for p, r in zip(preds, refs))
    return {"lines": len(refs), "cer": round(chars / max(1, sum(len(r) for r in refs)), 5),
            "wer": round(words / max(1, sum(len(r.split()) for r in refs)), 5),
            "line_exact": round(sum(p == r for p, r in zip(preds, refs)) / max(1, len(refs)), 4)}
