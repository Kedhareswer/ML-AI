"""Page images -> text lines, ruled tables and layout blocks (classical image processing, numpy only).

The neural part is only the line recognizer (model.py). Finding lines, columns and
tables here is rule-based, so its failure modes are predictable:
- ruled tables are read cell by cell; borderless tables are found from aligned gaps;
- multi-column text is split on wide vertical gaps;
- photos/figures are reported as `figure` blocks and not read.
"""
import re

import numpy as np
from PIL import Image

from .render import HEIGHT

LIST_RE = re.compile(r"^((•|-|\*|\d{1,3}[.)]|\(?[a-zA-Z][.)]|\([ivx]+\))\s+|\d{1,3}[.)](?=[A-Za-z]))")
DEF_RE = re.compile(r"^[A-Za-z][\w()./-]*( [\w()./-]+){0,3}:\s+\S")  # "Year_Birth: Customer's birth year"


# ---------- small numpy morphology ----------

def _runs(mask):
    """Start/end (exclusive) of True runs in a 1-D boolean array."""
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _box_mean(a, r):
    """Mean over a (2r+1)^2 window via an integral image."""
    p = np.pad(a, r + 1, mode="edge").astype(np.float64)
    s = p.cumsum(0).cumsum(1)
    k = 2 * r + 1
    tot = s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k]
    return (tot / (k * k))[:a.shape[0], :a.shape[1]]


def _open_1d(mask, k, axis):
    """Morphological opening with a 1 x k line: keeps only runs at least k long."""
    m = mask if axis == 1 else mask.T
    c = np.pad(m.astype(np.int32), ((0, 0), (1, 0))).cumsum(1)
    full = (c[:, k:] - c[:, :-k]) == k          # window starting at j is all ink
    out = np.zeros_like(m)
    # pad k-1 on the right too, so the last k-1 pixels of a row (covered only by windows that
    # start before them) are not dropped
    c2 = np.pad(full.astype(np.int32), ((0, 0), (k, k - 1))).cumsum(1)
    cover = (c2[:, k:] - c2[:, :-k]) > 0       # any full window covers this pixel
    out[:, :cover.shape[1]] = cover[:, :m.shape[1]]
    return out if axis == 1 else out.T


def components(ink):
    """8-connected components from row runs. Returns rows, starts, ends and a list of run-index groups."""
    H = ink.shape[0]
    m = np.pad(ink.astype(np.int8), ((0, 0), (1, 1)))
    d = np.diff(m, axis=1)
    rows, starts = np.nonzero(d == 1)
    _, ends = np.nonzero(d == -1)
    n = len(rows)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    row_start = np.searchsorted(rows, np.arange(H + 1))
    for y in range(1, H):
        i, a1 = row_start[y - 1], row_start[y]
        j, b1 = row_start[y], row_start[y + 1]
        while i < a1 and j < b1:  # runs in consecutive rows touch (diagonals included)
            if starts[i] <= ends[j] and starts[j] <= ends[i]:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
            if ends[i] < ends[j]:
                i += 1
            else:
                j += 1
    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    order = np.argsort(roots, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(roots[order])) + 1) if n else []
    return rows, starts, ends, groups


def drop_components(ink, test):
    """Remove every component for which test(h, w, pixels) is true."""
    rows, starts, ends, groups = components(ink)
    out = ink.copy()
    lengths = ends - starts
    for grp in groups:
        h = rows[grp].max() + 1 - rows[grp].min()
        w = ends[grp].max() - starts[grp].min()
        if test(h, w, int(lengths[grp].sum())):
            for k in grp:
                out[rows[k], starts[k]:ends[k]] = False
    return out


def remove_frames(ink, th):
    """Drop outline shapes (rounded boxes, circles, slide decorations): components much larger than
    a letter in both directions whose bounding box is mostly empty."""
    return drop_components(ink, lambda h, w, px: h > 2.5 * th and w > 2.5 * th and px < 0.08 * h * w)


# ---------- preprocessing ----------

def load_gray(img):
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    g = np.asarray(img.convert("L"), dtype=np.float32)
    if np.median(g) < 110:  # dark mode / inverted scan
        g = 255 - g
    return g


def binarize(g):
    """Ink mask. Dark text on light paper, plus light text on dark fills (buttons, banners, headers).
    Specks of scan noise (components of a few pixels) are removed."""
    r = max(7, min(g.shape) // 60)
    local = _box_mean(g, r)
    gs, k = g, 12.0
    if 1.4826 * float(np.median(np.abs(g - local))) > 6:  # grainy scan: noise level measured robustly (MAD)
        gs = _box_mean(g, 1)  # 3x3 smoothing: grain averages out, strokes do not
        k = max(12.0, 3.0 * 1.4826 * float(np.median(np.abs(gs - local))))
    dark_bg = _box_mean((g < 150).astype(np.float32), r) > 0.55  # mostly dark surroundings (outside edges stay < 0.5)
    ink = np.where(dark_bg, (gs > local + max(18.0, k)) & (gs > 120), (gs < local - k) & (gs < 215))
    return drop_components(ink, lambda h, w, px: px <= 4)


def _deskew_angle(ink):
    small = Image.fromarray((ink * 255).astype(np.uint8))
    scale = 900 / max(small.size)
    if scale < 1:
        small = small.resize((int(small.width * scale), int(small.height * scale)), Image.BILINEAR)
    best, best_a = -1.0, 0.0
    for a in np.arange(-4, 4.01, 0.25):
        prof = np.asarray(small.rotate(a, resample=Image.BILINEAR), dtype=np.float32).sum(1)
        score = float(np.square(np.diff(prof)).sum())
        if score > best:
            best, best_a = score, float(a)
    return best_a


def prepare(img, deskew=True):
    """-> gray (float), ink mask, scale applied, deskew angle."""
    g = load_gray(img)
    scale = 1.0
    if max(g.shape) < 1100:  # small screenshots: upsample so thin strokes survive binarization
        scale = 2.0
        g = np.asarray(Image.fromarray(g.astype(np.uint8)).resize((g.shape[1] * 2, g.shape[0] * 2), Image.BICUBIC),
                       dtype=np.float32)
    ink = binarize(g)
    angle = 0.0
    if deskew and ink.any():
        angle = _deskew_angle(ink)
        if abs(angle) >= 0.25:
            g = np.asarray(Image.fromarray(g.astype(np.uint8)).rotate(angle, resample=Image.BICUBIC, fillcolor=255),
                           dtype=np.float32)
            ink = binarize(g)
    return g, ink, scale, angle


def text_height(ink):
    """Typical text-line height. Row bands give it on plain pages; frames, logos and shapes merge
    bands, so it is capped by the length of vertical letter strokes (stems ~ 0.7 x line height)."""
    bands = [b - a for a, b in _runs(ink.sum(1) > max(1, ink.shape[1] * 0.002)) if b - a >= 4]
    th = float(np.median(bands)) if bands else 20.0
    sample = ink[:, ::3].T  # every third column is plenty
    m = np.pad(sample.astype(np.int8), ((0, 0), (1, 1)))
    d = np.diff(m, axis=1)
    runs = np.flatnonzero(d.ravel() == -1) - np.flatnonzero(d.ravel() == 1)
    runs = runs[(runs >= 5) & (runs < ink.shape[0] // 6)]
    if len(runs) > 200:
        th = min(th, 1.6 * float(np.percentile(runs, 90)))
    return th


# ---------- ruled tables ----------

def _segments(mask, axis):
    """Line pixels -> merged segments (pos0, pos1, start, end) along `axis` (1: horizontal lines)."""
    segs = []
    m = mask if axis == 1 else mask.T
    for i in np.flatnonzero(m.any(1)):
        for s, e in _runs(m[i]):
            for seg in segs:
                if seg[1] >= i - 1 and s < seg[3] and e > seg[2]:
                    seg[1] = i
                    seg[2], seg[3] = min(seg[2], s), max(seg[3], e)
                    break
            else:
                segs.append([i, i, s, e])
    return segs


def find_tables(ink, th):
    """Grid tables from long horizontal and vertical rules. Returns (tables, rule mask)."""
    h = _open_1d(ink, int(max(40, 5 * th)), 1)
    v = _open_1d(ink, int(max(20, 1.6 * th)), 0)
    hs, vs = _segments(h, 1), _segments(v, 0)
    # Union-find over rule segments that touch each other.
    items = [("h", s) for s in hs] + [("v", s) for s in vs]
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    tol = 4
    for i, (ki, a) in enumerate(items):
        if ki != "h":
            continue
        for j, (kj, b) in enumerate(items):
            if kj != "v":
                continue
            # h: rows a0..a1, cols a2..a3 ; v: cols b0..b1, rows b2..b3
            if b[0] - tol <= a[3] and b[1] + tol >= a[2] and a[0] + tol >= b[2] and a[1] - tol <= b[3]:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(items)):
        groups.setdefault(find(i), []).append(items[i])
    tables = []
    for g in groups.values():
        H = [s for k, s in g if k == "h"]
        V = [s for k, s in g if k == "v"]
        ys = _cluster([(s[0] + s[1]) / 2 for s in H])
        xs = _cluster([(s[0] + s[1]) / 2 for s in V])
        if len(ys) < 2 or len(xs) < 2 or (len(ys) == 2 and len(xs) == 2):
            continue  # a plain box or single rule is not a table
        tables.append({"ys": ys, "xs": xs, "H": H, "V": V,
                       "bbox": [min(xs), min(ys), max(xs), max(ys)]})
    return tables, h | v


def _cluster(vals, tol=6):
    out = []
    for v in sorted(vals):
        if out and v - out[-1][-1] <= tol:
            out[-1].append(v)
        else:
            out.append([v])
    return [float(np.mean(c)) for c in out]


def _covered(segs, pos, a, b, axis_pos, tol=4):
    """Fraction of [a,b] covered by rules lying at `pos` (rules given as (p0,p1,s,e))."""
    cov = np.zeros(int(b - a) + 1, bool)
    for s in segs:
        if s[0] - tol <= pos <= s[1] + tol:
            lo, hi = max(a, s[2]), min(b, s[3])
            if hi > lo:
                cov[int(lo - a):int(hi - a)] = True
    return cov.mean() if len(cov) else 0.0


# ---------- text lines ----------

def _trim(ink, y0, y1, x0, x1):
    sub = ink[y0:y1, x0:x1]
    rows, cols = np.flatnonzero(sub.any(1)), np.flatnonzero(sub.any(0))
    if not len(rows):
        return None
    return y0 + rows[0], y0 + rows[-1] + 1, x0 + cols[0], x0 + cols[-1] + 1


def _split_tall(ink, box, th):
    """Split a band that holds several touching lines at deep minima of its ink profile."""
    y0, y1, x0, x1 = box
    if y1 - y0 < 1.7 * th:
        return [box]
    prof = ink[y0:y1, x0:x1].sum(1).astype(float)
    lo, hi = int(0.6 * th), int(y1 - y0 - 0.6 * th)
    if hi <= lo:
        return [box]
    k = lo + int(np.argmin(prof[lo:hi]))
    if prof[k] > 0.15 * np.median(prof[prof > 0]):
        return [box]
    a, b = _trim(ink, y0, y0 + k, x0, x1), _trim(ink, y0 + k, y1, x0, x1)
    return (_split_tall(ink, a, th) if a else []) + (_split_tall(ink, b, th) if b else [])


def xy_cut(ink, box, th, out):
    """Recursive XY-cut. Appends text-line boxes (y0,y1,x0,x1) to `out` in reading order.

    Order of preference: unusually large row gaps (section breaks), then a column gutter
    that runs through the whole region, then ordinary line gaps.
    """
    t = _trim(ink, *box)
    if t is None:
        return
    y0, y1, x0, x1 = t
    sub = ink[y0:y1, x0:x1]
    row_gaps = _runs(~sub.any(1))
    if len(row_gaps) >= 2:
        typical = float(np.median([b - a for a, b in row_gaps]))
        big = [(a, b) for a, b in row_gaps if b - a >= max(0.8 * th, 1.6 * typical)]
        if big and len(big) < len(row_gaps):
            cuts = [y0] + [y0 + (a + b) // 2 for a, b in big] + [y1]
            for a, b in zip(cuts, cuts[1:]):
                xy_cut(ink, (a, b, x0, x1), th, out)
            return
    gutter = _column_gutter(sub, th)
    if gutter:
        a, b = gutter
        xy_cut(ink, (y0, y1, x0, x0 + a), th, out)
        xy_cut(ink, (y0, y1, x0 + b, x1), th, out)
        return
    if not row_gaps and y1 - y0 > 1.7 * th and len(_split_tall(ink, t, th)) > 1:
        # Tall region with no blank row: something (a frame's curved end, an icon) spans every row.
        # Cut it apart on blank columns, then the pieces can split into lines again.
        cols = [(a, b) for a, b in _runs(~sub.any(0)) if b - a >= 0.5 * th]
        if cols:
            edges = [0] + [p for a, b in cols for p in (a, b)] + [sub.shape[1]]
            for a, b in zip(edges[::2], edges[1::2]):
                xy_cut(ink, (y0, y1, x0 + a, x0 + b), th, out)
            return
    if len(row_gaps) == 1 and row_gaps[0][1] - row_gaps[0][0] >= 0.8 * th:
        a, b = row_gaps[0]
        xy_cut(ink, (y0, y0 + (a + b) // 2, x0, x1), th, out)
        xy_cut(ink, (y0 + (a + b) // 2, y1, x0, x1), th, out)
        return
    cuts = [0] + [(a + b) // 2 for a, b in row_gaps] + [sub.shape[0]]
    for a, b in zip(cuts, cuts[1:]):
        line = _trim(ink, y0 + a, y0 + b, x0, x1)
        if line and line[1] - line[0] >= 3:
            out.extend(_split_tall(ink, line, th))


def _column_gutter(sub, th):
    """A blank vertical strip through the whole region separating two blocks of running text."""
    H, W = sub.shape
    if H < 1.8 * th:
        return None
    gaps = [(a, b) for a, b in _runs(~sub.any(0)) if b - a >= 1.5 * th and a >= 0.2 * W and W - b >= 0.2 * W]
    for a, b in sorted(gaps, key=lambda g: g[0] - g[1]):
        left, right = sub[:, :a], sub[:, b:]
        n = len(_runs(left.any(1))) + len(_runs(right.any(1)))
        if n >= 4 and _is_text_column(left, th) and _is_text_column(right, th):
            return a, b
    return None


def _is_text_column(side, th):
    """Running text: several lines, wide enough for a phrase, and most lines span most of the width."""
    W = side.shape[1]
    if W < 10 * th:
        return False
    rows = [r for r in _runs(side.any(1)) if r[1] - r[0] >= 0.3 * th]
    if not rows:
        return False
    fills = []
    for a, b in rows:
        cols = np.flatnonzero(side[a:b].any(0))
        fills.append((cols[-1] - cols[0] + 1) / W)
    if len(rows) <= 2:  # a short column: its first line must run the full width (wrapped prose)
        return fills[0] >= 0.85
    return float(np.median(fills)) >= 0.7


def line_segments(ink, box, th, gap=1.4):
    """Split a text line on wide horizontal gaps (table cells, tab stops)."""
    y0, y1, x0, x1 = box
    cols = ink[y0:y1, x0:x1].any(0)
    segs, start = [], None
    gaps = [(a, b) for a, b in _runs(~cols) if b - a >= gap * th]
    edges = [0] + [p for a, b in gaps for p in (a, b)] + [len(cols)]
    for s, e in zip(edges[::2], edges[1::2]):
        t = _trim(ink, y0, y1, x0 + s, x0 + e)
        if t:
            segs.append(t)
    return segs or [box]


def crop(gray, box):
    """Gray crop with a little margin, resized to the recognizer's input height."""
    y0, y1, x0, x1 = box
    h = y1 - y0
    py, px = max(2, int(0.15 * h)), max(3, int(0.3 * h))
    a = gray[max(0, y0 - py):min(gray.shape[0], y1 + py), max(0, x0 - px):min(gray.shape[1], x1 + px)]
    border = np.median(np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]))
    if border < a.mean():  # light text on a dark fill: make it dark-on-light like the training lines
        a, border = 255 - a, 255 - border
    lo = np.percentile(a, 2)
    if border - lo > 20:  # stretch so the background is white and the strokes are dark
        a = (a - lo) * (230.0 / (border - lo)) + 10
    w = max(8, min(4000, round(a.shape[1] * HEIGHT / max(1, a.shape[0]))))
    return np.asarray(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).resize((w, HEIGHT), Image.BILINEAR))


# ---------- whole page ----------

def text_blocks(text_ink, th, read, figures=True):
    """Text mask -> grouped blocks. read(boxes) -> [(text, confidence)] supplies the words for each box."""
    lines = []
    xy_cut(text_ink, (0, text_ink.shape[0], 0, text_ink.shape[1]), th, lines)
    lines = [ln for ln in lines if (ln[1] - ln[0]) >= 0.35 * th or (ln[3] - ln[2]) >= 0.35 * th]
    if lines:
        th = float(np.median([ln[1] - ln[0] for ln in lines]))
    figs, text_lines = [], []
    for ln in lines:
        h, w = ln[1] - ln[0], ln[3] - ln[2]
        if figures and h > 4 * th and (text_ink[ln[0]:ln[1], ln[2]:ln[3]].mean() > 0.35 or w < 2 * h):
            figs.append({"kind": "figure", "box": ln})
        else:
            text_lines.append(ln)
    segs = [line_segments(text_ink, ln, th) for ln in text_lines]
    flat = [s for ss in segs for s in ss]
    got = read(flat) if flat else []
    blocks, k = [], 0
    for ln, ss in zip(text_lines, segs):
        blocks.append({"kind": "line", "box": ln, "segs": ss, "parts": got[k:k + len(ss)],
                       "xh": core_height(text_ink, ln)})
        k += len(ss)
    return blocks + figs, th


def core_height(ink, box):
    """Rows holding most of a line's ink (about the x-height): tracks font size better than the
    full ink box, which depends on whether a line happens to contain ascenders and descenders."""
    y0, y1, x0, x1 = box
    prof = ink[y0:y1, x0:x1].sum(1)
    return int((prof >= 0.4 * prof.max()).sum()) if prof.size and prof.max() > 0 else y1 - y0


def analyze(img, recognizer, deskew=True):
    """Image -> list of blocks: heading / paragraph / list_item / table / figure (bbox in original pixels)."""
    gray, ink, scale, angle = prepare(img, deskew)
    th = text_height(ink)
    tables, rules = find_tables(ink, th)
    H, W = ink.shape
    frames = _open_1d(ink, max(40, W // 8), 1) | _open_1d(ink, max(40, H // 8), 0)  # boxes, banners, page rules
    clean = remove_frames(ink & ~rules & ~frames, th)
    text_ink = clean.copy()
    for t in tables:  # table interiors are read cell by cell, not as running text
        x0, y0, x1, y1 = (int(v) for v in t["bbox"])
        text_ink[max(0, y0 - 2):y1 + 3, max(0, x0 - 2):x1 + 3] = False

    def ocr(boxes):
        return recognizer.read([crop(gray, b) for b in boxes])
    th = min(th, text_height(text_ink))  # re-estimate without frames and tables
    blocks, th = text_blocks(text_ink, th, ocr)
    for t in tables:
        blocks.append(read_grid(t, clean, th, ocr))
    out = group(blocks, th)
    for b in out:
        y0, y1, x0, x1 = b.pop("box")
        b["bbox"] = [round(x0 / scale), round(y0 / scale), round(x1 / scale), round(y1 / scale)]
    meta = {"deskew_degrees": angle, "upscaled": scale, "text_height_px": round(th / scale, 1)}
    return out, meta


def read_grid(t, clean, th, read):
    nr, nc = len(t["ys"]) - 1, len(t["xs"]) - 1
    cells, jobs = [], []
    for r in range(nr):
        for c in range(nc):
            y0, y1 = int(t["ys"][r]) + 3, int(t["ys"][r + 1]) - 2
            x0, x1 = int(t["xs"][c]) + 3, int(t["xs"][c + 1]) - 2
            sub = []
            if y1 > y0 and x1 > x0:
                xy_cut(clean, (y0, y1, x0, x1), th, sub)
            cells.append((r, c, len(jobs), len(sub)))
            jobs += sub
    got = read(jobs) if jobs else []
    grid = [["" for _ in range(nc)] for _ in range(nr)]
    for r, c, k, n in cells:
        grid[r][c] = " ".join(g[0] for g in got[k:k + n] if g[0])
    _apply_spans(t, grid)
    confs = [g[1] for g in got if g[1] is not None]
    x0, y0, x1, y1 = t["bbox"]
    return {"kind": "table", "box": (int(y0), int(y1), int(x0), int(x1)), "rows": grid,
            "confidence": round(float(np.mean(confs)), 4) if confs else None}


def _apply_spans(t, grid):
    """Cells whose dividing rule is missing are merged: the text stays in the first cell."""
    ys, xs = t["ys"], t["xs"]
    for r in range(len(grid)):
        for c in range(1, len(grid[0])):
            if _covered(t["V"], xs[c], ys[r] + 2, ys[r + 1] - 2, 0) < 0.5 and not grid[r][c]:
                grid[r][c] = None  # continuation of a merged cell
    for r in range(1, len(grid)):
        for c in range(len(grid[0])):
            if _covered(t["H"], ys[r], xs[c] + 2, xs[c + 1] - 2, 1) < 0.5 and not grid[r][c]:
                grid[r][c] = None
    for row in grid:
        for c, v in enumerate(row):
            if v is None:
                row[c] = ""


def group(blocks, th):
    """Lines -> headings, paragraphs, list items and borderless tables, keeping reading order.

    Lines arrive in XY-cut order (columns already respected); tables and figures are
    inserted before the first text block that starts below them.
    """
    lines = [b for b in blocks if b["kind"] == "line"]
    gaps = [b["box"][0] - a["box"][1] for a, b in zip(lines, lines[1:])]
    gaps = [g for g in gaps if 0 <= g < 2.5 * th]
    lead = float(np.median(gaps)) if gaps else 0.5 * th  # usual space between lines of one paragraph
    xh = float(np.median([b["xh"] for b in lines])) if lines else th
    others = [b for b in blocks if b["kind"] != "line"]
    deflist = _definition_lines(lines, th)
    out, i = [], 0
    while i < len(lines):
        ln = lines[i]
        # Borderless table: a run of lines that each split into 2+ aligned segments.
        j = i
        while j < len(lines) and len(lines[j]["segs"]) >= 2:
            j += 1
        if j - i >= 2:
            table = _borderless(lines[i:j])
            if table:
                out.append(table)
                i = j
                continue
            for ln2 in lines[i:j]:  # not aligned like a table: keep as text lines
                ln2["segs"], ln2["parts"] = [ln2["box"]], [(" ".join(p[0] for p in ln2["parts"] if p[0]),
                                                            _mean([p[1] for p in ln2["parts"]]))]
            # fall through and handle line i as text
        text = " ".join(p[0] for p in ln["parts"] if p[0])
        conf = [p[1] for p in ln["parts"]]
        if not text or (len(text) <= 2 and (_mean(conf) or 1) < 0.6):
            i += 1  # specks, frame corners, icons read as one or two unsure characters
            continue
        h = ln["box"][1] - ln["box"][0]
        size = ln["xh"] / max(1.0, xh)  # font size relative to body text
        prev = out[-1] if out else None
        gap = ln["box"][0] - prev["box"][1] if prev else 1e9
        big = size >= 1.22 or (size >= 1.1 and h >= 1.3 * th)
        if big and len(text) < 120:
            level = 1 if size >= 1.7 else 2 if size >= 1.32 else 3
            if prev and prev["type"] == "heading" and prev["level"] == level and 0 <= gap < 1.0 * h:
                _extend(prev, ln, text, conf)
            else:
                out.append({"type": "heading", "level": level, "text": text, "box": ln["box"], "conf": conf})
        elif LIST_RE.match(text) or i in deflist:
            out.append({"type": "list_item", "text": text, "box": ln["box"], "conf": conf})
        elif (prev and prev["type"] in ("paragraph", "list_item") and -0.5 * th < gap < max(0.9 * th, 1.35 * lead)
              and abs(ln["box"][2] - prev["box"][2]) < 2.5 * th and not big):
            _extend(prev, ln, text, conf)
        else:
            out.append({"type": "paragraph", "text": text, "box": ln["box"], "conf": conf})
        i += 1
    for o in others:
        if o["kind"] == "figure":
            o = {"type": "figure", "box": o["box"], **({"text": o["text"]} if o.get("text") else {})}
        elif o["kind"] == "code":
            o = {"type": "code", "text": o["text"], "box": o["box"]}
        else:
            o = {"type": "table", "rows": o["rows"], "box": o["box"], "confidence": o["confidence"]}
        k = next((n for n, b in enumerate(out) if b["box"][0] > o["box"][0] and b["box"][2] < o["box"][3]), len(out))
        out.insert(k, o)
    for b in out:
        if "lines" in b:
            b["text"] = _join_lines(b.pop("lines"))
        if "conf" in b:
            c = [v for v in b.pop("conf") if v is not None]
            b["confidence"] = round(float(np.mean(c)), 4) if c else None
    return out


def _definition_lines(lines, th):
    """Indices of lines in runs of 3+ "Term: description" lines that start at the same x.
    Two such lines (e.g. "Name: ..." and "Date: ...") stay ordinary text."""
    out, i = set(), 0
    while i < len(lines):
        j = i
        while (j < len(lines) and DEF_RE.match(" ".join(p[0] for p in lines[j]["parts"] if p[0]))
               and abs(lines[j]["box"][2] - lines[i]["box"][2]) < 0.5 * th):
            j += 1
        if j - i >= 3:
            out.update(range(i, j))
        i = max(j, i + 1)
    return out


def _extend(block, ln, text, conf):
    y0, y1, x0, x1 = block["box"]
    b = ln["box"]
    block["box"] = (min(y0, b[0]), max(y1, b[1]), min(x0, b[2]), max(x1, b[3]))
    block.setdefault("lines", [(block["text"], y1 - y0 and x1 - x0)]).append((text, b[3] - b[2]))
    block["conf"] += conf


def _join_lines(lines):
    """Join wrapped lines with spaces but keep real line breaks (code, addresses, poems).

    A wrapped line ends only because the next word did not fit; if the next line's first
    word would have fitted after this line, the break was deliberate.
    """
    widest = max(w for _, w in lines)
    out = lines[0][0]
    for (prev, w), (text, _) in zip(lines, lines[1:]):
        per_char = w / max(1, len(prev))
        first = text.split()[0] if text.split() else ""
        if w + per_char * (len(first) + 1) < 0.97 * widest:
            out += "\n" + text
        elif prev[-2:-1].islower() and prev.endswith("-") and text[:1].islower():
            out = out[:-1] + text  # rejoin a hyphenated word
        else:
            out += " " + text
    return out


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return round(float(np.mean(vals)), 4) if vals else None


def _borderless(lines):
    """Rows of aligned segments -> a table, or None when the segments do not line up in columns."""
    x0 = min(ln["box"][2] for ln in lines)
    x1 = max(ln["box"][3] for ln in lines)
    occupied = np.zeros(x1 - x0 + 1, bool)
    for ln in lines:
        for s in ln["segs"]:
            occupied[s[2] - x0:s[3] - x0] = True
    spans = _runs(occupied)
    if len(spans) < 2:
        return None
    rows, conf, fill = [], [], []
    for ln in lines:
        row, hit = [""] * len(spans), [False] * len(spans)
        for s, (t, c) in zip(ln["segs"], ln["parts"]):
            mid = (s[2] + s[3]) / 2 - x0
            k = next((n for n, (a, b) in enumerate(spans) if a <= mid <= b), len(spans) - 1)
            row[k] = (row[k] + " " + t).strip()
            hit[k] = True
            conf.append(c)
        fill.append(sum(hit) / len(spans))
        rows.append(row)
    if np.mean(fill) < 0.6:  # e.g. code with alignment spaces: many sparse "columns"
        return None
    box = (min(ln["box"][0] for ln in lines), max(ln["box"][1] for ln in lines), x0, x1)
    return {"type": "table", "rows": rows, "box": box, "conf": conf}
