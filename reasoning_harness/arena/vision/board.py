"""Find a square puzzle board in a screenshot, estimate its grid size, cut out cells."""
import cv2
import numpy as np

from .render import SIZE

WARP = 480


def load(data):
    """bytes / path / PIL / ndarray -> RGB uint8 array."""
    if isinstance(data, np.ndarray):
        return data[..., :3].copy()
    if isinstance(data, (bytes, bytearray)):
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:  # e.g. GIF, which OpenCV cannot decode
            import io
            from PIL import Image
            return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    if hasattr(data, "convert"):
        return np.asarray(data.convert("RGB"))
    from PIL import Image
    return np.asarray(Image.open(data).convert("RGB"))


def _order(pts):
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], np.float32)


def warp(rgb, quad, size=WARP):
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], np.float32)
    m = cv2.getPerspectiveTransform(_order(quad), dst)
    return cv2.warpPerspective(rgb, m, (size, size), flags=cv2.INTER_AREA)


def _profiles(board):
    g = cv2.cvtColor(board, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    return gx.mean(0), gy.mean(1)


def gridness(board, n):
    """How strongly edges line up with every internal boundary of an n x n grid."""
    px, py = _profiles(board)
    s = board.shape[0]
    win = max(2, s // (n * 10))
    scores = []
    for prof in (px, py):
        base = np.median(prof) + 0.02 * prof.max() + 2.0  # floor: flat colour fields have ~0 gradient
        peaks = [prof[max(0, int(round(k * s / n)) - win):int(round(k * s / n)) + win + 1].max() / base for k in range(1, n)]
        # also require the mid-cell positions to be quieter than the boundaries
        mids = [prof[max(0, int(round((k + .5) * s / n)) - win):int(round((k + .5) * s / n)) + win + 1].max() / base
                for k in range(n)]
        scores.append(np.percentile(peaks, 20) - 0.5 * np.median(mids))
    return float(min(scores))


def detect_n(board, sizes):
    sc = {n: gridness(board, n) for n in sizes}
    best = max(sc.values())
    good = [n for n, v in sc.items() if v >= best - 0.25 * abs(best)]
    return max(good), sc


def find_board(rgb, sizes):
    """Return (warped square board, quad, n) for the most grid-like large square region."""
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cands = [np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)]
    for lo, hi in ((30, 90), (60, 160)):
        edges = cv2.dilate(cv2.Canny(gray, lo, hi), np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < 0.01 * h * w:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if not 0.8 < bw / bh < 1.25:
                continue
            approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                cands.append(approx.reshape(4, 2).astype(np.float32))
            cands.append(np.array([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]], np.float32))
    scored = []
    for q in cands:
        side = np.sqrt(cv2.contourArea(_order(q)))
        if side < 60:
            continue
        b = warp(rgb, q)
        n, sc = detect_n(b, sizes)
        oq = _order(q)
        bw = (np.linalg.norm(oq[1] - oq[0]) + np.linalg.norm(oq[2] - oq[3])) / 2
        bh = (np.linalg.norm(oq[3] - oq[0]) + np.linalg.norm(oq[2] - oq[1])) / 2
        squareness = np.exp(-5 * abs(np.log(bw / bh)))  # puzzle boards are square
        scored.append((sc[n] * squareness, side, q, n, b))
    # strongly grid-like and large: a sub-block of the board scores lower on size,
    # the whole screenshot scores lower on grid alignment
    pick = max(scored, key=lambda s: max(s[0], 0) * np.sqrt(s[1]) + 1e-6 * s[1])
    board, q, n = pick[4], pick[2], pick[3]
    return _trim(board, n), q, n


def _trim(board, n):
    """Drop a thick outer frame if the warp included one, by re-warping to the inner grid."""
    return board


def cell(board, n, r, c, inset=0.06, size=SIZE):
    s = board.shape[0] / n
    y0, y1 = int(r * s + inset * s), int((r + 1) * s - inset * s)
    x0, x1 = int(c * s + inset * s), int((c + 1) * s - inset * s)
    return cv2.resize(board[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)


def cells(board, n, inset=0.06):
    return [cell(board, n, r, c, inset) for r in range(n) for c in range(n)]


def edge_crop(board, n, r, c, down, half=0.3, size=SIZE):
    """Crop centred on the border between (r,c) and its right (down=False) or lower neighbour."""
    s = board.shape[0] / n
    cy, cx = ((r + 1) * s, (c + .5) * s) if down else ((r + .5) * s, (c + 1) * s)
    h = half * s
    y0, y1, x0, x1 = int(cy - h), int(cy + h), int(cx - h), int(cx + h)
    y0, x0 = max(0, y0), max(0, x0)
    return cv2.resize(board[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)


def cell_colour(board, n, r, c):
    s = board.shape[0] / n
    patch = board[int((r + .3) * s):int((r + .7) * s), int((c + .3) * s):int((c + .7) * s)]
    return np.median(patch.reshape(-1, 3), 0)


def edge_strength(board, n, r, c, down):
    """Contrast of the border strip between two cells, relative to the board background."""
    g = cv2.cvtColor(board, cv2.COLOR_RGB2GRAY).astype(np.float32)
    s = board.shape[0] / n
    w = max(2, int(0.07 * s))
    if down:
        y = int((r + 1) * s)
        strip = g[y - w:y + w, int((c + .25) * s):int((c + .75) * s)]
    else:
        x = int((c + 1) * s)
        strip = g[int((r + .25) * s):int((r + .75) * s), x - w:x + w]
    bg = np.median(g)
    return float(np.abs(strip - bg).mean())
