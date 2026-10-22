"""Characters the OCR network can emit. Index 0 is the CTC blank."""
import unicodedata

CHARS = "".join(chr(c) for c in range(32, 127)) + "•€£°©"
BLANK = 0
INDEX = {c: i + 1 for i, c in enumerate(CHARS)}
CLASSES = len(CHARS) + 1

_MAP = {"‘": "'", "’": "'", "‚": "'", "“": '"', "”": '"', "„": '"',
        "–": "-", "—": "-", "−": "-", "…": "...", " ": " ", "\t": " ",
        "·": "•", "●": "•", "•": "•", "×": "x"}


def normalize(text):
    """Map text into CHARS: fold typographic punctuation and accents, drop the rest."""
    out = []
    for ch in text:
        ch = _MAP.get(ch, ch)
        if all(c in INDEX for c in ch):
            out.append(ch)
            continue
        base = unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode()
        out.append(base if base and all(c in INDEX for c in base) else "")
    return "".join(out)


def encode(text):
    return [INDEX[c] for c in text if c in INDEX]


def decode_greedy(ids):
    """Collapse repeats, drop blanks."""
    out, prev = [], BLANK
    for i in ids:
        if i != prev and i != BLANK:
            out.append(CHARS[i - 1])
        prev = i
    return "".join(out)
