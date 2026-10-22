"""Train every cell classifier from scratch on synthetic cells."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from arena.vision.cells import train  # noqa: E402
from arena.vision.render import SETS  # noqa: E402

COUNTS = {"digits": 30000, "zipnum": 62000, "tango": 30000, "tango_sign": 15000, "letters": 54000,
          "patches_num": 74000, "patches_shape": 20000, "chess": 52000, "checkers": 15000}
out = sys.argv[1] if len(sys.argv) > 1 else "artifacts/arena/vision"
kinds = sys.argv[2:] or list(SETS)
torch.set_num_threads(4)
for k in kinds:
    train(k, out, COUNTS[k], epochs=6, seed=0)
