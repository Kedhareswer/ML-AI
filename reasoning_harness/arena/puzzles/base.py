"""Shared plumbing for grid puzzles: padding, region tensors, log-probabilities."""
import numpy as np


def log_softmax(x, axis=0):
    x = x - x.max(axis=axis, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=axis, keepdims=True))


class GridPuzzle:
    name = "grid"
    pad = 9          # network canvas size
    max_regions = 1  # region pooling slots
    loss = "cell"    # "cell": per-cell classes; "spatial": one choice among cells

    def region_map(self):
        """(n,n) region ids, or None for a single whole-board region."""
        return None

    def tensors(self, state):
        n, p = self.n, self.pad
        planes = self.encode(state)
        x = np.zeros((planes.shape[0], p, p), np.float32)
        x[:, :n, :n] = planes
        mask = np.zeros((1, p, p), np.float32)
        mask[:, :n, :n] = 1
        regions = np.zeros((self.max_regions, p * p), np.float32)
        rmap = self.region_map()
        for r in range(n):
            for c in range(n):
                regions[0 if rmap is None else rmap[r, c], r * p + c] = 1
        return x, mask, regions

    def normalise(self, out):
        out = out[:, :self.n, :self.n]
        if self.loss == "spatial":
            flat = out[0].reshape(-1)
            return log_softmax(flat).reshape(self.n, self.n)[None]
        return log_softmax(out, 0)

    def labels(self, solution, state):
        """(target (n,n) int64, loss mask (n,n) float32) for training."""
        raise NotImplementedError
