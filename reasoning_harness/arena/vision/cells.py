"""Small CNN cell classifiers, trained from scratch on synthetic cells (render.py)."""
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .render import SETS, SIZE, dataset


class CellNet(nn.Module):
    def __init__(self, classes, width=48):
        super().__init__()
        self.config = {"classes": classes, "width": width}
        w = width
        self.body = nn.Sequential(
            nn.Conv2d(3, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(),
            nn.Conv2d(w, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(w, 2 * w, 3, padding=1), nn.BatchNorm2d(2 * w), nn.ReLU(),
            nn.Conv2d(2 * w, 2 * w, 3, padding=1), nn.BatchNorm2d(2 * w), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(2 * w, 4 * w, 3, padding=1), nn.BatchNorm2d(4 * w), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Dropout(0.2), nn.Linear(4 * w * (SIZE // 8) ** 2, 256), nn.ReLU(), nn.Linear(256, classes))

    def forward(self, x):
        return self.body(x)


def to_tensor(batch):
    """uint8 (B,H,W,3) -> normalised float (B,3,H,W)."""
    x = torch.from_numpy(np.ascontiguousarray(batch)).float().permute(0, 3, 1, 2) / 255.0
    return (x - 0.5) / 0.25


def _make(args):
    kind, count, seed = args
    return dataset(kind, count, seed)


def build(kind, count, seed, workers=4):
    from multiprocessing import Pool
    chunk = max(1, count // (workers * 4))
    jobs = [(kind, min(chunk, count - i), seed * 1000 + k) for k, i in enumerate(range(0, count, chunk))]
    with Pool(workers) as pool:
        parts = pool.map(_make, jobs)
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def train(kind, out_dir, count=40000, epochs=6, seed=0, workers=4, log=print):
    torch.manual_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    xs, ys = build(kind, count, seed + 1, workers)
    vx, vy = build(kind, 3000, seed + 99, workers)  # fresh synthetic samples, different seeds
    net = CellNet(len(SETS[kind]))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = epochs * (len(xs) // 128)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
    hist = []
    for ep in range(epochs):
        net.train()
        perm = np.random.default_rng(seed + ep).permutation(len(xs))
        for i in range(0, len(perm) - 127, 128):
            idx = perm[i:i + 128]
            loss = F.cross_entropy(net(to_tensor(xs[idx])), torch.from_numpy(ys[idx]), label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            pred = torch.cat([net(to_tensor(vx[i:i + 512])).argmax(1) for i in range(0, len(vx), 512)]).numpy()
        acc = float((pred == vy).mean())
        hist.append({"epoch": ep + 1, "loss": round(loss.item(), 4), "synthetic_val_acc": round(acc, 4)})
        log(f"{kind} {hist[-1]}")
    torch.save({"config": net.config, "state": net.state_dict(), "labels": SETS[kind]}, out_dir / f"{kind}.pt")
    info = {"kind": kind, "train_samples": int(len(xs)), "epochs": epochs, "parameters": sum(p.numel() for p in net.parameters()),
            "history": hist, "train_seconds": round(time.time() - t0, 1)}
    (out_dir / f"{kind}.json").write_text(json.dumps(info, indent=2))
    return net, info


class Classifier:
    def __init__(self, path):
        data = torch.load(path, map_location="cpu", weights_only=True)
        self.labels = data["labels"]
        self.net = CellNet(**data["config"])
        self.net.load_state_dict(data["state"])
        self.net.eval()

    def probs(self, cells):
        """cells: list of uint8 (SIZE,SIZE,3) arrays -> (N, classes) probabilities."""
        if not len(cells):
            return np.zeros((0, len(self.labels)))
        with torch.inference_mode():
            return torch.softmax(self.net(to_tensor(np.stack(cells))), 1).numpy()

    def predict(self, cells):
        p = self.probs(cells)
        return [self.labels[i] for i in p.argmax(1)], p.max(1) if len(p) else np.zeros(0)
