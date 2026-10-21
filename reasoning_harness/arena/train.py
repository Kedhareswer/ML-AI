"""Train a GridNet from random initialisation on solved puzzles.

Each example is a random partially-solved state of a training puzzle; the
network predicts the missing part of the unique solution. Checkpoints are
resumable (model, optimiser, RNG and step).
"""
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .nets import GridNet, save_net
from .registry import from_json


def make_batch(items, rng):
    xs, ms, rs, ts, ls = [], [], [], [], []
    for puzzle, solution in items:
        state = puzzle.random_partial(solution, rng)
        x, mask, regions = puzzle.tensors(state)
        target, lmask = puzzle.labels(solution, state)
        p, n = puzzle.pad, puzzle.n
        t = np.zeros((p, p), np.int64)
        lm = np.zeros((p, p), np.float32)
        t[:n, :n], lm[:n, :n] = target, lmask
        xs.append(x), ms.append(mask), rs.append(regions), ts.append(t), ls.append(lm)
    return [torch.from_numpy(np.stack(a)) for a in (xs, ms, rs, ts, ls)]


def loss_fn(kind_loss, out, mask, target, lmask):
    if kind_loss == "spatial":
        logits = out[:, 0].flatten(1).masked_fill(mask[:, 0].flatten(1) == 0, -1e9)
        idx = target.flatten(1).argmax(1)
        loss = F.cross_entropy(logits, idx)
        acc = (logits.argmax(1) == idx).float().mean()
        return loss, acc
    ce = F.cross_entropy(out, target, reduction="none")
    denom = lmask.sum().clamp(min=1)
    loss = (ce * lmask).sum() / denom
    acc = ((out.argmax(1) == target).float() * lmask).sum() / denom
    return loss, acc


def train(rows, run, steps=3000, batch_size=64, width=96, blocks=8, lr=1e-3, seed=0,
          resume=False, threads=2, val_rows=None, log_every=50, ckpt_every=250, bf16=True):
    torch.set_num_threads(threads)
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    cache = {}

    def item(i, pool):
        key = (id(pool), i)
        if key not in cache:
            d = pool[i]
            cache[key] = (from_json(d), d["solution"])
        return cache[key]

    probe = item(0, rows)[0]
    kind_loss = probe.loss
    ckpt = run / "last.pt"
    if resume and ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = state["config"]
        torch.manual_seed(cfg["seed"])
        net = GridNet(**state["net_config"])
        net.load_state_dict(state["model"])
        opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=0.01)
        opt.load_state_dict(state["opt"])
        rng = state["rng"]
        torch.set_rng_state(state["torch_rng"])
        start = state["step"]
    else:
        if (run / "weights.pt").exists() and not resume:
            raise SystemExit(f"{run} already holds a trained model; use a fresh directory or --resume")
        cfg = {"kind": probe.name, "steps": steps, "batch_size": batch_size, "lr": lr, "seed": seed,
               "train_puzzles": len(rows), "bf16_autocast": bf16}
        torch.manual_seed(seed)
        net = GridNet(probe.in_ch, probe.out_ch, width, blocks)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
        rng = np.random.default_rng(seed)
        start = 0
        (run / "metrics.jsonl").write_text("")
    cfg["steps"] = steps
    py_rng = random.Random(seed + start)
    t0 = time.time()

    def save(step):
        torch.save({"config": cfg, "net_config": net.config, "model": net.state_dict(), "opt": opt.state_dict(),
                    "rng": rng, "torch_rng": torch.get_rng_state(), "step": step}, run / "last.tmp")
        (run / "last.tmp").replace(ckpt)

    net.train()
    step = start
    try:
        for step in range(start + 1, steps + 1):
            frac = step / steps
            for g in opt.param_groups:  # warmup then cosine decay
                g["lr"] = cfg["lr"] * min(1.0, step / 100) * 0.5 * (1 + math.cos(math.pi * frac))
            idx = rng.integers(0, len(rows), batch_size)
            x, mask, regions, target, lmask = make_batch([item(int(i), rows) for i in idx], rng)
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
                out = net(x, mask, regions)
            loss, acc = loss_fn(kind_loss, out.float(), mask, target, lmask)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            if step % log_every == 0 or step == steps:
                rec = {"step": step, "loss": round(loss.item(), 4), "acc": round(acc.item(), 4),
                       "lr": opt.param_groups[0]["lr"], "elapsed_s": round(time.time() - t0, 1)}
                if val_rows and (step % (log_every * 4) == 0 or step == steps):
                    net.eval()
                    with torch.no_grad():
                        vrng = np.random.default_rng(12345)
                        vb = make_batch([item(i, val_rows) for i in range(min(256, len(val_rows)))], vrng)
                        vl, va = loss_fn(kind_loss, net(*vb[:3]), vb[1], vb[3], vb[4])
                    rec.update(val_loss=round(vl.item(), 4), val_acc=round(va.item(), 4))
                    net.train()
                with open(run / "metrics.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(json.dumps(rec), flush=True)
            if step % ckpt_every == 0:
                save(step)
    except KeyboardInterrupt:
        save(step)
        raise
    save(step)
    net.eval()
    save_net(net, run / "weights.pt", {"kind": cfg["kind"], "trained_steps": step, "train_puzzles": len(rows),
                                       "seed": cfg["seed"]})
    info = {**cfg, "trained_steps": step, "parameters": net.parameters_count(),
            "weights_bytes": (run / "weights.pt").stat().st_size, "train_seconds": round(time.time() - t0, 1),
            "torch": torch.__version__, "device": "cpu", "initialisation": "random (torch.manual_seed)"}
    (run / "manifest.json").write_text(json.dumps(info, indent=2))
    return net, info
