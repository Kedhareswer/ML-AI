"""Train the chess policy from random initialisation on Lichess puzzle positions."""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .chess_net import ChessNet, planes


def train(data, run, steps=6000, batch_size=256, width=96, blocks=6, lr=1e-3, seed=0, threads=2, resume=False):
    torch.set_num_threads(threads)
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    d = np.load(data)
    codes, flags, moves = d["codes"], d["flags"], d["moves"]
    n_val = 20000
    val = slice(len(moves) - n_val, len(moves))
    n_train = len(moves) - n_val
    torch.manual_seed(seed)
    net = ChessNet(width, blocks)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    start = 0
    ckpt = run / "last.pt"
    if resume and ckpt.exists():
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        net.load_state_dict(st["model"]), opt.load_state_dict(st["opt"])
        rng, start = st["rng"], st["step"]
    elif (run / "weights.pt").exists():
        raise SystemExit(f"{run} already holds a model; use --resume or a fresh directory")
    else:
        (run / "metrics.jsonl").write_text("")
    t0 = time.time()
    net.train()
    for step in range(start + 1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr * min(1, step / 200) * 0.5 * (1 + math.cos(math.pi * step / steps))
        idx = rng.integers(0, n_train, batch_size)
        x = planes(codes[idx], flags[idx])
        y = torch.from_numpy(moves[idx])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits = net(x)
        loss = F.cross_entropy(logits.float(), y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == steps:
            rec = {"step": step, "loss": round(loss.item(), 4), "elapsed_s": round(time.time() - t0, 1)}
            if step % 500 == 0 or step == steps:
                net.eval()
                with torch.no_grad():
                    vi = np.arange(val.start, val.start + 4096)
                    vl = net(planes(codes[vi], flags[vi]))
                    vy = torch.from_numpy(moves[vi])
                    rec["val_loss"] = round(F.cross_entropy(vl, vy).item(), 4)
                    rec["val_top1"] = round((vl.argmax(1) == vy).float().mean().item(), 4)
                net.train()
            with open(run / "metrics.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
        if step % 500 == 0:
            torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "rng": rng, "step": step}, run / "last.tmp")
            (run / "last.tmp").replace(ckpt)
    net.eval()
    torch.save({"config": net.config, "state": net.state_dict()}, run / "weights.pt")
    info = {"kind": "chess_policy", "steps": steps, "batch_size": batch_size, "train_positions": int(n_train),
            "parameters": sum(p.numel() for p in net.parameters()), "seed": seed,
            "train_seconds": round(time.time() - t0, 1), "initialisation": "random (torch.manual_seed)",
            "data": "Lichess puzzle database (CC0), split by source game"}
    (run / "manifest.json").write_text(json.dumps(info, indent=2))
    return net


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    train(a.data, a.run, a.steps, threads=a.threads, resume=a.resume)
