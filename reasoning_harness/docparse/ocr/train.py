"""Train the line recognizer from scratch on freshly rendered synthetic lines.

Each round renders a new set of lines (new seeds), so the network never sees the
same image twice. A checkpoint is written after every round; --resume continues.
"""
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .charset import encode
from .metrics import summary
from .model import LineOCR, Recognizer, batch_tensor
from .render import dataset


def _make(args):
    count, seed, split, degrade = args
    return dataset(count, seed, split, degrade)


def render(count, seed, split="train", degrade=True, workers=4):
    chunk = 2000
    jobs = [(min(chunk, count - i), seed * 10_007 + k, split, degrade) for k, i in enumerate(range(0, count, chunk))]
    with Pool(workers) as pool:
        parts = pool.map(_make, jobs)
    return [a for p in parts for a in p[0]], [t for p in parts for t in p[1]]


def batches(images, batch_size, rng):
    """Group lines of similar width (less padding), then shuffle the groups."""
    order = rng.permutation(len(images))
    groups = []
    for s in range(0, len(order), batch_size * 64):
        block = sorted(order[s:s + batch_size * 64], key=lambda i: images[i].shape[1])
        groups += [block[k:k + batch_size] for k in range(0, len(block), batch_size)]
    rng.shuffle(groups)
    return [g for g in groups if len(g) == batch_size]


def evaluate(net, images, texts):
    net.eval()
    rec = Recognizer.__new__(Recognizer)
    rec.net, rec.device = net, "cpu"
    preds = [t for t, _ in rec.read(images, batch=64)]
    net.train()
    return summary(preds, texts)


def train(run, rounds=4, per_round=120_000, batch_size=48, lr=2e-3, seed=0, threads=4, resume=False, log=print):
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    net = LineOCR()
    steps_per_round = per_round // batch_size
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=rounds * steps_per_round, pct_start=0.08)
    start, hist = 0, []
    ckpt = run / "last.pt"
    if resume and ckpt.exists():
        c = torch.load(ckpt, map_location="cpu", weights_only=False)
        net.load_state_dict(c["state"])
        opt.load_state_dict(c["opt"])
        sched.load_state_dict(c["sched"])
        start, hist = c["round"], c["history"]
        log(f"resumed after round {start}")
    val = render(3000, 900_001, "train")       # training fonts and text, unseen seeds
    test = render(3000, 900_002, "test")       # held-out fonts and held-out text
    t0 = time.time()
    for rnd in range(start, rounds):
        images, texts = render(per_round, seed * 100 + rnd + 1, "train")
        labels = [torch.tensor(encode(t), dtype=torch.long) for t in texts]
        rng = np.random.default_rng(seed * 100 + rnd)
        net.train()
        running = []
        for step, idx in enumerate(batches(images, batch_size, rng)[:steps_per_round]):
            x, lens = batch_tensor([images[i] for i in idx])
            logp = net(x).transpose(0, 1)  # (T,B,C)
            tgt = [labels[i] for i in idx]
            loss = F.ctc_loss(logp, torch.cat(tgt), torch.tensor(lens), torch.tensor([len(t) for t in tgt]),
                              blank=0, zero_infinity=True)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
            running.append(loss.item())
            if step % 250 == 0:
                log(f"round {rnd + 1} step {step}/{steps_per_round} loss {np.mean(running[-250:]):.4f} "
                    f"lr {sched.get_last_lr()[0]:.2e} elapsed {time.time() - t0:.0f}s")
        entry = {"round": rnd + 1, "train_loss": round(float(np.mean(running[-500:])), 4),
                 "val_train_fonts": evaluate(net, *val), "test_heldout_fonts": evaluate(net, *test),
                 "elapsed_s": round(time.time() - t0, 1)}
        hist.append(entry)
        log(json.dumps(entry))
        torch.save({"state": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "round": rnd + 1, "history": hist}, ckpt)
        del images, texts, labels
    info = {"parameters": sum(p.numel() for p in net.parameters()), "rounds": rounds, "lines_per_round": per_round,
            "batch_size": batch_size, "seed": seed, "history": hist}
    torch.save({"config": net.config, "state": net.state_dict(), "info": info}, run / "ocr.pt")
    (run / "train_info.json").write_text(json.dumps(info, indent=2))
    return net, info
