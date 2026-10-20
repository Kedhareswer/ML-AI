from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from .model import BOS, EOS, PAD, ModelConfig, Planner, encode
from .tasks import MOVES, example, generate, verify

def device_for(name="auto"):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if name not in ("cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    return name

def atomic_save(value, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)

def batch(examples, device, context):
    sequences, labels = [], []
    for prompt, answer in examples:
        prefix = [BOS] + encode(prompt)
        tail = encode(answer) + [EOS]
        full = prefix + tail
        if len(full) - 1 > context:
            raise ValueError("Training example exceeds context; increase context or reduce maze size")
        sequences.append(full[:-1])
        labels.append(([-100] * len(prefix) + tail)[1:])
    length = max(map(len, sequences))
    x = torch.full((len(sequences), length), PAD, dtype=torch.long)
    y = torch.full_like(x, -100)
    for i, (s, label) in enumerate(zip(sequences, labels)):
        x[i, :len(s)] = torch.tensor(s)
        y[i, :len(label)] = torch.tensor(label)
    return x.to(device), y.to(device)

def train(run, config=None, steps=5000, batch_size=32, lr=3e-4, size=4,
          train_examples=2048, seed=42, device="auto", resume=False, save_every=200):
    if min(steps, batch_size, train_examples, save_every) < 1 or lr <= 0:
        raise ValueError("steps, batch size, examples, checkpoint interval and learning rate must be positive")
    run, device = Path(run), device_for(device)
    run.mkdir(parents=True, exist_ok=True)
    checkpoint = None
    if resume:
        checkpoint = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
        config = ModelConfig(**checkpoint["config"])
        settings = checkpoint["settings"]
        batch_size, lr, size, train_examples, seed = (settings[k] for k in ["batch_size", "lr", "size", "train_examples", "seed"])
    else:
        if any(run.iterdir()):
            raise ValueError("Run directory is not empty; use a new directory or --resume")
        config = config or ModelConfig()
        settings = dict(batch_size=batch_size, lr=lr, size=size, train_examples=train_examples, seed=seed)
    torch.manual_seed(seed)
    model = Planner(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
    start = 0
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start = checkpoint["step"]
        torch.set_rng_state(checkpoint["rng"])
        if device == "cuda" and checkpoint.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    if steps <= start:
        raise ValueError(f"--steps is the total target and must exceed completed step {start}")
    dataset = [example(seed+i, size) for i in range(train_examples)]
    signatures = sorted({"\n".join(generate(seed+i, size).rows) for i in range(train_examples)})
    # Validate every example before any optimizer step.
    batch(dataset, "cpu", config.context)
    precision = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if device == "cuda" else "fp32"
    manifest = {"model": model.info(), "training": settings, "train_mazes": signatures,
                "torch": str(torch.__version__), "precision": precision}
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (run / "config.json").write_text(json.dumps(asdict(config), indent=2))
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    if checkpoint and checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    begin = time.perf_counter()
    print(json.dumps({"device": device, **model.info(), "precision": manifest["precision"]}), flush=True)
    if device == "cpu" and model.info()["parameters"] > 1_000_000:
        print("CPU training of the default model is slow. Use Colab GPU; --tiny is for smoke tests.", flush=True)

    def save(step):
        atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "step": step, "config": asdict(config), "settings": settings,
                     "rng": torch.get_rng_state(), "scaler": scaler.state_dict(),
                     "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else None}, run / "last.pt")
        # An independent FP32 inference artifact, without optimizer state.
        atomic_save({k: v.detach().float().cpu() for k, v in model.state_dict().items()}, run / "weights.pt")

    completed = start
    with (run / "metrics.jsonl").open("a") as log:
        try:
            for step in range(start, steps):
                rng = random.Random(seed + step * 1_000_003)
                chosen = [dataset[rng.randrange(len(dataset))] for _ in range(batch_size)]
                x, y = batch(chosen, device, config.context)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16) if device == "cuda" else nullcontext():
                    logits = model(x)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite loss; last good checkpoint retained")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(optimizer)
                scaler.update()
                completed = step + 1
                record = {"step": completed, "loss": float(loss.detach()), "elapsed_seconds": time.perf_counter()-begin,
                          "supervised_tokens": int((y != -100).sum()),
                          "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0}
                log.write(json.dumps(record) + "\n")
                log.flush()
                if completed == start+1 or completed % min(save_every, 25) == 0:
                    print(json.dumps(record), flush=True)
                if completed % save_every == 0:
                    save(completed)
        except KeyboardInterrupt:
            if completed > start:
                save(completed)
            raise
    save(completed)
    return {"step": completed, "weights_file_bytes": (run/"weights.pt").stat().st_size, **model.info()}

def load(run, device="auto"):
    run = Path(run)
    config_data = json.loads((run / "config.json").read_text())
    if config_data.get("kind") == "move_policy_v1":
        from .policy import MovePolicy, PolicyConfig
        model = MovePolicy(PolicyConfig(**config_data))
    else:
        model = Planner(ModelConfig(**config_data))
    model.load_state_dict(torch.load(run / "weights.pt", map_location="cpu", weights_only=True))
    model.to(device_for(device)).eval()
    metrics = run / "metrics.jsonl"
    if metrics.exists():
        lines = metrics.read_text().splitlines()
        model.training_steps = json.loads(lines[-1])["step"] if lines else 0
    return model

def solve(model, maze, attempts=4, temperature=.8, use_feedback=True):
    if attempts < 1 or attempts > 32:
        raise ValueError("attempts must be between 1 and 32")
    history, previous, feedback = [], None, None
    for _ in range(attempts):
        prompt = maze.prompt(previous, feedback) if use_feedback else maze.prompt()
        path = model.generate(prompt, max_moves=len(maze.rows)**2, temperature=temperature)
        result = verify(maze, path)
        history.append({"path": path, **result})
        if result["valid"]:
            break
        previous, feedback = path, result["reason"]
    return {"solved": history[-1]["valid"], "attempts_used": len(history), "history": history}

def evaluate(model, run, count=100, size=4, seed=1_000_000, attempts=4, temperature=.8):
    if count < 1:
        raise ValueError("count must be positive")
    torch.manual_seed(seed)
    manifest = json.loads((Path(run)/"manifest.json").read_text())
    identity = (lambda rows:"\n".join(rows).replace("S",".")) if hasattr(model,"search") else (lambda rows:"\n".join(rows))
    seen = {s.replace("S",".") if hasattr(model,"search") else s for s in manifest["train_mazes"]}
    records, candidate_seed = [], seed
    # Bound sampling: very small maze spaces can be exhausted by training.
    while len(records) < count and candidate_seed < seed + max(10000, count*100):
        maze = generate(candidate_seed, size)
        candidate_seed += 1
        signature = identity(maze.rows)
        if signature in seen:
            continue
        seen.add(signature)
        # Match initial RNG state for feedback and no-feedback resampling.
        gpu = next(model.parameters()).device
        devices = [gpu.index if gpu.index is not None else torch.cuda.current_device()] if gpu.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            control = solve(model, maze, attempts, temperature, use_feedback=False)
        result = solve(model, maze, attempts, temperature)
        # Random-walk baseline has the same per-attempt move budget, no oracle.
        rng = random.Random(candidate_seed)
        random_solved = any(verify(maze, "".join(rng.choices(list(MOVES), k=rng.randint(1, size*size))))["valid"] for _ in range(attempts))
        records.append({"maze": maze.rows, "seed": candidate_seed-1, "random_solved": random_solved,
                        "no_feedback": control, **result})
    if len(records) != count:
        raise ValueError("Insufficient unique held-out mazes; reduce count or increase maze size")
    first = sum(r["history"][0]["valid"] for r in records)
    solved = sum(r["solved"] for r in records)
    return {"count": count, "size": size, "seed": seed, "attempt_budget": attempts, "temperature": temperature,
            "single_attempt_success": first/count, "verified_retry_success": solved/count,
            "no_feedback_retry_success": sum(r["no_feedback"]["solved"] for r in records)/count,
            "optimal_solution_rate": sum(r["history"][-1]["optimal"] for r in records)/count,
            "random_retry_success": sum(r["random_solved"] for r in records)/count,
            "mean_attempts": sum(r["attempts_used"] for r in records)/count,
            "training_maze_overlap": 0, "records": records}
