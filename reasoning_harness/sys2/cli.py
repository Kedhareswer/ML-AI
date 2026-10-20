import argparse
import json
from pathlib import Path
import torch
from .engine import device_for, evaluate, load, solve, train
from .model import ModelConfig, Planner
from .tasks import Maze

def main():
    parser = argparse.ArgumentParser(description="Scratch-trained tiny maze planner")
    parser.add_argument("--threads", type=int, default=2, help="CPU threads; also limits oversubscription in notebooks")
    subs = parser.add_subparsers(dest="command", required=True)
    info = subs.add_parser("info", help="Inspect model size without training")
    info.add_argument("--tiny", action="store_true")
    policy = subs.add_parser("train-policy",help="Train the small move policy from scratch")
    policy.add_argument("--run",required=True)
    policy.add_argument("--steps",type=int,default=1500)
    policy.add_argument("--mazes",type=int,default=3000)
    policy.add_argument("--batch-size",type=int,default=64)
    policy.add_argument("--seed",type=int,default=314)
    policy.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    policy.add_argument("--resume",action="store_true")
    tr = subs.add_parser("train")
    tr.add_argument("--run", required=True)
    tr.add_argument("--steps", type=int, default=5000, help="Total optimizer steps, including resumed steps")
    tr.add_argument("--batch-size", type=int, default=32)
    tr.add_argument("--lr", type=float, default=3e-4)
    tr.add_argument("--size", type=int, default=4)
    tr.add_argument("--train-examples", type=int, default=2048)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--save-every", type=int, default=200)
    tr.add_argument("--tiny", action="store_true", help="CPU test model; deliberately below requested production size")
    tr.add_argument("--config", help="ModelConfig JSON; overrides --tiny, ignored when resuming")
    tr.add_argument("--resume", action="store_true")
    bench = subs.add_parser("benchmark", help="Run quality, control, robustness and resource benchmarks")
    bench.add_argument("--run", required=True)
    bench.add_argument("--output", required=True, help="Fresh output directory")
    bench.add_argument("--count", type=int, default=100, help="Cases per size and evaluation seed")
    bench.add_argument("--sizes", type=int, nargs="+", default=[3,4,5])
    bench.add_argument("--seeds", type=int, nargs="+", default=[1000000,2000000,3000000])
    bench.add_argument("--attempts", type=int, default=4)
    bench.add_argument("--temperature", type=float, default=.8)
    bench.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    bench.add_argument("--id-target", type=float, default=.8)
    bench.add_argument("--ood-target", type=float, default=.6)
    bench.add_argument("--require-ready", action="store_true", help="Exit 2 if acceptance gates fail, after saving evidence")
    bench.add_argument("--require-local-solver", action="store_true", help="Require the separately scoped local hybrid solver gates")
    for name in ["eval", "solve", "ui"]:
        p = subs.add_parser(name)
        p.add_argument("--run", required=True)
        p.add_argument("--attempts", type=int, default=4)
        p.add_argument("--temperature", type=float, default=.8)
        if name == "eval":
            p.add_argument("--count", type=int, default=100)
            p.add_argument("--size", type=int, default=4)
            p.add_argument("--seed", type=int, default=1_000_000)
            p.add_argument("--output", required=True)
        if name == "solve":
            p.add_argument("--maze", required=True, help="Rows separated by /, e.g. S../.X./..G")
            p.add_argument("--mode",choices=["model","guided"],default="model")
        if name == "ui":
            p.add_argument("--port", type=int, default=7860)
        p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    tr.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    cfg = ModelConfig(width=32, layers=2, heads=4) if getattr(args, "tiny", False) else ModelConfig()
    if args.command == "info":
        print(json.dumps(Planner(cfg).info(), indent=2))
    elif args.command == "train-policy":
        from .policy import train_policy
        options=vars(args).copy()
        for key in ["command","threads"]:
            options.pop(key)
        print(json.dumps(train_policy(**options),indent=2))
    elif args.command == "train":
        if args.config and not args.resume:
            cfg = ModelConfig(**json.loads(Path(args.config).read_text()))
        options = vars(args).copy()
        for key in ["command", "threads", "tiny", "config"]:
            options.pop(key)
        print(json.dumps(train(config=cfg, **options), indent=2))
    elif args.command == "benchmark":
        from .benchmarks import benchmark
        options = vars(args).copy()
        for key in ["command", "threads", "require_ready", "require_local_solver"]:
            options.pop(key)
        report = benchmark(**options)
        print(json.dumps({"count": report["count"], "gates": report["gates"], "ready": report["ready"],
                          "local_hybrid_solver_ready":report["local_hybrid_solver_ready"]}, indent=2))
        if args.require_ready and not report["ready"]:
            raise SystemExit(2)
        if args.require_local_solver and not report["local_hybrid_solver_ready"]:
            raise SystemExit(2)
    elif args.command == "ui":
        from .ui import serve
        serve(load(args.run, args.device), args.port, args.attempts, args.temperature)
    else:
        model = load(args.run, args.device)
        if args.command == "solve":
            maze=Maze(tuple(args.maze.split("/")))
            if args.mode=="guided":
                if not hasattr(model,"search"):
                    parser.error("Guided mode requires a move-policy checkpoint")
                result=model.search(maze,64)
            else:
                result=solve(model,maze,args.attempts,args.temperature)
            print(json.dumps(result,indent=2))
        else:
            output = Path(args.output)
            if output.exists():
                parser.error("Evaluation output already exists; choose a new path")
            result = evaluate(model, args.run, args.count, args.size, args.seed, args.attempts, args.temperature)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2))
            print(json.dumps({k:v for k,v in result.items() if k != "records"}, indent=2))

if __name__ == "__main__":
    main()
