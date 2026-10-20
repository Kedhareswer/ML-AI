"""Learned move decisions plus explicit, separately measured backtracking."""
from collections import deque
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.nn import functional as F

from .tasks import Maze, MOVES, generate, verify

ACTIONS = tuple(MOVES)
INPUTS = 207

def task_key(rows):
    # Every reachable current position is supervised, so S is not task identity.
    return "\n".join(rows).replace("S", ".")

@dataclass
class PolicyConfig:
    width: int = 1600
    layers: int = 4
    kind: str = "move_policy_v1"

    def __post_init__(self):
        if self.width < 4 or self.layers < 1 or self.kind != "move_policy_v1":
            raise ValueError("Invalid policy configuration")

def features(maze, position):
    planes = torch.zeros(3,8,8)
    planes[0] = 1
    for r,row in enumerate(maze.rows):
        for c,char in enumerate(row):
            planes[0,r,c] = float(char == "X")
    r,c = position
    gr,gc = maze.locate("G")
    planes[1,r,c], planes[2,gr,gc] = 1,1
    legal = [maze.open(r+dr,c+dc) for dr,dc in MOVES.values()]
    extra = [len(maze.rows)/8,r/7,c/7,gr/7,gc/7,(gr-r)/7,(gc-c)/7]
    extra += [float(x) for x in legal]
    extra += [(abs(gr-r-dr)+abs(gc-c-dc))/14 for dr,dc in MOVES.values()]
    return torch.cat([planes.flatten(),torch.tensor(extra)]), torch.tensor(legal)

def distances(maze):
    goal = maze.locate("G")
    result, queue = {goal:0}, deque([goal])
    while queue:
        r,c=queue.popleft()
        for dr,dc in MOVES.values():
            nxt=r+dr,c+dc
            if maze.open(*nxt) and nxt not in result:
                result[nxt]=result[r,c]+1
                queue.append(nxt)
    return result

class MovePolicy(nn.Module):
    def __init__(self,config=None):
        super().__init__()
        self.config=config or PolicyConfig()
        layers=[nn.Linear(INPUTS,self.config.width),nn.ReLU()]
        for _ in range(self.config.layers-1):
            layers += [nn.Linear(self.config.width,self.config.width),nn.ReLU()]
        self.network=nn.Sequential(*layers,nn.Linear(self.config.width,4))

    def forward(self,x):
        return self.network(x)

    def info(self):
        n=sum(p.numel() for p in self.parameters())
        return {"config":asdict(self.config),"parameters":n,"fp32_weight_MB":n*4/1e6}

    @torch.inference_mode()
    def scores(self,maze,position):
        x,legal=features(maze,position)
        device=next(self.parameters()).device
        return self(x.to(device)).cpu().masked_fill(~legal,-float("inf")),legal

    @torch.inference_mode()
    def rollout(self,maze,temperature=0):
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("Temperature must be finite and nonnegative")
        position=maze.locate("S")
        visited={position}
        path=""
        for _ in range(len(maze.rows)**2):
            if position==maze.locate("G"):
                break
            scores,legal=self.scores(maze,position)
            for i,(dr,dc) in enumerate(MOVES.values()):
                if (position[0]+dr,position[1]+dc) in visited:
                    scores[i]=-float("inf")
            if not torch.isfinite(scores).any():
                break
            action=int(scores.argmax()) if temperature==0 else int(torch.multinomial(F.softmax(scores/temperature,-1),1))
            dr,dc=MOVES[ACTIONS[action]]
            position=position[0]+dr,position[1]+dc
            path += ACTIONS[action]
            visited.add(position)
        return path

    def generate(self,prompt,max_moves=64,temperature=.8):
        # Adapter for the existing model-only evaluator. Feedback is not learned
        # by this policy; resampling is reported separately from search.
        if max_moves < 1:
            raise ValueError("max_moves must be positive")
        rows=[]
        for line in prompt.splitlines()[1:]:
            if set(line) <= set(".XSG") and line:
                rows.append(line)
            else:
                break
        return self.rollout(Maze(tuple(rows)),temperature)[:max_moves]

    @torch.inference_mode()
    def search(self,maze,budget=64):
        if not 1 <= budget <= 64:
            raise ValueError("Search budget must be 1–64 expanded states")
        frontier=[(maze.locate("S"),"")]
        visited=set()
        trace=[]
        while frontier and len(visited)<budget:
            position,path=frontier.pop()
            if position in visited:
                continue
            visited.add(position)
            trace.append({"position":list(position),"path":path})
            if position==maze.locate("G"):
                result=verify(maze,path)
                return {"solved":result["valid"],"mode":"policy_guided_search","expanded_states":len(visited),
                        "budget":budget,"attempts_used":1,"history":[{"path":path,**result}],"trace":trace}
            scores,legal=self.scores(maze,position)
            for i in torch.argsort(scores).tolist():
                if legal[i]:
                    dr,dc=MOVES[ACTIONS[i]]
                    nxt=position[0]+dr,position[1]+dc
                    if nxt not in visited:
                        frontier.append((nxt,path+ACTIONS[i]))
        return {"solved":False,"mode":"policy_guided_search","expanded_states":len(visited),"budget":budget,
                "attempts_used":1,"history":[{"path":"","valid":False,"optimal":False,"reason":"search_exhausted","at":0}],"trace":trace}

def train_policy(run,steps=1500,mazes=3000,batch_size=128,seed=314,device="auto",width=1600,layers=4,resume=False):
    from .engine import atomic_save,device_for
    run=Path(run)
    checkpoint = None
    if resume:
        checkpoint = torch.load(run/"last.pt",map_location="cpu",weights_only=True)
        settings = checkpoint["settings"]
        mazes,batch_size,seed = (settings[k] for k in ["mazes","batch_size","seed"])
        width,layers = checkpoint["config"]["width"],checkpoint["config"]["layers"]
    elif run.exists() and any(run.iterdir()):
        raise ValueError("Use a fresh run directory")
    if min(steps,mazes,batch_size)<1:
        raise ValueError("Positive training settings required")
    run.mkdir(parents=True,exist_ok=True)
    device=device_for(device)
    torch.manual_seed(seed)
    model=MovePolicy(PolicyConfig(width,layers)).to(device)
    xs,ys,ls=[],[],[]
    signatures=set()
    for i in range(mazes):
        maze=generate(10_000_000+seed*10000+i,size=3+i%2)
        signatures.add("\n".join(maze.rows))
        d=distances(maze)
        for position,distance in d.items():
            if distance==0:
                continue
            x,legal=features(maze,position)
            target=torch.tensor([float(d.get((position[0]+dr,position[1]+dc))==distance-1) for dr,dc in MOVES.values()])
            xs.append(x);ys.append(target/target.sum());ls.append(legal)
    x,y,legal=torch.stack(xs).to(device),torch.stack(ys).to(device),torch.stack(ls).to(device)
    settings={"seed":seed,"size":4,"train_sizes":[3,4],"mazes":mazes,"states":len(xs),"batch_size":batch_size,"steps":steps,
              "objective":"uniform distribution over all shortest-path next actions","optimizer":"AdamW","lr":.0005}
    manifest={"model":model.info(),"training":settings,"train_mazes":sorted(signatures),
              "train_task_keys":sorted({s.replace("S",".") for s in signatures}),
              "torch":str(torch.__version__),"precision":"fp32"}
    (run/"config.json").write_text(json.dumps(asdict(model.config),indent=2))
    (run/"manifest.json").write_text(json.dumps(manifest,indent=2))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.001)
    first_step = 0
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["rng"])
        if device == "cuda" and checkpoint.get("cuda_rng"):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        first_step = checkpoint["step"]
    if steps <= first_step:
        raise ValueError("Total target steps must exceed completed steps")
    start=time.perf_counter()
    print(json.dumps({"device":device,**model.info(),"training_states":len(xs)}),flush=True)
    def save(step):
        atomic_save({k:v.detach().float().cpu() for k,v in model.state_dict().items()},run/"weights.pt")
        atomic_save({"kind":"move_policy_v1","model":model.state_dict(),"optimizer":optimizer.state_dict(),
                     "step":step,"config":asdict(model.config),"settings":settings,"rng":torch.get_rng_state(),
                     "cuda_rng":torch.cuda.get_rng_state_all() if device=="cuda" else None},run/"last.pt")
    completed=first_step
    try:
        with (run/"metrics.jsonl").open("a" if resume else "w") as log:
            for step in range(first_step,steps):
                ids=torch.randint(len(x),(batch_size,),device=device)
                logits=model(x[ids]).masked_fill(~legal[ids],-1e4)
                loss=-(y[ids]*F.log_softmax(logits,-1)).sum(-1).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
                completed=step+1
                record={"step":completed,"loss":float(loss.detach()),"elapsed_seconds":time.perf_counter()-start}
                log.write(json.dumps(record)+"\n");log.flush()
                if completed==1 or completed%100==0:
                    print(json.dumps(record),flush=True)
                if completed%250==0:
                    save(completed)
    finally:
        if completed:
            save(completed)
    return {**model.info(),"steps":completed,"weights_file_bytes":(run/"weights.pt").stat().st_size}
