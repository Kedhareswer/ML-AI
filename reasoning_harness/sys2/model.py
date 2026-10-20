from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F

BOS, EOS, PAD, VOCAB = 128, 129, 130, 131

def encode(text):
    return list(text.encode("ascii"))

@dataclass
class ModelConfig:
    width: int = 256
    layers: int = 12
    heads: int = 8
    context: int = 256

    def __post_init__(self):
        if min(self.width, self.layers, self.heads, self.context) < 1 or self.width % self.heads:
            raise ValueError("Positive dimensions required; width must be divisible by heads")

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.heads
        self.norm1, self.norm2 = nn.LayerNorm(cfg.width), nn.LayerNorm(cfg.width)
        self.qkv = nn.Linear(cfg.width, 3 * cfg.width)
        self.proj = nn.Linear(cfg.width, cfg.width)
        self.ff = nn.Sequential(nn.Linear(cfg.width, 4 * cfg.width), nn.GELU(), nn.Linear(4 * cfg.width, cfg.width))

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        def heads(y):
            return y.view(b, t, self.heads, d // self.heads).transpose(1, 2)
        a = F.scaled_dot_product_attention(heads(q), heads(k), heads(v), is_causal=True)
        x = x + self.proj(a.transpose(1, 2).contiguous().view(b, t, d))
        return x + self.ff(self.norm2(x))

class Planner(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.tokens = nn.Embedding(VOCAB, c.width)
        self.positions = nn.Embedding(c.context, c.width)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.layers)])
        self.norm = nn.LayerNorm(c.width)
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)

    def forward(self, ids):
        if ids.shape[1] > self.config.context:
            raise ValueError("Input exceeds configured context length")
        x = self.tokens(ids) + self.positions(torch.arange(ids.shape[1], device=ids.device))
        for block in self.blocks:
            x = block(x)
        return F.linear(self.norm(x), self.tokens.weight)

    @torch.inference_mode()
    def generate(self, prompt, max_moves=64, temperature=.8):
        if not math.isfinite(temperature) or temperature < 0 or max_moves < 1:
            raise ValueError("temperature must be finite and nonnegative; max_moves must be positive")
        ids = [BOS] + encode(prompt)
        if len(ids) + max_moves > self.config.context:
            raise ValueError("Prompt plus move budget exceeds model context")
        self.eval()
        device = next(self.parameters()).device
        answer = []
        # Grammar constraints prevent invalid characters, not invalid decisions.
        allowed = [ord(x) for x in "UDLR"] + [EOS]
        for _ in range(max_moves):
            scores = self(torch.tensor([ids], device=device))[0, -1, allowed]
            index = scores.argmax() if temperature == 0 else torch.multinomial(F.softmax(scores / temperature, dim=-1), 1)[0]
            token = allowed[int(index)]
            if token == EOS:
                break
            ids.append(token)
            answer.append(chr(token))
        return "".join(answer)

    def info(self):
        parameters = sum(p.numel() for p in self.parameters())
        return {"config": asdict(self.config), "parameters": parameters, "fp32_weight_MB": parameters * 4 / 1e6}
