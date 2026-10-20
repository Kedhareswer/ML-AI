"""Grid networks trained from random initialisation.

Every block mixes local 3x3 context with masked means over the cell's row,
column and region, so a cell can "see" the constraints of a whole line or box.
"""
import torch
from torch import nn
import torch.nn.functional as F


class MixBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.local = nn.Conv2d(width, width, 3, padding=1)
        self.mix = nn.Conv2d(width * 4, width, 1)
        self.norm1 = nn.GroupNorm(8, width)
        self.ff = nn.Sequential(nn.Conv2d(width, width * 2, 1), nn.GELU(), nn.Conv2d(width * 2, width, 1))
        self.norm2 = nn.GroupNorm(8, width)

    def forward(self, x, mask, regions):
        # mask: (B,1,H,W) board cells; regions: (B,R,H*W) one-hot region membership
        b, c, h, w = x.shape
        xm = x * mask
        row = xm.sum(3, keepdim=True) / mask.sum(3, keepdim=True).clamp(min=1)
        col = xm.sum(2, keepdim=True) / mask.sum(2, keepdim=True).clamp(min=1)
        flat = xm.flatten(2)
        counts = regions.sum(2).clamp(min=1)
        pooled = torch.einsum("bcn,brn->bcr", flat, regions) / counts[:, None, :]
        reg = torch.einsum("bcr,brn->bcn", pooled, regions).view(b, c, h, w)
        h1 = self.local(x) + self.mix(torch.cat([x, row.expand_as(x), col.expand_as(x), reg], 1))
        x = x + F.gelu(self.norm1(h1))
        x = x + self.ff(self.norm2(x))
        return x * mask


class GridNet(nn.Module):
    def __init__(self, in_ch, out_ch, width=96, blocks=8):
        super().__init__()
        self.config = {"in_ch": in_ch, "out_ch": out_ch, "width": width, "blocks": blocks}
        self.stem = nn.Conv2d(in_ch + 1, width, 3, padding=1)
        self.blocks = nn.ModuleList(MixBlock(width) for _ in range(blocks))
        self.head = nn.Conv2d(width, out_ch, 1)

    def forward(self, x, mask, regions):
        h = self.stem(torch.cat([x, mask], 1)) * mask
        for block in self.blocks:
            h = block(h, mask, regions)
        return self.head(h)

    def parameters_count(self):
        return sum(p.numel() for p in self.parameters())


def save_net(net, path, extra=None):
    torch.save({"config": net.config, "state": net.state_dict(), "extra": extra or {}}, path)


def load_net(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    net = GridNet(**data["config"])
    net.load_state_dict(data["state"])
    net.eval()
    return net, data.get("extra", {})
