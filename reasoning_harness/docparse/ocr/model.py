"""A small CRNN line recognizer: convolutions read glyph shapes, a BiLSTM reads context, CTC aligns."""
import numpy as np
import torch
from torch import nn

from .charset import CLASSES, decode_greedy
from .render import HEIGHT


def _block(cin, cout, stride=1):
    return [nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


class LineOCR(nn.Module):
    def __init__(self, width=32, hidden=128):
        super().__init__()
        self.config = {"width": width, "hidden": hidden}
        w = width
        self.cnn = nn.Sequential(
            *_block(1, w, 2),                                      # 16 x W/2 (strided stem keeps CPU cost low)
            *_block(w, 2 * w), nn.MaxPool2d(2),                    # 8 x W/4
            *_block(2 * w, 4 * w), nn.MaxPool2d((2, 1)),           # 4 x W/4
            *_block(4 * w, 4 * w), nn.MaxPool2d((2, 1)),           # 2 x W/4
            *_block(4 * w, 8 * w),
            nn.Conv2d(8 * w, 8 * w, (2, 1)), nn.BatchNorm2d(8 * w), nn.ReLU(inplace=True))  # 1 x W/4
        self.rnn = nn.LSTM(8 * w, hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=0.1)
        self.head = nn.Linear(2 * hidden, CLASSES)

    def forward(self, x):
        """x: (B,1,HEIGHT,W) -> log-probs (B, W/4, CLASSES)."""
        f = self.cnn(x).squeeze(2).transpose(1, 2)
        f, _ = self.rnn(f)
        return self.head(f).log_softmax(-1)


def batch_tensor(images, pad_value=None):
    """List of uint8 (HEIGHT, w) arrays -> float tensor (B,1,HEIGHT,maxW), dark text made positive."""
    wmax = max(a.shape[1] for a in images)
    wmax = (wmax + 3) // 4 * 4
    out = np.zeros((len(images), 1, HEIGHT, wmax), np.float32)
    for i, a in enumerate(images):
        x = 1.0 - a.astype(np.float32) / 255.0
        if x[:, :2].mean() + x[:, -2:].mean() > 1.0 and x.mean() > 0.5:  # light text on dark: flip to dark-on-light
            x = 1.0 - x
        bg = np.median(np.concatenate([x[0], x[-1], x[:, 0], x[:, -1]]))
        x = x - bg  # background to zero, so right padding matches it
        out[i, 0, :, :a.shape[1]] = x
    return torch.from_numpy(out), [a.shape[1] // 4 for a in images]


class Recognizer:
    """Load trained weights and read line images."""

    def __init__(self, path, device="cpu"):
        data = torch.load(path, map_location=device, weights_only=True)
        self.net = LineOCR(**data["config"]).to(device).eval()
        self.net.load_state_dict(data["state"])
        self.device = device
        self.info = data.get("info", {})

    @torch.inference_mode()
    def read(self, images, batch=32):
        """images: uint8 grayscale arrays of height HEIGHT -> [(text, confidence)]."""
        results = [None] * len(images)
        order = sorted(range(len(images)), key=lambda i: images[i].shape[1])
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            x, lens = batch_tensor([images[i] for i in idx])
            logp = self.net(x.to(self.device)).cpu()
            for k, i in enumerate(idx):
                lp = logp[k, :max(1, lens[k])]
                best = lp.argmax(-1)
                text = decode_greedy(best.tolist())
                emitted = best != 0
                conf = float(lp.max(-1).values[emitted].exp().mean()) if emitted.any() else 0.0
                results[i] = (text.strip(), round(conf, 4))
        return results
