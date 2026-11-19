"""One KU's own network: [query vector ; relation code] → scores over its slots + none.

The parameters are KU state (saved with the KU, never shared). Answer = arg-max slot; fire strength f = 1 − p(none).
Training reads only the KU's own examples (mxn/neuron/cloze.py), encoded by the shared frozen encoder.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from mxn.neuron.signal import RELATIONS


class KUNet(nn.Module):
    def __init__(self, d_query: int, n_slots: int, hidden: int = 32):
        super().__init__()
        self.n_slots = n_slots
        self.net = nn.Sequential(nn.Linear(d_query + len(RELATIONS), hidden), nn.ReLU(), nn.Linear(hidden, n_slots + 1))

    def forward(self, q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([q, r], 1))          # last column = none

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def train(X: np.ndarray, R: np.ndarray, y: np.ndarray, n_slots: int, hidden: int = 32, epochs: int = 60,
          lr: float = 1e-2, wd: float = 1e-4, seed: int = 0) -> tuple[KUNet, list[float]]:
    """Full-batch AdamW on the KU's own examples; y = slot index or -1 (none). Returns the net and the loss curve."""
    torch.manual_seed(seed)
    net = KUNet(X.shape[1], n_slots, hidden)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    Xt, Rt = torch.from_numpy(X), torch.from_numpy(R)
    yt = torch.from_numpy(np.where(y < 0, n_slots, y).astype(np.int64))
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(net(Xt, Rt), yt)
        loss.backward(); opt.step(); losses.append(loss.item())
    net.eval()
    return net, losses


@torch.inference_mode()
def answer(net: KUNet, q: np.ndarray, r: np.ndarray) -> tuple[float, int | None]:
    """(fire strength, slot index or None) for one query signal."""
    p = torch.softmax(net(torch.from_numpy(q[None]), torch.from_numpy(r[None])), 1)[0]
    k = int(p[:-1].argmax()) if net.n_slots else None
    return float(1 - p[-1]), k
