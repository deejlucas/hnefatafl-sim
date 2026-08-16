"""Policy/value network for Fetlar Hnefatafl (AlphaZero-style).

A small ResNet sized for laptop training: conv stem, `blocks` residual
blocks of `channels` filters, then

 - policy head: 1x1 conv down to 40 channels whose layout matches the
   engine's action encoding exactly -- channel m at square sq is the logit
   for action ``sq * 40 + m`` -- so the head needs no giant FC layer;
 - value head: 1x1 conv + 2-layer FC to a tanh scalar, the expected outcome
   from the *side-to-move* perspective (+1 = side to move wins).

One shared network plays both sides; plane 3 of the input tells it whose
turn it is.  `NetEvaluator` is the bridge MCTS uses: a batch of States in,
sparse (legal_actions, priors, value) out.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .engine import MOVES_PER_SQ, N, N_SQ, State, encode, legal_actions


def best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class PolicyValueNet(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 6):
        super().__init__()
        self.channels = channels
        self.blocks = blocks
        self.stem = nn.Sequential(
            nn.Conv2d(5, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        self.tower = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.p_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        self.p_out = nn.Conv2d(channels, MOVES_PER_SQ, 1)
        self.v_conv = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False),
            nn.BatchNorm2d(2), nn.ReLU(inplace=True))
        self.v_fc = nn.Sequential(
            nn.Linear(2 * N_SQ, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))

    def forward(self, x):
        """x: (B, 5, 11, 11) -> policy logits (B, 4840), value (B,)."""
        h = self.tower(self.stem(x))
        p = self.p_out(self.p_conv(h))               # (B, 40, 11, 11)
        p = p.flatten(2).transpose(1, 2).reshape(x.shape[0], -1)
        v = torch.tanh(self.v_fc(self.v_conv(h).flatten(1))).squeeze(-1)
        return p, v

    def net_config(self) -> dict:
        return {"channels": self.channels, "blocks": self.blocks}


class NetEvaluator:
    """Batched evaluator: list of States -> [(acts, priors, value), ...].

    Priors are a softmax over the *legal* actions only (sparse: `acts` are
    the legal action ids, `priors` the matching probabilities).
    """

    def __init__(self, net: PolicyValueNet, device: torch.device | None = None):
        self.device = device or best_device()
        self.net = net.to(self.device)
        self.net.eval()

    @torch.no_grad()
    def evaluate(self, states: list[State]):
        obs = np.stack([encode(s) for s in states])
        x = torch.from_numpy(obs).to(self.device)
        logits, v = self.net(x)
        logits = logits.float().cpu().numpy()
        v = v.float().cpu().numpy()
        out = []
        for i, s in enumerate(states):
            acts = legal_actions(s)
            la = logits[i, acts]
            la -= la.max()
            p = np.exp(la)
            p /= p.sum()
            out.append((acts, p.astype(np.float32), float(v[i])))
        return out


def save_checkpoint(net: PolicyValueNet, path, **meta):
    torch.save({"model": net.state_dict(), "net_config": net.net_config(),
                "saved_at": time.time(), **meta}, path)


def load_checkpoint(path, device: torch.device | None = None):
    """Returns (net, checkpoint_dict); the net is on `device` in eval mode."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    net = PolicyValueNet(**ckpt["net_config"])
    net.load_state_dict(ckpt["model"])
    if device is not None:
        net = net.to(device)
    net.eval()
    return net, ckpt
