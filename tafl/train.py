"""Replay buffer, optimizer step, and the iterate/gate training loop.

One training iteration = generate `games_per_iter` self-play games with the
current net -> push samples into the replay buffer -> take optimizer steps
sized so each fresh sample is seen ~`sample_ratio` times -> every
`gate_every` iterations, pit the current net against the reigning best in a
paired color-balanced match and promote it only if it scores >= `gate_win`.

Loss = policy cross-entropy (MCTS visit distribution as target, over the
full 4840 logits) + value MSE + L2 via optimizer weight_decay.  Training
runs on MPS when available; self-play workers stay on CPU (see selfplay.py).
"""

from __future__ import annotations

import json
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .engine import ATT, N_ACTIONS
from .eval import ArenaConfig, elo_diff, play_match_mp
from .net import PolicyValueNet, best_device, save_checkpoint
from .selfplay import (GameRecord, Sample, SelfPlayConfig, generate_games,
                       sample_planes)


@dataclass
class TrainConfig:
    channels: int = 64
    blocks: int = 6
    iters: int = 200
    games_per_iter: int = 24
    workers: int = 8               # self-play + gating processes (perf cores)
    buffer_capacity: int = 250_000       # positions (~1000 games)
    min_buffer: int = 2_000              # start training once this full
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    sample_ratio: float = 1.0            # avg times each new sample is trained on
    gate_every: int = 5                  # iterations between gating matches
    gate_pairs: int = 12                 # game pairs per gating match
    gate_win: float = 0.55               # promotion threshold (score rate)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    seed: int = 0

    def to_json(self) -> dict:
        return asdict(self)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.data: deque[Sample] = deque(maxlen=capacity)

    def __len__(self):
        return len(self.data)

    def add_game(self, rec: GameRecord):
        self.data.extend(rec.samples)

    def sample_batch(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(len(self.data), size=batch_size)
        obs = np.empty((batch_size, 5, 11, 11), dtype=np.float32)
        pol = np.zeros((batch_size, N_ACTIONS), dtype=np.float32)
        z = np.empty(batch_size, dtype=np.float32)
        for j, i in enumerate(idx):
            s = self.data[i]
            obs[j] = sample_planes(s)
            pol[j, s.acts] = s.probs
            z[j] = s.z
        return obs, pol, z


class Trainer:
    def __init__(self, cfg: TrainConfig, device: torch.device | None = None):
        self.cfg = cfg
        self.device = device or best_device()
        self.net = PolicyValueNet(cfg.channels, cfg.blocks).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr,
                                    weight_decay=cfg.weight_decay)
        self.rng = np.random.default_rng(cfg.seed)

    def train_steps(self, buffer: ReplayBuffer, n_steps: int) -> dict:
        self.net.train()
        p_losses, v_losses = [], []
        for _ in range(n_steps):
            obs, pol, z = buffer.sample_batch(self.cfg.batch_size, self.rng)
            obs_t = torch.from_numpy(obs).to(self.device)
            pol_t = torch.from_numpy(pol).to(self.device)
            z_t = torch.from_numpy(z).to(self.device)
            logits, v = self.net(obs_t)
            p_loss = -(pol_t * F.log_softmax(logits, dim=1)).sum(1).mean()
            v_loss = F.mse_loss(v, z_t)
            loss = p_loss + v_loss
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            p_losses.append(p_loss.item())
            v_losses.append(v_loss.item())
        self.net.eval()
        return {"policy_loss": float(np.mean(p_losses)) if p_losses else None,
                "value_loss": float(np.mean(v_losses)) if v_losses else None,
                "steps": n_steps}

    def cpu_state_dict(self):
        """Snapshot of the weights on CPU.  The .clone() matters: .cpu() is a
        no-op alias when training runs on CPU, and gating/promotion must hold
        a frozen copy, not references to the live parameters."""
        return {k: v.detach().cpu().clone()
                for k, v in self.net.state_dict().items()}


def write_meta(ckpt_path: Path, entry: dict):
    """Sidecar `<ckpt>.json` (games seen, eval Elo, timestamp) so tools can
    read a checkpoint's provenance without loading its weights."""
    meta = {k: v for k, v in entry.items()
            if isinstance(v, (int, float, str, bool, dict)) or v is None}
    meta["saved_at"] = time.time()
    ckpt_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def selfplay_stats(records: list[GameRecord]) -> dict:
    reasons = Counter(r.result.reason for r in records)
    winners = Counter("draw" if r.result.winner is None
                      else ("att" if r.result.winner == ATT else "def")
                      for r in records)
    lengths = [r.plies for r in records]
    return {"games": len(records),
            "positions": int(sum(lengths)),
            "avg_len": float(np.mean(lengths)),
            "winners": dict(winners),
            "reasons": dict(reasons)}


def run_training(cfg: TrainConfig, run_dir: str | Path,
                 models_dir: str | Path = "models") -> Path:
    """The full iterate/gate loop.  Returns the path of the best checkpoint."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl"
    (run_dir / "config.json").write_text(json.dumps(cfg.to_json(), indent=2))

    trainer = Trainer(cfg)
    buffer = ReplayBuffer(cfg.buffer_capacity)
    best_sd = trainer.cpu_state_dict()
    best_path = models_dir / "best.pt"
    save_checkpoint(trainer.net, best_path, iter=0, games=0)
    write_meta(best_path, {"iter": 0, "total_games": 0, "promoted": True})
    total_games = 0
    print(f"training on {trainer.device}, self-play on {cfg.workers} "
          f"CPU workers, run dir {run_dir}", flush=True)

    for it in range(1, cfg.iters + 1):
        t0 = time.perf_counter()
        records = generate_games(trainer.cpu_state_dict(),
                                 trainer.net.net_config(), cfg.selfplay,
                                 cfg.games_per_iter, cfg.workers,
                                 seed=cfg.seed * 7_919 + it)
        sp_time = time.perf_counter() - t0
        stats = selfplay_stats(records)
        for rec in records:
            buffer.add_game(rec)
        total_games += len(records)

        t1 = time.perf_counter()
        if len(buffer) >= cfg.min_buffer:
            n_steps = max(1, int(stats["positions"] * cfg.sample_ratio
                                 / cfg.batch_size))
            metrics = trainer.train_steps(buffer, n_steps)
        else:
            metrics = {"policy_loss": None, "value_loss": None, "steps": 0}
        tr_time = time.perf_counter() - t1

        entry = {"iter": it, "total_games": total_games,
                 "buffer": len(buffer),
                 "selfplay_sec": round(sp_time, 1),
                 "train_sec": round(tr_time, 1),
                 "positions_per_sec": round(stats["positions"] / sp_time, 1),
                 **stats, **metrics}

        if it % cfg.gate_every == 0 and metrics["steps"] > 0:
            t2 = time.perf_counter()
            match = play_match_mp(trainer.cpu_state_dict(), best_sd,
                                  trainer.net.net_config(), cfg.gate_pairs,
                                  cfg.arena, workers=cfg.workers,
                                  seed=cfg.seed * 104_729 + it)
            entry["gate_score"] = match.score_rate
            entry["gate_elo"] = round(elo_diff(match.score_rate), 1)
            entry["gate"] = match.summary()
            entry["gate_sec"] = round(time.perf_counter() - t2, 1)
            if match.score_rate >= cfg.gate_win:
                best_sd = trainer.cpu_state_dict()
                save_checkpoint(trainer.net, best_path, iter=it,
                                games=total_games,
                                gate_score=match.score_rate)
                entry["promoted"] = True
                write_meta(best_path, entry)
            ckpt = models_dir / f"ckpt_{it:04d}.pt"
            save_checkpoint(trainer.net, ckpt, iter=it, games=total_games)
            write_meta(ckpt, entry)

        with log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[iter {it}] games={total_games} buffer={len(buffer)} "
              f"p_loss={metrics['policy_loss']} v_loss={metrics['value_loss']} "
              f"sp={sp_time:.0f}s tr={tr_time:.0f}s "
              f"winners={stats['winners']}"
              + (f" gate={entry['gate_score']:.2f} {entry['gate']}"
                 if "gate_score" in entry else ""),
              flush=True)
    return best_path
