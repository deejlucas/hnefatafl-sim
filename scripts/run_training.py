#!/usr/bin/env python
"""Entry point for self-play training runs.

Smoke run (validate that learning happens, ~15 min):
    .venv/bin/python scripts/run_training.py --smoke

Full overnight run:
    .venv/bin/python scripts/run_training.py --iters 400 --run-dir runs/overnight

Progress is streamed to stdout and appended to <run-dir>/log.jsonl;
checkpoints land in models/.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tafl.train import TrainConfig, run_training  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny fast configuration to validate learning")
    ap.add_argument("--iters", type=int)
    ap.add_argument("--games-per-iter", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--sims", type=int)
    ap.add_argument("--channels", type=int)
    ap.add_argument("--blocks", type=int)
    ap.add_argument("--gate-every", type=int)
    ap.add_argument("--move-limit", type=int)
    ap.add_argument("--draw-penalty", type=float,
                    help="self-play value target for draws is -penalty for "
                         "both sides; keeps the value head off a constant 0 "
                         "when self-play is draw-heavy (smoke run was ~75%% "
                         "draws)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--models-dir", default="models",
                    help="where checkpoints land; a fresh run OVERWRITES "
                         "best.pt there, so point smoke/test runs elsewhere")
    args = ap.parse_args()

    cfg = TrainConfig(seed=args.seed)
    if args.smoke:
        cfg.channels, cfg.blocks = 32, 4
        cfg.iters = 6
        cfg.games_per_iter = 12
        cfg.gate_every = 3
        cfg.gate_pairs = 4
        cfg.min_buffer = 500
        cfg.selfplay.sims = 64
        cfg.selfplay.draw_penalty = 0.1   # weak play is draw-heavy; keep the
        cfg.arena.sims = 64               # value target off the constant 0
    for name, dst in (("iters", "iters"), ("games_per_iter", "games_per_iter"),
                      ("workers", "workers"), ("channels", "channels"),
                      ("blocks", "blocks"), ("gate_every", "gate_every")):
        v = getattr(args, name)
        if v is not None:
            setattr(cfg, dst, v)
    if args.sims is not None:
        cfg.selfplay.sims = args.sims
        cfg.arena.sims = args.sims
    if args.move_limit is not None:
        cfg.selfplay.move_limit = args.move_limit
        cfg.arena.move_limit = args.move_limit
    if args.draw_penalty is not None:
        cfg.selfplay.draw_penalty = args.draw_penalty

    run_dir = args.run_dir or time.strftime("runs/%Y%m%d-%H%M%S")
    best = run_training(cfg, run_dir, models_dir=args.models_dir)
    print(f"done; best checkpoint: {best}")


if __name__ == "__main__":
    main()
