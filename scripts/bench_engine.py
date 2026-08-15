#!/usr/bin/env python
"""Engine throughput benchmark: random self-play games.

Usage: .venv/bin/python scripts/bench_engine.py [--seconds 3] [--profile]

Reports plies/second (move generation + apply, including capture, terminal
and repetition handling), which is the number that bounds self-play speed.
"""

import argparse
import cProfile
import pstats
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tafl.engine import apply, initial_state, legal_actions  # noqa: E402


def run(seconds: float, seed: int = 0):
    rng = random.Random(seed)
    plies = games = 0
    lengths = []
    reasons = Counter()
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        st = initial_state()
        n = 0
        while st.result is None:
            acts = legal_actions(st)
            st = apply(st, int(acts[rng.randrange(len(acts))]))
            n += 1
        plies += n
        games += 1
        lengths.append(n)
        reasons[st.result.reason] += 1
    dt = time.perf_counter() - t0
    print(f"{plies} plies in {dt:.2f}s over {games} games "
          f"-> {plies / dt:,.0f} plies/sec")
    if games:
        print(f"avg game length: {sum(lengths) / games:.1f} plies")
        print("outcomes:", dict(reasons))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()
    if args.profile:
        pr = cProfile.Profile()
        pr.enable()
        run(args.seconds, args.seed)
        pr.disable()
        pstats.Stats(pr).sort_stats("cumulative").print_stats(20)
    else:
        run(args.seconds, args.seed)


if __name__ == "__main__":
    main()
