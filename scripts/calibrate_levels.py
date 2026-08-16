#!/usr/bin/env python
"""Calibrate the five difficulty levels per side and write models/levels.json.

Each candidate configuration (checkpoint x sims x randomness) plays a fixed
panel of opponents on the *other* side from fixed seeds; the ladder is then
picked from the measured within-side ordering.  See tafl/calibrate.py.

Quick pipeline check on smoke checkpoints (~minutes):
    .venv/bin/python scripts/calibrate_levels.py --smoke

Real calibration after the overnight run:
    .venv/bin/python scripts/calibrate_levels.py --games 12 --workers 8

Costs 2 sides x candidates x panel x --games games (--dry-run prints the
schedule without playing).  A candidate's score SE is ~sqrt(0.25 / games
per side), so small --games values cannot certify min_margin-sized steps:
treat sub-margin warnings at low counts as "re-run with more games".
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tafl.agents import list_checkpoints  # noqa: E402
from tafl.calibrate import (CalibrationConfig, build_jobs,  # noqa: E402
                            calibrate, candidate_pool, format_table,
                            panel_specs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--out", default=None,
                    help="default: <models-dir>/levels.json")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny/fast settings to validate the pipeline")
    ap.add_argument("--games", type=int, help="games per candidate/opponent")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--move-limit", type=int)
    ap.add_argument("--min-margin", type=float)
    ap.add_argument("--panel-sims", type=int)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the schedule and exit")
    args = ap.parse_args()

    cfg = CalibrationConfig()
    if args.smoke:
        cfg.games = 1
        cfg.panel_sims = 16
        cfg.sims_ladder = (0, 8, 16, 32, 64)
        cfg.match.move_limit = 120
        cfg.match.opening_plies = 6
    for name, dst in (("games", "games"), ("workers", "workers"),
                      ("seed", "seed"), ("min_margin", "min_margin"),
                      ("panel_sims", "panel_sims")):
        v = getattr(args, name)
        if v is not None:
            setattr(cfg, dst, v)
    if args.move_limit is not None:
        cfg.match.move_limit = args.move_limit

    models_dir = Path(args.models_dir)
    out_path = Path(args.out) if args.out else models_dir / "levels.json"
    ckpts = list_checkpoints(models_dir)
    if not ckpts:
        sys.exit(f"no checkpoints in {models_dir}; run training first")
    cands = candidate_pool(ckpts, cfg)
    panel = panel_specs(ckpts, cfg)
    n_jobs = len(build_jobs(cands, panel, cfg))
    print(f"checkpoints: {', '.join(c.name for c in ckpts)}")
    print(f"{len(cands)} candidates x {len(panel)} panel opponents x "
          f"{cfg.games} games x 2 sides = {n_jobs} games "
          f"on {cfg.workers} workers")
    print("panel: " + ", ".join(p["label"] for p in panel))
    if args.dry_run:
        for c in cands:
            print(f"  candidate {c['label']}")
        return

    t0 = time.perf_counter()
    done = [0]

    def progress(_r):
        done[0] += 1
        if done[0] % 20 == 0 or done[0] == n_jobs:
            el = time.perf_counter() - t0
            print(f"  {done[0]}/{n_jobs} games  {el:.0f}s "
                  f"({el / done[0]:.1f}s/game)", flush=True)

    doc = calibrate(models_dir, cfg, on_result=progress)
    doc["method"]["wall_sec"] = round(time.perf_counter() - t0, 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2) + "\n")
    print(format_table(doc))
    print(f"\nwrote {out_path} ({doc['method']['wall_sec']}s)")


if __name__ == "__main__":
    main()
