# Hnefatafl RL

AlphaZero-style self-play reinforcement learning for **Fetlar Hnefatafl**
(11×11), with five calibrated difficulty levels per side and (eventually) a
browser UI to play against.  [plan.md](plan.md) has the full project plan.

**Status:** Phases 1–3 are done — engine, RL stack, and the agent/calibration
layer, all smoke-tested (92 tests passing).  **Phase 4 — the real overnight
training run — is next**, and this README walks through it.  Calibration of
the difficulty levels happens *after* Phase 4, using the checkpoints the
training run leaves behind; the commands for that are at the bottom.

## Layout

```
tafl/       engine.py  net.py  mcts.py  selfplay.py  train.py  agents.py  eval.py  calibrate.py
tests/      92 tests (engine rules, MCTS, training, agents, calibration)
scripts/    run_training.py  calibrate_levels.py  bench_engine.py
models/     checkpoints + levels.json (real run; smoke artifacts archived in models/smoke/)
runs/       one directory per training run: config.json + log.jsonl
```

Setup (already done on this machine — the venv exists):

```bash
python3 -m venv .venv && .venv/bin/pip install torch numpy pytest
```

Sanity check before anything long-running:

```bash
.venv/bin/python -m pytest -q
```

---

## Phase 4: the overnight training run

### Before you start

- `models/` must contain **no stale checkpoints** — the run writes
  `best.pt` and `ckpt_*.pt` there, and calibration later reads *everything*
  matching `ckpt_*.pt`.  The 32-channel smoke checkpoints are already
  archived in `models/smoke/`, so you're clean.
- **Plug the Mac in and leave the lid open.**  The launch command uses
  `caffeinate` to block idle/system sleep, but closing the lid on a laptop
  still sleeps it.
- Know that **there is no resume**: relaunching starts training from
  scratch and overwrites `models/best.pt`.  (Interrupting with Ctrl-C is
  safe — everything already written to `models/` and the log survives.)

### Launch

```bash
caffeinate -is .venv/bin/python scripts/run_training.py --iters 400 --draw-penalty 0.1 --run-dir runs/overnight 2>&1 | tee runs/overnight.out
```

- `--draw-penalty 0.1` gives draws a slightly negative value target for
  both sides.  The smoke run was ~75% draws; without a penalty the value
  head would spend the early hours learning the constant 0.
- `--iters 400` is an upper bound, not a commitment — you will trim it
  after measuring throughput (next step), and you can Ctrl-C any morning
  the log says it has plateaued.

### First 15 minutes: measure, then decide

Each log line reports `selfplay_sec` and `train_sec` per iteration.  After
2–3 iterations, compute roughly:

```
hours ≈ iters × (selfplay_sec + train_sec + gate_sec/5) / 3600
```

(gating runs every 5th iteration).  If 400 iterations lands outside your
8–12 h budget, Ctrl-C now — a restart this early costs nothing — and
relaunch with a matching `--iters`, or reduce `--games-per-iter` /
`--sims` (per plan.md, only adjust dials if throughput is clearly off in
the first hour).

While you're watching, check the `winners=` counts in the stream: some
attacker and defender wins should appear, not 100% draws.

### Monitoring during the night (all optional)

Live stream:

```bash
tail -f runs/overnight.out
```

Gate history — one line per gating match, `PROMOTED` when the new net
displaced the reigning best:

```bash
.venv/bin/python -c "
import json
for line in open('runs/overnight/log.jsonl'):
    e = json.loads(line)
    if 'gate_score' in e:
        print(f\"iter {e['iter']:>3}  games {e['total_games']:>5}  \"
              f\"gate {e['gate_score']:.2f}  elo {e['gate_elo']:>6}  \"
              + ('PROMOTED' if e.get('promoted') else ''))
"
```

### When is it done?

Plateau criterion (plan.md): new checkpoints stop beating the reigning
best — gate score **< 0.55 for several consecutive gates** (no `PROMOTED`
lines).  The run stops itself after `--iters`; if it's still going in the
morning but the last ~4 gates show no promotion, Ctrl-C it — the reigning
`best.pt` and all gate-iteration checkpoints are already on disk.

Afterwards, confirm the harvest:

```bash
ls -la models/
```

You want `best.pt` (+ `.json` sidecar) and a spread of `ckpt_*.pt` — the
calibration ladder needs early *and* late checkpoints.

---

## After Phase 4: calibrate the difficulty levels

This turns the checkpoint pile into `models/levels.json` — five graded
opponents per side, measured within-side against a fixed opponent panel so
the game's structural side advantage cancels (see `tafl/calibrate.py`).

Preview the schedule (free, prints the candidate pool and game count):

```bash
.venv/bin/python scripts/calibrate_levels.py --dry-run
```

Run it (~1500 games at these settings; expect an hour or more, dominated
by the 150–400-sim candidates — the progress lines report s/game so you
can extrapolate after a minute):

```bash
.venv/bin/python scripts/calibrate_levels.py --games 12 --workers 8
```

Reading the output:

- The candidate table should be roughly monotone: random at the bottom,
  `best/400sims` at the top, and — **the real success criterion for the
  whole training run — the net configs above `heuristic`.**  (On the smoke
  checkpoints the heuristic beat every net; real training must invert
  that.)
- `WARNING: ... beats level N-1 by only ...` means adjacent rungs weren't
  separable at this sample size.  A candidate's score SE is
  ~`sqrt(0.25 / games-per-side)` (≈0.07 at `--games 12`), so a couple of
  sub-margin steps are probably noise — re-run with more `--games` before
  concluding the ladder is actually compressed.
- `levels.json` is whitelisted in `.gitignore` — commit the calibrated
  one.

Then: Phase 5 (balance report) and Phases 6–7 (web app), per
[plan.md](plan.md).
