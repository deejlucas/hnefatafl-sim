"""Calibration of the 1-5 difficulty ladder (Phase 3).

The game is structurally asymmetric, so a win rate between two agents on
opposite sides mixes strength with side advantage and cannot order a
ladder.  Calibration therefore measures **within a side**:

  * a fixed *panel* of opponents is assembled on the opposite side (the
    heuristic baseline plus net agents of assorted strengths);
  * every candidate configuration for the side being calibrated plays the
    same panel, from the same seeds, so the comparison between candidates
    is paired and the side advantage is a constant that cancels;
  * the five levels are then chosen from the measured ordering, spread as
    evenly as possible over the observed score range, with level N
    required to outscore level N-1 by `min_margin`.

Absolute score levels are *not* comparable across sides (attackers scoring
0.4 against their panel and defenders scoring 0.6 against theirs says
nothing about the ladder); only the within-side ordering is used.  The
fair-fight baseline that makes cross-side numbers interpretable comes out
of the Phase 5 balance study.

`calibrate` returns the levels.json document; `scripts/calibrate_levels.py`
is the CLI wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .agents import (CheckpointInfo, N_LEVELS, describe_spec,
                     list_checkpoints, side_name, spec)
from .engine import ATT, DEF
from .eval import AgentMatchConfig, Job, run_jobs, score_for


@dataclass
class CalibrationConfig:
    # A candidate's score comes from `games * len(panel)` games per side;
    # its standard error is roughly sqrt(0.25 / that).  At games=8 (32
    # games/side, SE ~0.09) a min_margin miss is still noise-dominated:
    # treat warnings as "re-run with more games", not as a verdict.
    games: int = 8                 # games per (candidate, panel opponent)
    min_margin: float = 0.05       # required score gap between levels
    workers: int = 8
    seed: int = 0
    panel_sims: int = 60           # search budget of the strongest panel member
    sims_ladder: tuple = (0, 25, 60, 150, 400)
    randomness_ladder: tuple = (0.75, 0.5, 0.25, 0.1)
    match: AgentMatchConfig = field(default_factory=AgentMatchConfig)


# --- candidate pool and panel ---------------------------------------------

def spread_checkpoints(ckpts: list[CheckpointInfo], k: int = 4) -> list[str]:
    """`k` checkpoint names sampled evenly across the training curve.

    A long run leaves dozens of checkpoints; taking a few spread over the
    whole curve gives the ladder genuinely different *understanding* levels
    (the first dial) instead of only different search budgets.
    """
    if not ckpts:
        raise ValueError("no checkpoints found; train first")
    if len(ckpts) <= k:
        names = [c.name for c in ckpts]
    else:
        idx = sorted({round(i * (len(ckpts) - 1) / (k - 1))
                      for i in range(k)})
        names = [ckpts[i].name for i in idx]
    while len(names) < k:                 # pad by repeating the weakest
        names.insert(0, names[0])
    return names


def candidate_pool(ckpts: list[CheckpointInfo],
                   cfg: CalibrationConfig) -> list[dict]:
    """Configurations to measure, ordered by *expected* strength.

    The pool walks both dials: checkpoints sampled across the training
    curve (weak understanding -> strong) crossed with randomness, then the
    best checkpoint at a growing search budget.  The expected order is only
    a prior -- measured scores decide the ladder -- but keeping the pool
    ordered makes the calibration table readable and breaks ties sensibly.

    Cost note: the pool size multiplies the schedule, and the high-sims
    candidates at the end dominate wall time.  Trim `sims_ladder` when the
    calibration budget is tight.
    """
    c0, c1, c2, best = spread_checkpoints(ckpts, 4)
    rnd = cfg.randomness_ladder
    out = [spec("random"),
           spec("policy", c0, randomness=rnd[0], temperature=1.0),
           spec("policy", c0, randomness=rnd[1], temperature=1.0),
           spec("policy", c1, randomness=rnd[2], temperature=1.0),
           spec("policy", c1, randomness=rnd[3], temperature=0.6),
           spec("policy", c2, temperature=0.6),
           spec("policy", best, temperature=0.0),
           spec("heuristic"),
           spec("mcts", c1, sims=cfg.panel_sims),
           spec("mcts", c2, sims=cfg.panel_sims)]
    out += [spec("mcts", best, sims=s) for s in cfg.sims_ladder if s > 0]
    return out


def panel_specs(ckpts: list[CheckpointInfo],
                cfg: CalibrationConfig) -> list[dict]:
    """The fixed opponent panel, from weak to strong.

    Assorted strengths matter: a panel of only strong opponents floors
    every weak candidate at 0 (and a weak-only panel ceilings the strong
    ones), which would collapse the ladder into ties.
    """
    first, _c1, _c2, best = spread_checkpoints(ckpts, 4)
    return [spec("policy", first, randomness=0.5, temperature=1.0),
            spec("heuristic"),
            spec("policy", best, temperature=0.0),
            spec("mcts", best, sims=cfg.panel_sims)]


# --- measurement ----------------------------------------------------------

def build_jobs(candidates: list[dict], panel: list[dict],
               cfg: CalibrationConfig) -> list[Job]:
    """Every (side, candidate, panel opponent, game) pairing.

    The seed depends on the opponent and game index but *not* on the
    candidate, so every candidate meets the panel under identical
    conditions.
    """
    jobs = []
    for side in (ATT, DEF):
        for ci, cand in enumerate(candidates):
            for pi, opp in enumerate(panel):
                for g in range(cfg.games):
                    seed = cfg.seed * 1_000_003 + pi * 10_007 + g * 101 + side
                    att, dfn = ((cand, opp) if side == ATT else (opp, cand))
                    jobs.append(Job((side, ci), att, dfn, seed))
    return jobs


def tally(results: list[dict], n_candidates: int) -> dict:
    """(side, candidate index) -> {score, games, wins, draws, losses, ...}."""
    out = {(side, ci): {"score": 0.0, "games": 0, "w": 0, "d": 0, "l": 0,
                        "plies": 0, "reasons": {}}
           for side in (ATT, DEF) for ci in range(n_candidates)}
    for r in results:
        side, _ci = r["key"]
        rec = out[tuple(r["key"])]
        s = score_for(side, r["winner"])
        rec["score"] += s
        rec["games"] += 1
        rec["plies"] += r["plies"]
        rec["w" if s == 1.0 else ("d" if s == 0.5 else "l")] += 1
        rec["reasons"][r["reason"]] = rec["reasons"].get(r["reason"], 0) + 1
    for rec in out.values():
        n = max(rec["games"], 1)
        rec["score_rate"] = rec["score"] / n
        rec["avg_len"] = rec["plies"] / n
    return out


# --- ladder selection -----------------------------------------------------

def select_ladder(scores: list[float], n_levels: int = N_LEVELS) -> list[int]:
    """Pick `n_levels` candidate indices with increasing measured score.

    Candidates are sorted by measured score (ties broken by the pool's
    a-priori strength order), then the level targets are placed evenly
    across the observed score range and the nearest unused candidate is
    taken for each.  Picks stay in sorted order, so the ladder is monotone
    by construction; the *size* of each step is what `margins` reports and
    what `min_margin` checks.
    """
    if not scores:
        raise ValueError("no candidates")
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    lo, hi = scores[order[0]], scores[order[-1]]
    picked, start = [], 0
    for k in range(n_levels):
        target = lo + (hi - lo) * k / max(n_levels - 1, 1)
        # leave one candidate per remaining level so picks stay distinct
        last = max(start, len(order) - (n_levels - k))
        window = range(start, min(last, len(order) - 1) + 1)
        j = min(window, key=lambda j: (abs(scores[order[j]] - target), j))
        picked.append(order[j])
        start = min(j + 1, len(order) - 1)
    return picked


def margins(scores: list[float]) -> list[float | None]:
    """Score gap from the level below (None for level 1)."""
    return [None] + [round(scores[i] - scores[i - 1], 4)
                     for i in range(1, len(scores))]


# --- top level ------------------------------------------------------------

def calibrate(models_dir="models", cfg: CalibrationConfig | None = None,
              on_result=None) -> dict:
    """Measure the pool, choose the ladders, return the levels.json document."""
    from .agents import build_levels

    cfg = cfg or CalibrationConfig()
    ckpts = list_checkpoints(models_dir)
    candidates = candidate_pool(ckpts, cfg)
    panel = panel_specs(ckpts, cfg)
    jobs = build_jobs(candidates, panel, cfg)
    results = run_jobs(jobs, models_dir, cfg.match, workers=cfg.workers,
                       on_result=on_result)
    stats = tally(results, len(candidates))

    ladders, evidence, warnings = {}, {}, []
    for side in (ATT, DEF):
        rates = [stats[(side, ci)]["score_rate"]
                 for ci in range(len(candidates))]
        chosen = select_ladder(rates, N_LEVELS)
        chosen_rates = [rates[i] for i in chosen]
        gaps = margins(chosen_rates)
        ladder = []
        for lvl, (ci, gap) in enumerate(zip(chosen, gaps), start=1):
            s = dict(candidates[ci])
            s["level"] = lvl
            s["panel_score"] = round(rates[ci], 4)
            s["margin"] = gap
            s["record"] = {k: stats[(side, ci)][k]
                           for k in ("w", "d", "l", "games")}
            ladder.append(s)
            if gap is not None and gap < cfg.min_margin:
                n = stats[(side, ci)]["games"]
                warnings.append(
                    f"{side_name(side)} level {lvl} ({describe_spec(s)}) beats "
                    f"level {lvl - 1} by only {gap:+.3f} over {n} games/side "
                    f"(< min_margin {cfg.min_margin}; score SE at that n is "
                    f"~{(0.25 / max(n, 1)) ** 0.5:.3f})")
        ladders[side_name(side)] = ladder
        evidence[side_name(side)] = [
            {**candidates[ci], "panel_score": round(rates[ci], 4),
             "record": {k: stats[(side, ci)][k] for k in ("w", "d", "l")},
             "avg_len": round(stats[(side, ci)]["avg_len"], 1),
             "reasons": stats[(side, ci)]["reasons"]}
            for ci in range(len(candidates))]

    panels = {side_name(ATT): panel, side_name(DEF): panel}
    doc = build_levels(ladders, panels,
                       {"calibrated": True,
                        # False = the measurement could not separate every
                        # rung (see "warnings"); the ladder then falls back
                        # on the pool's a-priori strength order
                        "ladder_ok": not warnings,
                        "method": {"games_per_opponent": cfg.games,
                                   "min_margin": cfg.min_margin,
                                   "seed": cfg.seed,
                                   "move_limit": cfg.match.move_limit,
                                   "opening_plies": cfg.match.opening_plies,
                                   "opening_temperature":
                                       cfg.match.opening_temperature,
                                   "games_played": len(results)},
                        "checkpoints": [{"name": c.name, "iter": c.iter,
                                         "games": c.games, "elo": c.elo}
                                        for c in ckpts],
                        "candidates": evidence,
                        "warnings": warnings})
    return doc


def format_table(doc: dict) -> str:
    """Human-readable calibration summary (printed by the CLI)."""
    lines = []
    for side in (ATT, DEF):
        name = side_name(side)
        lines.append(f"\n{name} candidates (score vs fixed panel):")
        for c in doc["candidates"][name]:
            r = c["record"]
            lines.append(f"  {c['panel_score']:.3f}  {c['label']:<28}"
                         f" W{r['w']}/D{r['d']}/L{r['l']}"
                         f"  len {c['avg_len']}")
        lines.append(f"{name} ladder:")
        for s in doc["levels"][name]:
            gap = "" if s["margin"] is None else f"  (+{s['margin']:.3f})"
            lines.append(f"  L{s['level']}  {s['panel_score']:.3f}  "
                         f"{s['label']}{gap}")
    for w in doc.get("warnings", []):
        lines.append(f"WARNING: {w}")
    return "\n".join(lines)
