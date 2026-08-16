"""Head-to-head evaluation: paired, color-balanced matches.

Fetlar is structurally asymmetric, so every strength comparison here plays
*pairs* of games: the same two agents swap colors, and the score is summed
over the pair.  The systematic side advantage then cancels out of any
A-vs-B comparison (see plan.md, Phase 3).

Games are varied by sampling the first `temp_plies` moves from the visit
distribution at a small temperature; after that both sides play argmax.
No Dirichlet noise is used in arena games.
"""

from __future__ import annotations

import math
import multiprocessing as mp
from dataclasses import dataclass, field

import numpy as np

from .engine import ATT, DEF, MOVE_LIMIT, Result, initial_state
from .mcts import MCTS


@dataclass
class ArenaConfig:
    sims: int = 100
    batch_size: int = 8
    c_puct: float = 1.5
    temp_plies: int = 8            # opening variety
    temperature: float = 0.7
    move_limit: int = MOVE_LIMIT


@dataclass
class MatchResult:
    score_a: float = 0.0           # 1 per win, 0.5 per draw for agent A
    games: int = 0
    a_as_att: list[Result] = field(default_factory=list)
    a_as_def: list[Result] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)

    @property
    def score_rate(self) -> float:
        return self.score_a / self.games if self.games else 0.5

    def summary(self) -> dict:
        """Win/draw/loss breakdown per color for agent A, for logging --
        a 0.5 score rate from all draws and one from split wins are very
        different diagnostics."""
        def wdl(results, side):
            w = sum(1 for r in results if r.winner == side)
            d = sum(1 for r in results if r.winner is None)
            return {"w": w, "d": d, "l": len(results) - w - d}
        return {"as_att": wdl(self.a_as_att, ATT),
                "as_def": wdl(self.a_as_def, DEF),
                "avg_len": round(float(np.mean(self.lengths)), 1)
                if self.lengths else 0.0}


def play_game(ev_att, ev_def, cfg: ArenaConfig,
              rng: np.random.Generator) -> tuple[Result, int]:
    """One game, `ev_att` playing the attackers.  Returns (result, plies)."""
    st = initial_state()
    trees = {ATT: MCTS(ev_att, c_puct=cfg.c_puct, dirichlet_eps=0.0,
                       batch_size=cfg.batch_size, move_limit=cfg.move_limit,
                       rng=rng),
             DEF: MCTS(ev_def, c_puct=cfg.c_puct, dirichlet_eps=0.0,
                       batch_size=cfg.batch_size, move_limit=cfg.move_limit,
                       rng=rng)}
    for t in trees.values():
        t.set_root(st)
    while st.result is None:
        mover = trees[st.to_move]
        mover.run(cfg.sims)
        temp = cfg.temperature if st.ply < cfg.temp_plies else 0.0
        acts, probs = mover.root_policy(temp)
        action = int(rng.choice(acts, p=probs))
        mover.advance(action)
        opp = trees[DEF if st.to_move == ATT else ATT]
        if opp.root.expanded:        # keep the opponent's subtree when it
            opp.advance(action)      # has one under this reply
        else:
            opp.set_root(mover.root.state)
        st = mover.root.state
    return st.result, st.ply


def _play_pair(ev_a, ev_b, cfg: ArenaConfig, seed: int):
    """One color-balanced pair.  Returns [(a_side, result, plies), ...]."""
    rng = np.random.default_rng(seed)
    out = []
    result, plies = play_game(ev_a, ev_b, cfg, rng)
    out.append((ATT, result, plies))
    result, plies = play_game(ev_b, ev_a, cfg, rng)
    out.append((DEF, result, plies))
    return out


def _tally(out: MatchResult, games):
    for a_side, result, plies in games:
        (out.a_as_att if a_side == ATT else out.a_as_def).append(result)
        if result.winner is None:
            out.score_a += 0.5
        elif result.winner == a_side:
            out.score_a += 1.0
        out.games += 1
        out.lengths.append(plies)


def _pair_seeds(pairs: int, seed: int):
    return [seed * 1_000_003 + k for k in range(pairs)]


def play_match(ev_a, ev_b, pairs: int, cfg: ArenaConfig,
               seed: int = 0) -> MatchResult:
    """`pairs` color-balanced game pairs between agents A and B."""
    out = MatchResult()
    for s in _pair_seeds(pairs, seed):
        _tally(out, _play_pair(ev_a, ev_b, cfg, s))
    return out


def _match_worker(sd_a, sd_b, net_config, cfg, seeds, queue):
    import torch
    torch.set_num_threads(1)
    from .net import NetEvaluator, PolicyValueNet

    def make(sd):
        net = PolicyValueNet(**net_config)
        net.load_state_dict(sd)
        return NetEvaluator(net, torch.device("cpu"))

    ev_a, ev_b = make(sd_a), make(sd_b)
    for s in seeds:
        queue.put(_play_pair(ev_a, ev_b, cfg, s))


def play_match_mp(sd_a, sd_b, net_config: dict, pairs: int, cfg: ArenaConfig,
                  workers: int = 1, seed: int = 0) -> MatchResult:
    """`play_match` from CPU state dicts, with pairs fanned out over worker
    processes so gating does not serialize the training loop (a full gate at
    default settings costs minutes of otherwise-idle CPU time)."""
    seeds = _pair_seeds(pairs, seed)
    out = MatchResult()
    workers = min(workers, pairs)
    if workers <= 1:
        import torch
        from .net import NetEvaluator, PolicyValueNet

        def make(sd):
            net = PolicyValueNet(**net_config)
            net.load_state_dict(sd)
            return NetEvaluator(net, torch.device("cpu"))
        ev_a, ev_b = make(sd_a), make(sd_b)
        for s in seeds:
            _tally(out, _play_pair(ev_a, ev_b, cfg, s))
        return out
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = []
    for i in range(workers):
        chunk = seeds[i::workers]
        if not chunk:
            continue
        p = ctx.Process(target=_match_worker,
                        args=(sd_a, sd_b, net_config, cfg, chunk, queue),
                        daemon=True)
        p.start()
        procs.append(p)
    for _ in range(pairs):
        _tally(out, queue.get())
    for p in procs:
        p.join()
    return out


def elo_diff(score_rate: float) -> float:
    """Elo difference implied by a score rate, clamped to +-800."""
    s = min(max(score_rate, 1e-3), 1 - 1e-3)
    return max(-800.0, min(800.0, 400.0 * math.log10(s / (1.0 - s))))
