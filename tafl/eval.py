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


def play_match(ev_a, ev_b, pairs: int, cfg: ArenaConfig,
               seed: int = 0) -> MatchResult:
    """`pairs` color-balanced game pairs between agents A and B."""
    out = MatchResult()
    for k in range(pairs):
        rng = np.random.default_rng(seed * 1_000_003 + k)
        for a_side in (ATT, DEF):
            if a_side == ATT:
                result, plies = play_game(ev_a, ev_b, cfg, rng)
                out.a_as_att.append(result)
            else:
                result, plies = play_game(ev_b, ev_a, cfg, rng)
                out.a_as_def.append(result)
            if result.winner is None:
                out.score_a += 0.5
            elif result.winner == a_side:
                out.score_a += 1.0
            out.games += 1
            out.lengths.append(plies)
    return out


def elo_diff(score_rate: float) -> float:
    """Elo difference implied by a score rate, clamped to +-800."""
    s = min(max(score_rate, 1e-3), 1 - 1e-3)
    return max(-800.0, min(800.0, 400.0 * math.log10(s / (1.0 - s))))
