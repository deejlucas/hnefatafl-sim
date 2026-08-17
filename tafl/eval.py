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
import queue as queue_mod
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .engine import ATT, DEF, MOVE_LIMIT, Result, apply, initial_state
from .mcts import MCTS


@dataclass
class ArenaConfig:
    sims: int = 150                # keep at or above the ~100 legal moves at
                                   # a typical root: below that PUCT fans out
                                   # instead of concentrating, root visits end
                                   # up tied, and every game shuffles to a draw
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
        different diagnostics.  `reasons` separates those draws further:
        a match that is all repetition/move limit means the search never
        resolved anything, not that the two agents are evenly matched."""
        def wdl(results, side):
            w = sum(1 for r in results if r.winner == side)
            d = sum(1 for r in results if r.winner is None)
            return {"w": w, "d": d, "l": len(results) - w - d}
        return {"as_att": wdl(self.a_as_att, ATT),
                "as_def": wdl(self.a_as_def, DEF),
                "reasons": dict(Counter(r.reason for r in
                                        self.a_as_att + self.a_as_def)),
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


WORKER_POLL_SEC = 10.0


def _drain(queue, procs, n: int):
    """Yield `n` results from `queue`, raising instead of hanging forever
    if a worker process dies before delivering its share.  Results a dead
    worker already queued are still drained first (`get` only times out
    once the queue is empty)."""
    got = 0
    while got < n:
        try:
            r = queue.get(timeout=WORKER_POLL_SEC)
        except queue_mod.Empty:
            dead = [p.exitcode for p in procs
                    if not p.is_alive() and p.exitcode != 0]
            if dead:
                raise RuntimeError(
                    f"match worker died (exitcode {dead[0]}) after "
                    f"{got}/{n} results") from None
            if all(not p.is_alive() for p in procs):
                raise RuntimeError(
                    f"all match workers exited but only {got}/{n} "
                    f"results arrived") from None
            continue
        got += 1
        yield r


def _shutdown(procs, error: bool):
    for p in procs:
        if error:
            p.terminate()
        p.join()


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
    try:
        for games in _drain(queue, procs, pairs):
            _tally(out, games)
    except BaseException:
        _shutdown(procs, error=True)
        raise
    _shutdown(procs, error=False)
    return out


def elo_diff(score_rate: float) -> float:
    """Elo difference implied by a score rate, clamped to +-800."""
    s = min(max(score_rate, 1e-3), 1 - 1e-3)
    return max(-800.0, min(800.0, 400.0 * math.log10(s / (1.0 - s))))


# --- agent-vs-agent games (Phase 3 calibration, balance study, web app) ----
#
# The matches above pit two *evaluators* against each other at one shared
# search configuration -- exactly what gating needs.  Calibration instead
# compares agents that differ in checkpoint, search budget and randomness,
# and some of them (the heuristic baseline) have no network at all, so the
# driver below talks to the `tafl.agents.Agent` protocol instead.

@dataclass
class AgentMatchConfig:
    opening_plies: int = 8         # plies played at `opening_temperature`
    opening_temperature: float = 0.7
    move_limit: int = MOVE_LIMIT


def play_agent_game(agent_att, agent_def,
                    cfg: AgentMatchConfig | None = None) -> tuple[Result, int]:
    """One game between two agents.  Returns (result, plies)."""
    cfg = cfg or AgentMatchConfig()
    agents = {ATT: agent_att, DEF: agent_def}
    for a in agents.values():
        a.new_game()
    st = initial_state()
    while st.result is None:
        temp = (cfg.opening_temperature if st.ply < cfg.opening_plies
                else None)
        action = agents[st.to_move].select_action(st, temperature=temp)
        st = apply(st, action, cfg.move_limit)
        for a in agents.values():
            a.observe(action)
    return st.result, st.ply


@dataclass(frozen=True)
class Job:
    """One scheduled agent-vs-agent game.

    `key` is any hashable label the caller wants the result tagged with
    (calibration uses (side, candidate index)); `seed` drives both agents'
    randomness, so re-using the same seed across candidates makes the
    comparison paired.
    """
    key: tuple
    spec_att: dict
    spec_def: dict
    seed: int


def _run_job(job: Job, models_dir, cfg: AgentMatchConfig) -> dict:
    from .agents import make_agent
    a = make_agent(job.spec_att, models_dir,
                   rng=np.random.default_rng(job.seed * 2 + 1),
                   move_limit=cfg.move_limit)
    d = make_agent(job.spec_def, models_dir,
                   rng=np.random.default_rng(job.seed * 2 + 2),
                   move_limit=cfg.move_limit)
    result, plies = play_agent_game(a, d, cfg)
    return {"key": job.key, "winner": result.winner, "reason": result.reason,
            "plies": plies, "seed": job.seed}


def _job_worker(jobs, models_dir, cfg, queue):
    import torch
    torch.set_num_threads(1)
    for job in jobs:
        queue.put(_run_job(job, models_dir, cfg))


def run_jobs(jobs: list[Job], models_dir="models",
             cfg: AgentMatchConfig | None = None, workers: int = 1,
             on_result=None) -> list[dict]:
    """Play every job, fanned out over `workers` processes.

    Agents are rebuilt per job but network checkpoints are cached per
    process (see `agents.evaluator_for`), so the load cost is paid once.
    """
    cfg = cfg or AgentMatchConfig()
    models_dir = str(models_dir)
    out = []
    if workers <= 1 or len(jobs) <= 1:
        for job in jobs:
            r = _run_job(job, models_dir, cfg)
            out.append(r)
            if on_result:
                on_result(r)
        return out
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = []
    for i in range(min(workers, len(jobs))):
        chunk = jobs[i::workers]
        if not chunk:
            continue
        p = ctx.Process(target=_job_worker,
                        args=(chunk, models_dir, cfg, queue), daemon=True)
        p.start()
        procs.append(p)
    try:
        for r in _drain(queue, procs, len(jobs)):
            out.append(r)
            if on_result:
                on_result(r)
    except BaseException:
        _shutdown(procs, error=True)
        raise
    _shutdown(procs, error=False)
    return out


def score_for(side: int, result_winner) -> float:
    """Match points for `side`: 1 win, 0.5 draw, 0 loss."""
    if result_winner is None:
        return 0.5
    return 1.0 if result_winner == side else 0.0
