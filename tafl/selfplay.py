"""Self-play game generation for training.

`play_game` runs one full game of MCTS-vs-itself and returns training
samples: for every position, the board (compact bytes, re-encoded to planes
at training time), the MCTS visit distribution over legal actions (sparse),
and later the final outcome z from that position's side-to-move perspective
(+1 win / -1 loss / -draw_penalty for draws, both sides).

`generate_games` fans games out over worker processes.  Each worker builds
its own copy of the network and runs inference on CPU with a single torch
thread -- the net is small enough that many CPU workers beat one shared GPU
evaluator, and MPS cannot be shared across processes anyway.  Training
itself uses the GPU in the parent (see train.py).
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field

import numpy as np

from .engine import ATT, MOVE_LIMIT, N, Result, initial_state
from .mcts import MCTS


@dataclass
class SelfPlayConfig:
    sims: int = 150                # MCTS simulations per move
    batch_size: int = 8            # leaf batch inside each search
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.1   # ~10 / typical branching factor (~110)
    dirichlet_eps: float = 0.25
    temp_plies: int = 20           # sample moves with τ=1 this long, then argmax
    move_limit: int = MOVE_LIMIT   # NB: encode()'s move-count plane is always
                                   # normalized by the engine MOVE_LIMIT (300);
                                   # train and serve at the same limit or the
                                   # net's horizon plane is miscalibrated
    draw_penalty: float = 0.0      # z = -penalty for both sides on draws


@dataclass
class Sample:
    board_b: bytes                 # engine board, 121 bytes
    to_move: int
    ply: int
    acts: np.ndarray               # legal action ids visited at the root
    probs: np.ndarray              # MCTS visit distribution over acts (τ=1)
    z: float = 0.0                 # outcome, side-to-move perspective


@dataclass
class GameRecord:
    samples: list[Sample]
    result: Result
    plies: int = field(init=False)

    def __post_init__(self):
        self.plies = len(self.samples)


def sample_planes(s: Sample) -> np.ndarray:
    """Rebuild the engine's `encode()` planes from a stored Sample."""
    from .engine import ATTACKER, DEFENDER, KING
    g = np.frombuffer(s.board_b, dtype=np.int8).reshape(N, N)
    planes = np.zeros((5, N, N), dtype=np.float32)
    planes[0] = g == ATTACKER
    planes[1] = g == DEFENDER
    planes[2] = g == KING
    planes[3] = 1.0 if s.to_move == ATT else 0.0
    planes[4] = s.ply / MOVE_LIMIT
    return planes


def play_game(evaluator, cfg: SelfPlayConfig,
              rng: np.random.Generator) -> GameRecord:
    st = initial_state()
    mcts = MCTS(evaluator, c_puct=cfg.c_puct,
                dirichlet_alpha=cfg.dirichlet_alpha,
                dirichlet_eps=cfg.dirichlet_eps,
                batch_size=cfg.batch_size, move_limit=cfg.move_limit, rng=rng)
    mcts.set_root(st)
    samples: list[Sample] = []
    while st.result is None:
        mcts.run(cfg.sims, noise=True)
        acts, probs = mcts.root_policy(1.0)      # training target is always τ=1
        samples.append(Sample(st.board_b, st.to_move, st.ply,
                              acts.copy(), probs.astype(np.float32)))
        if st.ply < cfg.temp_plies:
            action = int(rng.choice(acts, p=probs))
        else:
            action = int(acts[int(np.argmax(probs))])
        mcts.advance(action)
        st = mcts.root.state
    w = st.result.winner
    for s in samples:
        if w is None:
            s.z = -cfg.draw_penalty
        else:
            s.z = 1.0 if s.to_move == w else -1.0
    return GameRecord(samples, st.result)


# --- multiprocess generation ----------------------------------------------

def _worker(state_dict, net_config, cfg: SelfPlayConfig, n_games: int,
            seed: int, queue):
    import torch
    torch.set_num_threads(1)
    from .net import NetEvaluator, PolicyValueNet
    net = PolicyValueNet(**net_config)
    net.load_state_dict(state_dict)
    evaluator = NetEvaluator(net, torch.device("cpu"))
    rng = np.random.default_rng(seed)
    for _ in range(n_games):
        queue.put(play_game(evaluator, cfg, rng))


def generate_games(state_dict, net_config: dict, cfg: SelfPlayConfig,
                   n_games: int, workers: int, seed: int = 0,
                   on_game=None) -> list[GameRecord]:
    """Generate `n_games` self-play games with `workers` processes.

    `state_dict` must be a CPU state dict.  With workers <= 1 the games run
    inline in this process (used by tests and easier to debug).
    """
    if workers <= 1:
        import torch
        from .net import NetEvaluator, PolicyValueNet
        net = PolicyValueNet(**net_config)
        net.load_state_dict(state_dict)
        evaluator = NetEvaluator(net, torch.device("cpu"))
        rng = np.random.default_rng(seed)
        records = []
        for _ in range(n_games):
            rec = play_game(evaluator, cfg, rng)
            records.append(rec)
            if on_game:
                on_game(rec)
        return records

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    counts = [n_games // workers + (1 if i < n_games % workers else 0)
              for i in range(workers)]
    procs = []
    for i, cnt in enumerate(counts):
        if cnt == 0:
            continue
        p = ctx.Process(target=_worker,
                        args=(state_dict, net_config, cfg, cnt,
                              seed * 100_003 + i, queue),
                        daemon=True)
        p.start()
        procs.append(p)
    records = []
    for _ in range(n_games):
        rec = queue.get()
        records.append(rec)
        if on_game:
            on_game(rec)
    for p in procs:
        p.join()
    return records
