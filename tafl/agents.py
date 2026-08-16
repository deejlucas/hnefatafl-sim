"""Playable agents and the persisted difficulty ladder (Phase 3).

Everything that *plays* the game outside of training lives here: the agent
interface used by the arena, the calibration script and (later) the web
server, plus the checkpoint registry and `models/levels.json`.

Agents
 - `RandomAgent`      uniform legal moves; the floor of the ladder.
 - `HeuristicAgent`   1-ply search over a material + king-distance
                      evaluation; a *fixed* external baseline that never
                      changes as training progresses, so it measures
                      absolute progress (unlike gate scores, which only
                      compare a net to its own predecessor).
 - `PolicyAgent`      the network's raw policy, no search.
 - `MCTSAgent`        PUCT search at a configurable simulation budget, with
                      tree reuse across moves.

Three dials set an agent's strength: which **checkpoint** it uses, its
**sims** budget, and a **randomness** dial (probability of playing a
uniformly random legal move instead of its own choice).  The randomness
dial is what keeps level 1 beatable for a beginner no matter how strong the
earliest surviving checkpoint turns out to be, and it gives calibration a
continuous knob for spreading the ladder.

Protocol.  The match driver (`tafl.eval.play_agent_game`) calls
`select_action(state)` on the side to move and then `observe(action)` on
*both* agents, which is how `MCTSAgent` keeps its subtree.  An optional
`temperature` override on `select_action` lets the driver randomize the
opening plies without changing the agents' configured strength.

Specs.  An agent is described by a plain JSON dict so it can be persisted
in `levels.json` and shipped to worker processes:

    {"agent": "mcts", "checkpoint": "best.pt", "sims": 200,
     "temperature": 0.0, "randomness": 0.0, "label": "best@200"}

`checkpoint` is resolved relative to the models directory (the directory
holding `levels.json`), so the models tree stays relocatable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .engine import (ATT, DEF, IS_CORNER, MOVE_LIMIT, N, RAYS, State, apply,
                     legal_actions)
from .mcts import MCTS

DEFAULT_LEVELS_PATH = Path("models") / "levels.json"
N_LEVELS = 5

SIDE_NAMES = {ATT: "attacker", DEF: "defender"}
SIDE_IDS = {"attacker": ATT, "defender": DEF, "att": ATT, "def": DEF}


def side_id(side) -> int:
    """Accept ATT/DEF or 'attacker'/'defender' and return the engine id."""
    if isinstance(side, str):
        try:
            return SIDE_IDS[side.lower()]
        except KeyError:
            raise ValueError(f"unknown side {side!r}") from None
    if side not in (ATT, DEF):
        raise ValueError(f"unknown side {side!r}")
    return side


def side_name(side) -> str:
    return SIDE_NAMES[side_id(side)]


# --- agents ---------------------------------------------------------------

class Agent:
    """Base class; subclasses implement `_choose`."""

    def __init__(self, name: str = "agent", randomness: float = 0.0,
                 temperature: float = 0.0,
                 rng: np.random.Generator | None = None):
        self.name = name
        self.randomness = float(randomness)
        self.temperature = float(temperature)
        self.rng = rng if rng is not None else np.random.default_rng()

    # -- protocol ----------------------------------------------------------

    def new_game(self):
        """Reset per-game state (called before the first move)."""

    def observe(self, action: int):
        """Told about every move played, by either side."""

    def select_action(self, state: State,
                      temperature: float | None = None) -> int:
        if state.result is not None:
            raise ValueError("game is over")
        acts = legal_actions(state)
        if self.randomness > 0.0 and self.rng.random() < self.randomness:
            return int(self.rng.choice(acts))
        t = self.temperature if temperature is None else float(temperature)
        return self._choose(state, acts, t)

    def _choose(self, state: State, acts: np.ndarray, temperature: float) -> int:
        raise NotImplementedError

    # -- helpers -----------------------------------------------------------

    def _pick(self, acts: np.ndarray, scores: np.ndarray,
              temperature: float, scale: float = 1.0) -> int:
        """argmax at τ≈0, else sample from softmax(scores / (τ*scale))."""
        if temperature < 0.05:
            best = np.flatnonzero(scores >= scores.max() - 1e-9)
            return int(acts[self.rng.choice(best)])
        z = scores / max(temperature * scale, 1e-6)
        z -= z.max()
        p = np.exp(z)
        return int(self.rng.choice(acts, p=p / p.sum()))

    def __repr__(self):
        return f"<{type(self).__name__} {self.name}>"


class RandomAgent(Agent):
    def __init__(self, name: str = "random", **kw):
        kw.pop("randomness", None)
        super().__init__(name=name, randomness=1.0, **kw)

    def _choose(self, state, acts, temperature):   # pragma: no cover
        return int(self.rng.choice(acts))          # randomness=1.0 short-circuits


# 1-ply evaluation weights, all from the *attacker's* perspective.
W_ATTACKER = 1.0        # attacker piece
W_DEFENDER = 2.0        # defender piece (12 defenders vs 24 attackers)
W_KING_DIST = 0.35      # per step of king distance from the nearest corner
W_ESCAPE_THREAT = 6.0   # per corner the king can reach in one move
W_WIN = 1e6


def king_escape_threats(state: State) -> int:
    """How many corners the king can move to right now (clear rook path)."""
    b = state.board_b
    n = 0
    for ray in RAYS[state.king]:
        for t in ray:
            if b[t]:
                break
            if IS_CORNER[t]:
                n += 1
    return n


def king_corner_distance(state: State) -> int:
    """Manhattan distance from the king to the nearest corner."""
    r, c = divmod(state.king, N)
    return min(r, N - 1 - r) + min(c, N - 1 - c)


def heuristic_value(state: State) -> float:
    """Static evaluation from the attacker's perspective (+ = attackers better).

    Material (defenders weighted double, so the start position scores 0),
    plus how far the king is from a corner, minus immediate escape threats.
    Terminal positions dominate everything else.
    """
    if state.result is not None:
        w = state.result.winner
        if w is None:
            return 0.0
        return W_WIN if w == ATT else -W_WIN
    return (W_ATTACKER * len(state.atts)
            - W_DEFENDER * len(state.defs)
            + W_KING_DIST * king_corner_distance(state)
            - W_ESCAPE_THREAT * king_escape_threats(state))


class HeuristicAgent(Agent):
    """1-ply greedy search over `heuristic_value`.

    Fixed reference opponent: it never changes, so scores against it are
    comparable across the whole training run.
    """

    def __init__(self, name: str = "heuristic", move_limit: int = MOVE_LIMIT,
                 **kw):
        super().__init__(name=name, **kw)
        self.move_limit = move_limit

    def _choose(self, state, acts, temperature):
        sign = 1.0 if state.to_move == ATT else -1.0
        scores = np.array(
            [sign * heuristic_value(apply(state, int(a), self.move_limit))
             for a in acts], dtype=np.float64)
        # tiny jitter so equal-scoring moves are not always the same one
        scores += self.rng.random(len(scores)) * 1e-3
        return self._pick(acts, scores, temperature, scale=2.0)


class PolicyAgent(Agent):
    """Plays the network's policy head directly (no search)."""

    def __init__(self, evaluator, name: str = "policy", **kw):
        super().__init__(name=name, **kw)
        self.evaluator = evaluator

    def _choose(self, state, acts, temperature):
        (_acts, priors, _v), = self.evaluator.evaluate([state])
        p = np.asarray(priors, dtype=np.float64)
        if temperature < 0.05:
            best = np.flatnonzero(p >= p.max() - 1e-12)
            return int(acts[self.rng.choice(best)])
        p = np.maximum(p, 1e-12) ** (1.0 / temperature)
        return int(self.rng.choice(acts, p=p / p.sum()))


class MCTSAgent(Agent):
    """PUCT search at a fixed simulation budget, reusing the tree.

    No Dirichlet noise: exploration noise belongs to self-play, not to a
    graded opponent.  Variety comes from `temperature` / `randomness` and
    from the driver's opening temperature.
    """

    def __init__(self, evaluator, sims: int = 100, c_puct: float = 1.5,
                 batch_size: int = 8, move_limit: int = MOVE_LIMIT,
                 name: str = "mcts", **kw):
        super().__init__(name=name, **kw)
        self.sims = int(sims)
        self.move_limit = move_limit
        self.mcts = MCTS(evaluator, c_puct=c_puct, dirichlet_eps=0.0,
                         batch_size=batch_size, move_limit=move_limit,
                         rng=self.rng)

    def new_game(self):
        self.mcts.root = None

    def observe(self, action: int):
        root = self.mcts.root
        if root is not None and root.expanded and int(action) in root.acts:
            self.mcts.advance(int(action))
        else:
            self.mcts.root = None

    def _choose(self, state, acts, temperature):
        root = self.mcts.root
        if (root is None or root.state.hash != state.hash
                or root.state.ply != state.ply):
            self.mcts.set_root(state)     # tree reuse missed; start fresh
        self.mcts.run(self.sims)
        acts, probs = self.mcts.root_policy(temperature)
        return int(self.rng.choice(acts, p=probs))

    def root_value(self) -> float:
        """Search value of the last position searched (mover's perspective)."""
        return self.mcts.root_value() if self.mcts.root is not None else 0.0


# --- checkpoints ----------------------------------------------------------

class CheckpointInfo:
    """A checkpoint file plus whatever metadata the training run recorded."""

    __slots__ = ("path", "name", "iter", "games", "elo", "saved_at", "meta")

    def __init__(self, path: Path, meta: dict):
        self.path = path
        self.name = path.name
        self.meta = meta
        self.iter = int(meta.get("iter", 0) or 0)
        self.games = int(meta.get("total_games", meta.get("games", 0)) or 0)
        elo = meta.get("gate_elo", meta.get("elo"))
        self.elo = float(elo) if elo is not None else None
        self.saved_at = meta.get("saved_at")

    def __repr__(self):
        return (f"<Checkpoint {self.name} iter={self.iter} "
                f"games={self.games} elo={self.elo}>")


def _checkpoint_meta(path: Path) -> dict:
    """Sidecar JSON if the training run wrote one, else the .pt's own meta."""
    sidecar = path.with_suffix(".json")
    meta = {}
    if sidecar.exists():
        try:
            meta = json.loads(sidecar.read_text())
        except json.JSONDecodeError:
            meta = {}
    if "iter" not in meta:
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        meta = {**{k: v for k, v in ckpt.items()
                   if k not in ("model", "net_config")}, **meta}
    return meta


def list_checkpoints(models_dir="models") -> list[CheckpointInfo]:
    """Checkpoints in `models_dir`, weakest (earliest) first.

    `best.pt` sorts last: it is the reigning gated net, i.e. at least as
    strong as any `ckpt_*.pt` that was promoted into it.
    """
    models_dir = Path(models_dir)
    out = [CheckpointInfo(p, _checkpoint_meta(p))
           for p in sorted(models_dir.glob("ckpt_*.pt"))]
    out.sort(key=lambda c: (c.iter, c.name))
    best = models_dir / "best.pt"
    if best.exists():
        out.append(CheckpointInfo(best, _checkpoint_meta(best)))
    return out


_EVALUATOR_CACHE: dict = {}


def evaluator_for(path, device=None):
    """Cached `NetEvaluator` for a checkpoint (one per path per process)."""
    import torch
    from .net import NetEvaluator, load_checkpoint
    dev = torch.device(device) if device is not None else torch.device("cpu")
    key = (str(Path(path).resolve()), str(dev))
    ev = _EVALUATOR_CACHE.get(key)
    if ev is None:
        net, _ = load_checkpoint(path, dev)
        ev = NetEvaluator(net, dev)
        _EVALUATOR_CACHE[key] = ev
    return ev


def clear_evaluator_cache():
    _EVALUATOR_CACHE.clear()


# --- specs ----------------------------------------------------------------

def spec(agent: str, checkpoint: str | None = None, sims: int = 0,
         temperature: float = 0.0, randomness: float = 0.0,
         label: str | None = None, **extra) -> dict:
    """Build an agent spec dict (the JSON form stored in levels.json)."""
    d = {"agent": agent, "sims": int(sims), "temperature": float(temperature),
         "randomness": float(randomness)}
    if checkpoint is not None:
        d["checkpoint"] = str(checkpoint)
    d.update(extra)
    d["label"] = label or describe_spec(d)
    return d


def describe_spec(s: dict) -> str:
    kind = s.get("agent", "mcts")
    bits = []
    if s.get("checkpoint"):
        bits.append(Path(s["checkpoint"]).stem)
    if kind == "mcts":
        bits.append(f"{int(s.get('sims', 0))}sims")
    else:
        bits.append(kind)
    if s.get("temperature", 0.0) >= 0.05:
        bits.append(f"T{s['temperature']:g}")
    if s.get("randomness", 0.0) > 0.0:
        bits.append(f"rnd{s['randomness']:g}")
    return "/".join(bits)


def make_agent(s: dict, models_dir="models", rng=None, device=None,
               move_limit: int = MOVE_LIMIT) -> Agent:
    """Instantiate the agent described by spec `s`.

    `checkpoint` is resolved relative to `models_dir` unless it is absolute
    or already points at an existing file.
    """
    kind = s.get("agent")
    if kind is None:
        kind = "mcts" if int(s.get("sims", 0)) > 0 else "policy"
    common = {"randomness": float(s.get("randomness", 0.0)),
              "temperature": float(s.get("temperature", 0.0)),
              "rng": rng,
              "name": s.get("label") or describe_spec(s)}
    if kind == "random":
        return RandomAgent(rng=rng, name=common["name"],
                           temperature=common["temperature"])
    if kind == "heuristic":
        return HeuristicAgent(move_limit=move_limit, **common)
    ckpt = s.get("checkpoint")
    if not ckpt:
        raise ValueError(f"spec {s!r} needs a checkpoint")
    path = Path(ckpt)
    if not path.is_absolute() and not path.exists():
        path = Path(models_dir) / path
    ev = evaluator_for(path, device)
    if kind == "policy":
        return PolicyAgent(ev, **common)
    if kind == "mcts":
        return MCTSAgent(ev, sims=int(s.get("sims", 100)),
                         c_puct=float(s.get("c_puct", 1.5)),
                         batch_size=int(s.get("batch_size", 8)),
                         move_limit=move_limit, **common)
    raise ValueError(f"unknown agent type {kind!r}")


# --- the difficulty ladder ------------------------------------------------

class Levels:
    """`models/levels.json`: (side, level 1-5) -> agent spec."""

    def __init__(self, data: dict, models_dir):
        self.data = data
        self.models_dir = Path(models_dir)

    @classmethod
    def load(cls, path=DEFAULT_LEVELS_PATH) -> "Levels":
        path = Path(path)
        return cls(json.loads(path.read_text()), path.parent)

    def save(self, path=DEFAULT_LEVELS_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.data, indent=2) + "\n")
        return path

    # -- lookup ------------------------------------------------------------

    @property
    def n_levels(self) -> int:
        return len(self.data["levels"][side_name(ATT)])

    def specs(self, side) -> list[dict]:
        return self.data["levels"][side_name(side)]

    def spec(self, side, level: int) -> dict:
        entries = self.specs(side)
        if not 1 <= level <= len(entries):
            raise ValueError(f"level {level} out of range 1..{len(entries)}")
        return entries[level - 1]

    def label(self, side, level: int) -> str:
        s = self.spec(side, level)
        return s.get("label") or describe_spec(s)

    def agent(self, side, level: int, rng=None, device=None,
              move_limit: int = MOVE_LIMIT) -> Agent:
        return make_agent(self.spec(side, level), self.models_dir, rng=rng,
                          device=device, move_limit=move_limit)


def build_levels(ladders: dict, panels: dict, meta: dict | None = None) -> dict:
    """Assemble the levels.json document.

    `ladders[side_name]` is the ordered list of 5 chosen specs (each may
    carry calibration evidence under "panel_score"/"margin"); `panels`
    records the fixed opponent panel each side was measured against.
    """
    import time
    return {"version": 1,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_levels": len(next(iter(ladders.values()))),
            "levels": ladders,
            "panels": panels,
            **(meta or {})}


def default_ladder(ckpts: list[CheckpointInfo], sims=(0, 25, 100, 400),
                   randomness=(0.6, 0.25, 0.0)) -> list[dict]:
    """Uncalibrated fallback ladder, weakest first.

    Used when `levels.json` does not exist yet (and as the seed ordering for
    calibration's candidate pool).  Strength rises by walking the two dials
    in order: first randomness down on the earliest checkpoint, then the
    best checkpoint with a growing search budget.
    """
    if not ckpts:
        raise ValueError("no checkpoints in models dir")
    first, best = ckpts[0].name, ckpts[-1].name
    mid = ckpts[len(ckpts) // 2].name
    out = [spec("policy", first, randomness=randomness[0], temperature=1.0),
           spec("policy", first, randomness=randomness[1], temperature=1.0),
           spec("policy", mid, randomness=randomness[2], temperature=0.6),
           spec("mcts", best, sims=sims[1]),
           spec("mcts", best, sims=sims[3])]
    return out


def default_levels(models_dir="models") -> Levels:
    """A Levels object for a models dir that has never been calibrated."""
    ckpts = list_checkpoints(models_dir)
    ladder = default_ladder(ckpts)
    ladders = {side_name(s): [dict(x) for x in ladder] for s in (ATT, DEF)}
    return Levels(build_levels(ladders, {side_name(ATT): [],
                                         side_name(DEF): []},
                               {"calibrated": False}), models_dir)


def load_levels(path=DEFAULT_LEVELS_PATH, models_dir=None) -> Levels:
    """Load `levels.json`, falling back to an uncalibrated default ladder."""
    path = Path(path)
    if path.exists():
        return Levels.load(path)
    return default_levels(models_dir or path.parent)
