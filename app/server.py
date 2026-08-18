"""FastAPI server for human play (Phases 6/7).

Serves the static board UI and a small JSON API.  Games live in memory,
keyed by id; each side of a game is either a human (moves arrive via
POST /move) or an agent built from `tafl.agents` (moves computed on
POST /ai_move).  The full state history is kept so undo can rewind to the
last human-to-move position; MCTS agents recover from a rewind on their
own because `MCTSAgent._choose` re-roots whenever the position hash does
not match its tree.

Run from the project root:  uvicorn app.server:app --port 8123
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tafl.agents import (HeuristicAgent, RandomAgent, load_levels, side_name)
from tafl.engine import (ATT, DEF, EMPTY, MOVE_LIMIT, N, apply, decode_action,
                         initial_state, legal_actions, make_action)

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT.parent / "models"
LEVELS_PATH = MODELS_DIR / "levels.json"

app = FastAPI(title="hnefatafl")

_PIECE_CHAR = ".ADK"                     # EMPTY, ATTACKER, DEFENDER, KING


# --- agent registry -------------------------------------------------------

def _levels():
    """Levels ladder, or None when there are no usable checkpoints."""
    try:
        return load_levels(LEVELS_PATH, MODELS_DIR)
    except Exception:
        return None


def agent_options() -> list[dict]:
    opts = [{"id": "human", "label": "Human"},
            {"id": "random", "label": "Random"},
            {"id": "heuristic", "label": "Heuristic (1-ply)"},
            {"id": "heuristic-manhattan",
             "label": "Heuristic (classic, blind to blockades)"}]
    lv = _levels()
    if lv is not None:
        for i in range(1, lv.n_levels + 1):
            opts.append({"id": f"level{i}", "label": f"Level {i}",
                         "detail": {side_name(s): lv.label(s, i)
                                    for s in (ATT, DEF)}})
    return opts


def build_agent(agent_id: str, side: int, seed: int):
    """Agent instance for one side of one game, or None for a human."""
    rng = np.random.default_rng(seed)
    if agent_id == "human":
        return None
    if agent_id == "random":
        return RandomAgent(rng=rng)
    if agent_id == "heuristic":
        return HeuristicAgent(rng=rng)
    if agent_id == "heuristic-manhattan":
        return HeuristicAgent(rng=rng, king_metric="manhattan",
                              name="heuristic/manhattan")
    if agent_id.startswith("level"):
        lv = _levels()
        if lv is None:
            raise HTTPException(400, "no checkpoints available for levels")
        try:
            return lv.agent(side, int(agent_id[len("level"):]), rng=rng)
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(400, f"cannot build {agent_id}: {e}")
    raise HTTPException(400, f"unknown agent {agent_id!r}")


# --- games ----------------------------------------------------------------

class Game:
    def __init__(self, att_id: str, def_id: str):
        seed = uuid.uuid4().int & 0xFFFFFFFF
        self.lock = threading.Lock()
        self.agent_ids = {ATT: att_id, DEF: def_id}
        self.agents = {ATT: build_agent(att_id, ATT, seed * 2 + 1),
                       DEF: build_agent(def_id, DEF, seed * 2 + 2)}
        self.states = [initial_state()]
        self.moves: list[dict] = []
        for a in self.agents.values():
            if a is not None:
                a.new_game()

    @property
    def state(self):
        return self.states[-1]

    def push(self, action: int):
        st = self.state
        nxt = apply(st, action, MOVE_LIMIT)
        frm, to = decode_action(action)
        captured = [sq for sq in range(N * N)
                    if sq != frm and st.board_b[sq] != EMPTY
                    and nxt.board_b[sq] == EMPTY]
        self.moves.append({"ply": st.ply, "side": side_name(st.to_move),
                           "from": frm, "to": to, "captured": captured})
        self.states.append(nxt)
        for a in self.agents.values():
            if a is not None:
                a.observe(action)

    def human_move(self, frm: int, to: int):
        st = self.state
        if st.result is not None:
            raise HTTPException(409, "game is over")
        if self.agents[st.to_move] is not None:
            raise HTTPException(409, "it is not a human's turn")
        try:
            action = make_action(frm, to)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if action not in legal_actions(st):
            raise HTTPException(400, f"illegal move {frm}->{to}")
        self.push(action)

    def ai_move(self) -> None:
        st = self.state
        if st.result is not None:
            raise HTTPException(409, "game is over")
        agent = self.agents[st.to_move]
        if agent is None:
            raise HTTPException(409, "a human is to move")
        self.push(agent.select_action(st))

    def undo(self):
        if len(self.states) < 2:
            raise HTTPException(409, "nothing to undo")
        self.states.pop()
        self.moves.pop()
        # rewind through AI replies to the last human-to-move position
        if any(a is None for a in self.agents.values()):
            while (len(self.states) > 1
                   and self.agents[self.state.to_move] is not None):
                self.states.pop()
                self.moves.pop()


GAMES: dict[str, Game] = {}
_GAMES_LOCK = threading.Lock()


def _game(gid: str) -> Game:
    g = GAMES.get(gid)
    if g is None:
        raise HTTPException(404, "no such game")
    return g


def payload(gid: str, g: Game) -> dict:
    st = g.state
    legal: dict[str, list[int]] = {}
    if st.result is None:
        for a in legal_actions(st):
            frm, to = decode_action(int(a))
            legal.setdefault(str(frm), []).append(to)
    result = None
    if st.result is not None:
        result = {"winner": None if st.result.winner is None
                  else side_name(st.result.winner),
                  "reason": st.result.reason}
    return {"game_id": gid,
            "board": "".join(_PIECE_CHAR[b] for b in st.board_b),
            "to_move": side_name(st.to_move),
            "ply": st.ply,
            "move_limit": MOVE_LIMIT,
            "result": result,
            "legal": legal,
            "last_move": g.moves[-1] if g.moves else None,
            "moves": g.moves,
            "agents": {side_name(s): g.agent_ids[s] for s in (ATT, DEF)},
            "human": {side_name(s): g.agents[s] is None for s in (ATT, DEF)}}


# --- api ------------------------------------------------------------------

class NewGame(BaseModel):
    attacker: str = "human"
    defender: str = "human"


class Move(BaseModel):
    frm: int
    to: int


@app.get("/api/meta")
def meta():
    return {"board_size": N, "agents": agent_options()}


@app.post("/api/game")
def new_game(req: NewGame):
    g = Game(req.attacker, req.defender)
    gid = uuid.uuid4().hex[:12]
    with _GAMES_LOCK:
        # keep memory bounded during long play sessions
        while len(GAMES) >= 32:
            GAMES.pop(next(iter(GAMES)))
        GAMES[gid] = g
    return payload(gid, g)


@app.get("/api/game/{gid}")
def get_game(gid: str):
    g = _game(gid)
    with g.lock:
        return payload(gid, g)


@app.post("/api/game/{gid}/move")
def post_move(gid: str, mv: Move):
    g = _game(gid)
    with g.lock:
        g.human_move(mv.frm, mv.to)
        return payload(gid, g)


@app.post("/api/game/{gid}/ai_move")
def post_ai_move(gid: str):
    g = _game(gid)
    with g.lock:
        g.ai_move()
        return payload(gid, g)


@app.post("/api/game/{gid}/undo")
def post_undo(gid: str):
    g = _game(gid)
    with g.lock:
        g.undo()
        return payload(gid, g)


app.mount("/", StaticFiles(directory=ROOT / "static", html=True),
          name="static")
