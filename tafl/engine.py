"""Fetlar Hnefatafl (11x11) game engine.

Rules implemented per https://aagenielsen.dk/fetlar_rules_en.php (re-verified
against the source page on 2026-08-15):

 1. 11x11 board; 24 attackers, 12 defenders + 1 king; attackers move first.
 2. All pieces move any number of vacant squares along a row or column (rook
    moves, no diagonals).
 3. Restricted squares = central throne + 4 corners: only the king may land on
    them.  Any piece may pass through the throne while it is empty; the king
    may re-enter the throne.
 4. Hostility (a hostile square substitutes for one of the two capturing
    pieces): the corners are always hostile to both sides; the throne is
    always hostile to attackers but hostile to defenders only while empty.
    The board edge is NOT hostile.
 5. Capture: a piece (never the king) is captured when the *mover* closes an
    orthogonal sandwich -- enemy piece on one side, enemy piece or hostile
    square on the other.  Moving into a sandwich is safe.  Several pieces can
    be captured in one move (one per direction); pieces standing in a row are
    not captured (Fetlar has no shieldwall rule).  The king takes part in
    captures like any defender piece.
 6. Defenders win when the king reaches any corner square ("escape").
 7a. Attackers win by capturing the king: all 4 orthogonal neighbours are
    attackers, or -- when the king stands next to the throne -- the 3 remaining
    neighbours are attackers.  The king is never captured by a 2-piece
    sandwich and cannot be captured while standing on the board edge (a fully
    boxed-in king on the edge loses via rule 8 "no legal moves" instead,
    exactly as the source rules describe).  King capture is only checked
    after an attacker move: the trap must be closed by the aggressor.
 7b. Attackers win when an unbroken ring of attackers encircles the king and
    ALL remaining defenders.  Implemented as: no defender/king can reach any
    edge square by orthogonal steps through attacker-free squares (flood
    fill).  Only an attacker move or capture can create this, so it is
    checked after attacker moves only.
 8. A player with no legal move on their turn loses.
 9. Draws: threefold repetition of the same position with the same side to
    move, or reaching `move_limit` plies ("it is not possible to end the
    game").

Representation
 - Squares are flat indices ``sq = row*11 + col`` with row 0 at the top.
 - ``State.board`` is a read-only NumPy int8 array of shape (121,) with
   values EMPTY/ATTACKER/DEFENDER/KING.  The canonical immutable storage is
   ``State.board_b`` (``bytes``); the ndarray is a zero-copy view of it.
 - Actions are ints: ``action = from_sq*40 + dir*10 + (dist-1)`` with dir in
   (up, down, left, right) and dist in 1..10, giving an action space of
   121*40 = 4840 (``N_ACTIONS``).  ``legal_actions`` returns the legal action
   ids, ``legal_mask`` the bool vector of length 4840.
 - ``apply(state, action)`` is immutable-style: it returns a new State and
   never mutates its argument (MCTS tree reuse relies on this).
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np

# --- board geometry -------------------------------------------------------

N = 11
N_SQ = N * N                       # 121
N_DIRS = 4                         # up, down, left, right
MAX_DIST = N - 1                   # 10
MOVES_PER_SQ = N_DIRS * MAX_DIST   # 40
N_ACTIONS = N_SQ * MOVES_PER_SQ    # 4840

EMPTY, ATTACKER, DEFENDER, KING = 0, 1, 2, 3
ATT, DEF = 0, 1                    # side indices; ATT moves first

THRONE = (N // 2) * N + N // 2     # 60
CORNERS = (0, N - 1, N_SQ - N, N_SQ - 1)

MOVE_LIMIT = 300                   # plies; game is drawn when reached

STEP = (-N, N, -1, 1)              # flat-index deltas for the 4 directions


def _in_board(r: int, c: int) -> bool:
    return 0 <= r < N and 0 <= c < N


def _build_tables():
    rays, adj2, nb4, nb8 = [], [], [], []
    is_restricted = bytearray(N_SQ)
    is_corner = bytearray(N_SQ)
    is_edge = bytearray(N_SQ)
    for sq in CORNERS:
        is_corner[sq] = 1
        is_restricted[sq] = 1
    is_restricted[THRONE] = 1
    dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for sq in range(N_SQ):
        r, c = divmod(sq, N)
        if r in (0, N - 1) or c in (0, N - 1):
            is_edge[sq] = 1
        per_dir = []
        pairs = []
        n4 = []
        for dr, dc in dirs:
            ray = []
            rr, cc = r + dr, c + dc
            while _in_board(rr, cc):
                ray.append(rr * N + cc)
                rr, cc = rr + dr, cc + dc
            per_dir.append(tuple(ray))
            if len(ray) >= 1:
                n4.append(ray[0])
            if len(ray) >= 2:
                pairs.append((ray[0], ray[1]))
        n8 = list(n4)
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            if _in_board(r + dr, c + dc):
                n8.append((r + dr) * N + (c + dc))
        rays.append(tuple(per_dir))
        adj2.append(tuple(pairs))
        nb4.append(tuple(n4))
        nb8.append(tuple(n8))
    return (tuple(rays), tuple(adj2), tuple(nb4), tuple(nb8),
            bytes(is_restricted), bytes(is_corner), bytes(is_edge))


# RAYS[sq][dir]  -> squares along the ray, nearest first
# ADJ2[sq]       -> ((victim_sq, support_sq), ...) valid neighbour pairs
# NEIGHBORS4[sq] -> orthogonal neighbours (len < 4 on the edge)
# NEIGHBORS8[sq] -> orthogonal + diagonal neighbours
RAYS, ADJ2, NEIGHBORS4, NEIGHBORS8, IS_RESTRICTED, IS_CORNER, IS_EDGE = _build_tables()

# --- Zobrist hashing ------------------------------------------------------

_rng = np.random.default_rng(20260815)
ZP = [[int(x) for x in row]
      for row in _rng.integers(1, 2 ** 63, size=(N_SQ, 4), dtype=np.int64)]
for _row in ZP:
    _row[EMPTY] = 0
Z_SIDE = int(_rng.integers(1, 2 ** 63, dtype=np.int64))  # xored in when DEF to move


class Result(NamedTuple):
    winner: Optional[int]          # ATT, DEF, or None for a draw
    reason: str                    # escape / king captured / encirclement /
                                   # no moves / repetition / move limit


_RESULT_ESCAPE = Result(DEF, "escape")
_RESULT_KING_CAPTURED = Result(ATT, "king captured")
_RESULT_ENCIRCLED = Result(ATT, "encirclement")
_RESULT_REPETITION = Result(None, "repetition")
_RESULT_MOVE_LIMIT = Result(None, "move limit")


class State:
    """Immutable-by-convention game state.  Do not mutate fields."""

    __slots__ = ("board_b", "atts", "defs", "king", "to_move", "ply",
                 "hash", "hist", "result", "_acts", "_acts_np", "_mask",
                 "_board")

    def __init__(self, board_b, atts, defs, king, to_move, ply, h, hist,
                 result):
        self.board_b = board_b     # bytes, len 121
        self.atts = atts           # tuple of attacker squares
        self.defs = defs           # tuple of defender squares (king excluded)
        self.king = king           # king square
        self.to_move = to_move     # ATT or DEF
        self.ply = ply
        self.hash = h              # Zobrist hash incl. side to move
        self.hist = hist           # tuple of position hashes since the last
                                   # capture (repetition detection)
        self.result = result       # Result or None while ongoing
        self._acts = None
        self._acts_np = None
        self._mask = None
        self._board = None

    @property
    def board(self) -> np.ndarray:
        if self._board is None:
            b = np.frombuffer(self.board_b, dtype=np.int8)
            self._board = b        # frombuffer arrays are already read-only
        return self._board

    def __str__(self) -> str:
        return to_ascii(self)

    def __repr__(self) -> str:
        who = "ATT" if self.to_move == ATT else "DEF"
        return (f"<State ply={self.ply} to_move={who} "
                f"result={self.result}>")


# --- construction ---------------------------------------------------------

START = """
. . . A A A A A . . .
. . . . . A . . . . .
. . . . . . . . . . .
A . . . . D . . . . A
A . . . D D D . . . A
A A . D D K D D . A A
A . . . D D D . . . A
A . . . . D . . . . A
. . . . . . . . . . .
. . . . . A . . . . .
. . . A A A A A . . .
"""

_CHAR_TO_PIECE = {".": EMPTY, "A": ATTACKER, "D": DEFENDER, "K": KING}
_PIECE_TO_CHAR = {v: k for k, v in _CHAR_TO_PIECE.items()}


def state_from_ascii(text: str, to_move: int = ATT, ply: int = 0) -> State:
    """Build a State from an 11x11 ascii diagram ('.', 'A', 'D', 'K')."""
    toks = text.split()
    if len(toks) != N_SQ:
        raise ValueError(f"expected {N_SQ} squares, got {len(toks)}")
    b = bytearray(N_SQ)
    atts, defs, king = [], [], -1
    for sq, t in enumerate(toks):
        p = _CHAR_TO_PIECE[t]
        b[sq] = p
        if p == ATTACKER:
            atts.append(sq)
        elif p == DEFENDER:
            defs.append(sq)
        elif p == KING:
            if king != -1:
                raise ValueError("more than one king")
            king = sq
    if king == -1:
        raise ValueError("no king on the board")
    for sq in atts + defs:
        if IS_RESTRICTED[sq]:
            raise ValueError(f"non-king piece on restricted square {sq}")
    h = 0
    for sq in range(N_SQ):
        h ^= ZP[sq][b[sq]]
    if to_move == DEF:
        h ^= Z_SIDE
    return State(bytes(b), tuple(atts), tuple(defs), king, to_move, ply, h,
                 (h,), None)


def to_ascii(state: State) -> str:
    b = state.board_b
    return "\n".join(
        " ".join(_PIECE_TO_CHAR[b[r * N + c]] for c in range(N))
        for r in range(N))


def initial_state() -> State:
    return state_from_ascii(START, ATT)


# --- actions --------------------------------------------------------------

def make_action(frm: int, to: int) -> int:
    """Action id for moving the piece on `frm` to `to` (must share row/col)."""
    fr, fc = divmod(frm, N)
    tr, tc = divmod(to, N)
    if fr == tr and fc != tc:
        d, dist = (2, fc - tc) if tc < fc else (3, tc - fc)
    elif fc == tc and fr != tr:
        d, dist = (0, fr - tr) if tr < fr else (1, tr - fr)
    else:
        raise ValueError(f"{frm}->{to} is not a rook move")
    return frm * MOVES_PER_SQ + d * MAX_DIST + dist - 1


def decode_action(action: int) -> tuple[int, int]:
    """(from_sq, to_sq) of an action.  Only meaningful for legal actions."""
    frm, m = divmod(action, MOVES_PER_SQ)
    d, i = divmod(m, MAX_DIST)
    return frm, frm + STEP[d] * (i + 1)


def _gen(state: State) -> list:
    """All legal action ids for the side to move (no terminal checks)."""
    b = state.board_b
    acts = []
    ap = acts.append
    for sq in (state.atts if state.to_move == ATT else state.defs):
        rays = RAYS[sq]
        base = sq * MOVES_PER_SQ
        for d in range(N_DIRS):
            a0 = base + d * MAX_DIST
            for i, t in enumerate(rays[d]):
                if b[t]:
                    break
                if not IS_RESTRICTED[t]:
                    ap(a0 + i)
    if state.to_move == DEF:
        sq = state.king
        rays = RAYS[sq]
        base = sq * MOVES_PER_SQ
        for d in range(N_DIRS):
            a0 = base + d * MAX_DIST
            for i, t in enumerate(rays[d]):
                if b[t]:
                    break
                ap(a0 + i)                 # king may land anywhere vacant
    return acts


def legal_actions(state: State) -> np.ndarray:
    """Legal action ids (int64).  Empty for any terminal state."""
    if state.result is not None:
        return np.empty(0, dtype=np.int64)
    if state._acts is None:
        state._acts = _gen(state)
    if state._acts_np is None:
        state._acts_np = np.array(state._acts, dtype=np.int64)
    return state._acts_np


def legal_mask(state: State) -> np.ndarray:
    """Bool vector of length N_ACTIONS marking the legal actions."""
    if state._mask is None:
        m = np.zeros(N_ACTIONS, dtype=bool)
        acts = legal_actions(state)
        if len(acts):
            m[acts] = True
        state._mask = m
    return state._mask


def is_terminal(state: State) -> bool:
    return state.result is not None


# --- applying a move ------------------------------------------------------

def _pseudo_legal(state: State, frm: int, d: int, i: int) -> bool:
    b = state.board_b
    p = b[frm]
    if state.to_move == ATT:
        if p != ATTACKER:
            return False
    elif p != DEFENDER and p != KING:
        return False
    ray = RAYS[frm][d]
    if i >= len(ray):
        return False
    for j in range(i + 1):
        if b[ray[j]]:
            return False
    if p != KING and IS_RESTRICTED[ray[i]]:
        return False
    return True


def _surrounded(b, defs, king) -> bool:
    """True if no defender/king can reach the board edge through
    attacker-free squares (rule 7b encirclement)."""
    seen = bytearray(N_SQ)
    stack = []
    for sq in defs + (king,):
        if IS_EDGE[sq]:
            return False
        seen[sq] = 1
        stack.append(sq)
    while stack:
        for t in NEIGHBORS4[stack.pop()]:
            if not seen[t] and b[t] != ATTACKER:
                if IS_EDGE[t]:
                    return False
                seen[t] = 1
                stack.append(t)
    return True


def apply(state: State, action: int, move_limit: int = MOVE_LIMIT) -> State:
    """Play `action`, returning the successor state (never mutates `state`)."""
    if state.result is not None:
        raise ValueError("game is over")
    if not 0 <= action < N_ACTIONS:
        raise ValueError(f"action {action} out of range")
    frm, m = divmod(action, MOVES_PER_SQ)
    d, i = divmod(m, MAX_DIST)
    if not _pseudo_legal(state, frm, d, i):
        raise ValueError(f"illegal action {action}")
    to = frm + STEP[d] * (i + 1)

    b = bytearray(state.board_b)
    piece = b[frm]
    b[frm] = EMPTY
    b[to] = piece
    h = state.hash ^ ZP[frm][piece] ^ ZP[to][piece] ^ Z_SIDE
    mover = state.to_move
    atts, defs, king = state.atts, state.defs, state.king
    if piece == ATTACKER:
        atts = tuple(to if s == frm else s for s in atts)
    elif piece == DEFENDER:
        defs = tuple(to if s == frm else s for s in defs)
    else:
        king = to

    # captures closed by the moved piece (one victim per direction)
    captured = False
    if mover == ATT:
        removed = None
        for v, w in ADJ2[to]:
            if b[v] == DEFENDER:
                x = b[w]
                if (x == ATTACKER or IS_CORNER[w]
                        or (w == THRONE and x == EMPTY)):
                    b[v] = EMPTY
                    h ^= ZP[v][DEFENDER]
                    removed = (v,) if removed is None else removed + (v,)
        if removed:
            captured = True
            defs = tuple(s for s in defs if s not in removed)
    else:
        removed = None
        for v, w in ADJ2[to]:
            if b[v] == ATTACKER:
                x = b[w]
                if (x == DEFENDER or x == KING or IS_CORNER[w]
                        or w == THRONE):
                    b[v] = EMPTY
                    h ^= ZP[v][ATTACKER]
                    removed = (v,) if removed is None else removed + (v,)
        if removed:
            captured = True
            atts = tuple(s for s in atts if s not in removed)

    # wins decided by the mover
    result = None
    if piece == KING and IS_CORNER[to]:
        result = _RESULT_ESCAPE
    elif mover == ATT:
        kn = NEIGHBORS4[king]
        if (to in kn and len(kn) == 4
                and all(b[q] == ATTACKER or q == THRONE for q in kn)):
            result = _RESULT_KING_CAPTURED
        elif ((captured or any(b[q] == ATTACKER for q in NEIGHBORS8[to]))
                and _surrounded(b, defs, king)):
            result = _RESULT_ENCIRCLED

    ply = state.ply + 1
    hist = (h,) if captured else state.hist + (h,)
    s = State(bytes(b), atts, defs, king, DEF if mover == ATT else ATT, ply,
              h, hist, result)
    if result is None:
        acts = _gen(s)
        if not acts:
            s.result = Result(mover, "no moves")
        elif hist.count(h) >= 3:
            s.result = _RESULT_REPETITION
        elif ply >= move_limit:
            s.result = _RESULT_MOVE_LIMIT
        else:
            s._acts = acts
    return s


# --- network encoding -----------------------------------------------------

def encode(state: State) -> np.ndarray:
    """(5, 11, 11) float32 planes: attackers, defenders, king, side-to-move
    (all 1s when the attacker is to move), normalized move count."""
    g = state.board.reshape(N, N)
    planes = np.zeros((5, N, N), dtype=np.float32)
    planes[0] = g == ATTACKER
    planes[1] = g == DEFENDER
    planes[2] = g == KING
    planes[3] = 1.0 if state.to_move == ATT else 0.0
    planes[4] = state.ply / MOVE_LIMIT
    return planes
