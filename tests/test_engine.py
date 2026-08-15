"""Rule and invariant tests for the Fetlar engine.

Positions are built with `board({(row, col): 'A'|'D'|'K'})`; rows count from
the top, columns from the left.  Every test constructs the exact squares it
talks about, so failures point straight at the broken rule.
"""

import numpy as np
import pytest

from tafl import engine as E
from tafl.engine import (
    ATT, DEF, EMPTY, ATTACKER, DEFENDER, KING,
    MOVE_LIMIT, N_ACTIONS, Result,
    apply, decode_action, encode, initial_state, legal_actions, legal_mask,
    make_action, state_from_ascii,
)


def sq(r, c):
    return r * 11 + c


def board(pieces, to_move=ATT):
    g = [["." for _ in range(11)] for _ in range(11)]
    for (r, c), ch in pieces.items():
        g[r][c] = ch
    return state_from_ascii("\n".join(" ".join(row) for row in g), to_move)


def mv(st, frm, to):
    return apply(st, make_action(sq(*frm), sq(*to)))


def dests(st, frm):
    """Set of (row, col) destinations for the piece on `frm`."""
    out = set()
    for a in legal_actions(st):
        f, t = decode_action(int(a))
        if f == sq(*frm):
            out.add(divmod(t, 11))
    return out


def piece_counts(st):
    b = st.board
    return (int((b == ATTACKER).sum()), int((b == DEFENDER).sum()),
            int((b == KING).sum()))


# --- setup and movement ---------------------------------------------------

def test_initial_position():
    st = initial_state()
    assert piece_counts(st) == (24, 12, 1)
    assert st.board[sq(5, 5)] == KING
    assert st.to_move == ATT
    assert st.result is None
    # known opening mobility for the Fetlar/Copenhagen 11x11 setup
    assert len(legal_actions(st)) == 116
    assert len(legal_actions(state_from_ascii(E.START, DEF))) == 60


def test_rook_movement_and_blocking():
    st = initial_state()
    # attacker c11 (row 0, col 3): down until the d6 defender, left to the
    # corner exclusive; right is blocked by its neighbour.
    assert dests(st, (0, 3)) == {(1, 3), (2, 3), (3, 3), (4, 3),
                                 (0, 2), (0, 1)}


def test_non_king_passes_through_empty_throne_but_cannot_land():
    st = board({(5, 2): "D", (9, 9): "K", (0, 5): "A"}, DEF)
    d = dests(st, (5, 2))
    assert (5, 5) not in d                       # cannot land on the throne
    for c in (3, 4, 6, 7, 8, 9, 10):             # but passes through it
        assert (5, c) in d


def test_occupied_throne_blocks_movement():
    st = board({(5, 2): "D", (5, 5): "K", (0, 5): "A"}, DEF)
    d = dests(st, (5, 2))
    assert (5, 3) in d and (5, 4) in d
    assert (5, 5) not in d and (5, 6) not in d


def test_king_may_reenter_throne():
    st = board({(5, 7): "K", (0, 5): "A", (9, 1): "D"}, DEF)
    assert (5, 5) in dests(st, (5, 7))
    st2 = mv(st, (5, 7), (5, 5))
    assert st2.board[sq(5, 5)] == KING
    assert st2.result is None


def test_non_king_cannot_land_on_corner():
    st = board({(0, 3): "A", (9, 9): "K", (8, 1): "D"}, ATT)
    d = dests(st, (0, 3))
    assert (0, 1) in d and (0, 2) in d and (0, 0) not in d


def test_king_escape_to_corner_wins():
    st = board({(0, 5): "K", (9, 9): "A", (8, 1): "D"}, DEF)
    assert (0, 0) in dests(st, (0, 5))
    st2 = mv(st, (0, 5), (0, 0))
    assert st2.result == Result(DEF, "escape")


# --- captures -------------------------------------------------------------

def test_basic_capture():
    st = board({(4, 0): "A", (4, 5): "A", (4, 4): "D",
                (9, 9): "K", (8, 8): "D"}, ATT)
    st2 = mv(st, (4, 0), (4, 3))
    assert st2.board[sq(4, 4)] == EMPTY
    assert piece_counts(st2) == (2, 1, 1)


def test_moving_into_sandwich_is_safe_and_captures_need_the_aggressor():
    st = board({(3, 3): "A", (3, 5): "A", (7, 4): "D",
                (9, 0): "A", (9, 9): "K"}, DEF)
    st2 = mv(st, (7, 4), (3, 4))                 # steps between two attackers
    assert st2.board[sq(3, 4)] == DEFENDER
    st3 = mv(st2, (9, 0), (8, 0))                # unrelated attacker move
    assert st3.board[sq(3, 4)] == DEFENDER       # still not captured


def test_triple_capture_through_throne():
    st = board({(5, 7): "A", (5, 2): "A", (3, 4): "A", (7, 4): "A",
                (5, 3): "D", (4, 4): "D", (6, 4): "D", (9, 9): "K"}, ATT)
    st2 = mv(st, (5, 7), (5, 4))                 # passes over the empty throne
    assert piece_counts(st2) == (4, 0, 1)
    assert st2.result is None


def test_capture_against_corner():
    st = board({(0, 1): "D", (3, 2): "A", (5, 0): "A", (9, 9): "K"}, ATT)
    st2 = mv(st, (3, 2), (0, 2))
    assert st2.board[sq(0, 1)] == EMPTY

    st = board({(10, 9): "A", (7, 8): "D", (0, 5): "A", (2, 2): "K"}, DEF)
    st2 = mv(st, (7, 8), (10, 8))
    assert st2.board[sq(10, 9)] == EMPTY


def test_defender_captured_against_empty_throne():
    st = board({(8, 7): "A", (5, 6): "D", (9, 9): "K", (2, 0): "D"}, ATT)
    st2 = mv(st, (8, 7), (5, 7))
    assert st2.board[sq(5, 6)] == EMPTY


def test_defender_safe_against_king_occupied_throne():
    st = board({(8, 7): "A", (5, 6): "D", (5, 5): "K", (2, 0): "D"}, ATT)
    st2 = mv(st, (8, 7), (5, 7))
    assert st2.board[sq(5, 6)] == DEFENDER
    assert st2.result is None


def test_attacker_captured_against_throne_empty_and_occupied():
    st = board({(5, 6): "A", (2, 7): "D", (9, 9): "K", (0, 4): "A"}, DEF)
    st2 = mv(st, (2, 7), (5, 7))
    assert st2.board[sq(5, 6)] == EMPTY          # empty throne is hostile

    st = board({(5, 6): "A", (2, 7): "D", (5, 5): "K", (0, 4): "A"}, DEF)
    st2 = mv(st, (2, 7), (5, 7))
    assert st2.board[sq(5, 6)] == EMPTY          # throne always hostile to ATT


def test_king_takes_part_in_captures():
    # king closes the sandwich himself
    st = board({(6, 6): "A", (7, 6): "D", (5, 9): "K", (10, 4): "A"}, DEF)
    st2 = mv(st, (5, 9), (5, 6))
    assert st2.board[sq(6, 6)] == EMPTY
    # king as the far support piece
    st = board({(6, 6): "A", (7, 6): "K", (5, 2): "D", (10, 4): "A"}, DEF)
    st2 = mv(st, (5, 2), (5, 6))
    assert st2.board[sq(6, 6)] == EMPTY


def test_edge_capture_is_normal_but_edge_is_not_hostile():
    # sandwich along the edge works
    st = board({(0, 4): "D", (0, 3): "A", (4, 5): "A", (9, 9): "K"}, ATT)
    st2 = mv(st, (4, 5), (0, 5))
    assert st2.board[sq(0, 4)] == EMPTY
    # a lone piece pressed against the edge is NOT captured
    st = board({(0, 4): "D", (4, 4): "A", (0, 9): "A", (9, 9): "K"}, ATT)
    st2 = mv(st, (4, 4), (1, 4))
    assert st2.board[sq(0, 4)] == DEFENDER


def test_no_shieldwall_capture_in_fetlar():
    st = board({(0, 4): "D", (0, 5): "D", (1, 4): "A", (1, 5): "A",
                (0, 3): "A", (4, 6): "A", (9, 9): "K"}, ATT)
    st2 = mv(st, (4, 6), (0, 6))                 # brackets the edge row
    assert st2.board[sq(0, 4)] == DEFENDER
    assert st2.board[sq(0, 5)] == DEFENDER


# --- king capture ---------------------------------------------------------

def test_king_not_captured_by_two_attackers():
    st = board({(6, 6): "K", (6, 7): "A", (6, 2): "A", (3, 3): "D"}, ATT)
    st2 = mv(st, (6, 2), (6, 5))
    assert st2.result is None
    assert st2.board[sq(6, 6)] == KING


def test_king_captured_on_four_sides():
    st = board({(7, 5): "K", (6, 5): "A", (8, 5): "A", (7, 4): "A",
                (7, 9): "A", (2, 2): "D"}, ATT)
    st2 = mv(st, (7, 9), (7, 6))
    assert st2.result == Result(ATT, "king captured")


def test_king_captured_on_throne():
    st = board({(5, 5): "K", (4, 5): "A", (6, 5): "A", (5, 4): "A",
                (5, 8): "A", (2, 2): "D"}, ATT)
    st2 = mv(st, (5, 8), (5, 6))
    assert st2.result == Result(ATT, "king captured")


def test_king_next_to_throne_needs_three_attackers():
    st = board({(4, 5): "K", (3, 5): "A", (4, 4): "A", (4, 9): "A",
                (2, 2): "D"}, ATT)
    st2 = mv(st, (4, 9), (4, 6))                 # 3 attackers + throne
    assert st2.result == Result(ATT, "king captured")

    st = board({(4, 5): "K", (3, 5): "A", (4, 9): "A", (2, 2): "D"}, ATT)
    st2 = mv(st, (4, 9), (4, 6))                 # only 2 attackers + throne
    assert st2.result is None


def test_king_cannot_be_captured_on_the_edge():
    st = board({(0, 5): "K", (0, 4): "A", (0, 6): "A", (4, 5): "A",
                (9, 1): "D"}, ATT)
    st2 = mv(st, (4, 5), (1, 5))                 # boxes the king in
    assert st2.result is None                    # not a capture...


def test_boxed_king_on_edge_loses_by_no_moves():
    # ...but if the king is the last white piece, white cannot move and
    # loses by rule 8, exactly as rule 7a's exception describes.
    st = board({(0, 5): "K", (0, 4): "A", (0, 6): "A", (4, 5): "A"}, ATT)
    st2 = mv(st, (4, 5), (1, 5))
    assert st2.result == Result(ATT, "no moves")


# --- encirclement ---------------------------------------------------------

RING = {(3, 5): "A", (3, 6): "A", (4, 4): "A", (4, 7): "A",
        (5, 4): "A", (5, 7): "A", (6, 5): "A", (6, 6): "A"}


def _ring_position(extra=None, drop=None):
    pieces = dict(RING)
    del pieces[(6, 6)]
    pieces[(9, 6)] = "A"                         # closing piece, ready to move
    pieces[(4, 5)] = "K"
    pieces[(4, 6)] = "D"
    if drop:
        del pieces[drop]
    if extra:
        pieces.update(extra)
    return board(pieces, ATT)


def test_encirclement_win():
    st2 = mv(_ring_position(), (9, 6), (6, 6))
    assert st2.result == Result(ATT, "encirclement")


def test_broken_ring_is_not_a_win():
    st2 = mv(_ring_position(drop=(5, 4)), (9, 6), (6, 6))
    assert st2.result is None


def test_defender_outside_ring_prevents_the_win():
    st2 = mv(_ring_position(extra={(9, 9): "D"}), (9, 6), (6, 6))
    assert st2.result is None


# --- no moves, repetition, move limit ------------------------------------

def test_attacker_with_no_moves_loses():
    # lone attacker walled in by two defenders and the corner
    st = board({(0, 1): "A", (0, 2): "D", (1, 1): "D",
                (9, 9): "K", (9, 5): "D"}, DEF)
    st2 = mv(st, (9, 5), (9, 4))
    assert st2.result == Result(DEF, "no moves")


def test_threefold_repetition_draw():
    st = board({(9, 1): "D", (1, 8): "A", (6, 9): "K"}, DEF)
    cycle = [((9, 1), (9, 2)), ((1, 8), (1, 9)),
             ((9, 2), (9, 1)), ((1, 9), (1, 8))]
    for frm, to in cycle:                        # position occurs twice
        st = mv(st, frm, to)
        assert st.result is None
    for frm, to in cycle[:-1]:
        st = mv(st, frm, to)
        assert st.result is None
    st = mv(st, *cycle[-1])                      # third occurrence
    assert st.result == Result(None, "repetition")


def test_move_limit_draw():
    st = board({(9, 1): "D", (1, 8): "A", (6, 9): "K"}, DEF)
    st = apply(st, make_action(sq(9, 1), sq(9, 2)), move_limit=2)
    assert st.result is None
    st = apply(st, make_action(sq(1, 8), sq(1, 9)), move_limit=2)
    assert st.result == Result(None, "move limit")


# --- API, encoding, immutability -----------------------------------------

def test_illegal_actions_raise():
    st = initial_state()
    with pytest.raises(ValueError):              # not the mover's piece
        apply(st, make_action(sq(5, 4), sq(4, 4)))
    with pytest.raises(ValueError):              # destination occupied
        apply(st, make_action(sq(0, 3), sq(0, 4)))
    with pytest.raises(ValueError):              # path blocked
        apply(st, make_action(sq(0, 3), sq(0, 8)))
    with pytest.raises(ValueError):              # out of range
        apply(st, N_ACTIONS)
    d = board({(5, 2): "D", (9, 9): "K", (0, 5): "A"}, DEF)
    with pytest.raises(ValueError):              # non-king onto the throne
        apply(d, make_action(sq(5, 2), sq(5, 5)))
    done = mv(board({(0, 5): "K", (9, 9): "A", (8, 1): "D"}, DEF),
              (0, 5), (0, 0))
    with pytest.raises(ValueError):              # game already over
        apply(done, make_action(sq(9, 9), sq(9, 8)))


def test_action_encoding_roundtrip():
    st = initial_state()
    for a in legal_actions(st):
        frm, to = decode_action(int(a))
        assert make_action(frm, to) == int(a)


def test_apply_does_not_mutate():
    st = initial_state()
    snapshot = (st.board_b, st.hash, st.hist, st.to_move, st.ply)
    board_copy = st.board.copy()
    st2 = apply(st, int(legal_actions(st)[0]))
    assert st2 is not st
    assert (st.board_b, st.hash, st.hist, st.to_move, st.ply) == snapshot
    assert np.array_equal(st.board, board_copy)


def test_encode():
    st = initial_state()
    planes = encode(st)
    assert planes.shape == (5, 11, 11) and planes.dtype == np.float32
    assert planes[0].sum() == 24
    assert planes[1].sum() == 12
    assert planes[2].sum() == 1 and planes[2, 5, 5] == 1
    assert (planes[3] == 1).all()                # attacker to move
    assert (planes[4] == 0).all()
    st2 = apply(st, int(legal_actions(st)[0]))
    p2 = encode(st2)
    assert (p2[3] == 0).all()
    assert np.allclose(p2[4], 1 / MOVE_LIMIT)


# --- random playouts vs a reference implementation ------------------------

def ref_moves(state):
    """Independent, obviously-correct move generator: set of (frm, to)."""
    g = state.board.reshape(11, 11)
    restricted = {(0, 0), (0, 10), (10, 0), (10, 10), (5, 5)}
    out = set()
    for r in range(11):
        for c in range(11):
            p = int(g[r, c])
            if p == EMPTY:
                continue
            if state.to_move == ATT and p != ATTACKER:
                continue
            if state.to_move == DEF and p == ATTACKER:
                continue
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                while 0 <= rr < 11 and 0 <= cc < 11 and g[rr, cc] == EMPTY:
                    if p == KING or (rr, cc) not in restricted:
                        out.add((r * 11 + c, rr * 11 + cc))
                    rr, cc = rr + dr, cc + dc
    return out


def full_hash(state):
    h = 0
    for s, p in enumerate(state.board.tolist()):
        if p:
            h ^= E.ZP[s][p]
    if state.to_move == DEF:
        h ^= E.Z_SIDE
    return h


@pytest.mark.parametrize("seed", range(5))
def test_random_playout_invariants(seed):
    rng = np.random.default_rng(seed)
    st = initial_state()
    prev = piece_counts(st)
    plies = 0
    while st.result is None:
        acts = legal_actions(st)
        assert len(acts) > 0
        assert {decode_action(int(a)) for a in acts} == ref_moves(st)
        m = legal_mask(st)
        assert m.shape == (N_ACTIONS,) and m.sum() == len(acts)
        assert m[acts].all()
        assert full_hash(st) == st.hash
        assert sorted(st.atts) == [s for s in range(121)
                                   if st.board_b[s] == ATTACKER]
        assert sorted(st.defs) == [s for s in range(121)
                                   if st.board_b[s] == DEFENDER]
        assert st.board_b[st.king] == KING
        st = apply(st, int(acts[rng.integers(len(acts))]))
        counts = piece_counts(st)
        assert all(a <= b for a, b in zip(counts, prev))
        prev = counts
        plies += 1
        assert plies <= MOVE_LIMIT
    assert st.result.reason in {"escape", "king captured", "encirclement",
                                "no moves", "repetition", "move limit"}
    assert len(legal_actions(st)) == 0
