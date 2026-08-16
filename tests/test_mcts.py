"""Phase 2 tests: network heads, MCTS search behaviour, self-play sample
integrity, training-step learning signal, and the paired arena.

MCTS tests use a uniform-prior/zero-value evaluator so they exercise pure
search behaviour with no network in the loop; the tactical positions have
exactly one winning move, so `argmax(visits)` failures point straight at
selection or backup bugs.  Torch-based tests run tiny nets on CPU.
"""

import numpy as np
import pytest
import torch

from tafl.engine import (
    ATT, DEF, apply, encode, initial_state, legal_actions, make_action,
    state_from_ascii,
)
from tafl.eval import ArenaConfig, elo_diff, play_match
from tafl.mcts import MCTS, terminal_value
from tafl.net import NetEvaluator, PolicyValueNet
from tafl.selfplay import SelfPlayConfig, play_game, sample_planes
from tafl.train import ReplayBuffer, TrainConfig, Trainer

torch.manual_seed(0)


def sq(r, c):
    return r * 11 + c


def board(pieces, to_move=ATT):
    g = [["." for _ in range(11)] for _ in range(11)]
    for (r, c), ch in pieces.items():
        g[r][c] = ch
    return state_from_ascii("\n".join(" ".join(row) for row in g), to_move)


class UniformEval:
    """Uniform priors, zero value: search signal comes from terminals only."""

    def evaluate(self, states):
        out = []
        for s in states:
            acts = legal_actions(s)
            p = np.full(len(acts), 1.0 / len(acts), dtype=np.float32)
            out.append((acts, p, 0.0))
        return out


def tiny_evaluator(seed=0):
    torch.manual_seed(seed)
    net = PolicyValueNet(channels=16, blocks=1)
    return NetEvaluator(net, torch.device("cpu"))


# --- network ---------------------------------------------------------------

def test_net_shapes_and_value_range():
    net = PolicyValueNet(channels=16, blocks=2)
    x = torch.randn(3, 5, 11, 11)
    logits, v = net(x)
    assert logits.shape == (3, 4840)
    assert v.shape == (3,)
    assert torch.all(v > -1) and torch.all(v < 1)


def test_policy_head_matches_action_encoding():
    """Channel m at square sq must be the logit for action sq*40 + m."""
    net = PolicyValueNet(channels=8, blocks=1)
    net.eval()
    with torch.no_grad():
        net.p_out.weight.zero_()
        net.p_out.bias.copy_(torch.arange(40, dtype=torch.float32))
        logits, _ = net(torch.zeros(1, 5, 11, 11))
    expect = torch.arange(40, dtype=torch.float32).repeat(121)
    assert torch.equal(logits[0], expect)


def test_evaluator_sparse_priors():
    st = initial_state()
    (acts, p, v), = tiny_evaluator().evaluate([st])
    assert np.array_equal(acts, legal_actions(st))
    assert p.shape == acts.shape
    assert abs(float(p.sum()) - 1.0) < 1e-5
    assert -1.0 <= v <= 1.0


# --- MCTS ------------------------------------------------------------------

def escape_in_one():
    """DEF to move; the only winning move is king (2,0) -> corner (0,0)."""
    return board({(2, 0): "K", (3, 0): "A", (9, 9): "A"}, DEF)


def test_terminal_value_perspective():
    st = escape_in_one()
    won = apply(st, make_action(sq(2, 0), sq(0, 0)))
    assert won.result.winner == DEF
    assert won.to_move == ATT
    assert terminal_value(won) == -1.0    # the side to move has lost


def test_mcts_finds_escape_in_one():
    m = MCTS(UniformEval(), rng=np.random.default_rng(0))
    m.set_root(escape_in_one())
    root = m.run(200)
    acts, probs = m.root_policy(0.0)
    assert int(acts[np.argmax(probs)]) == make_action(sq(2, 0), sq(0, 0))
    assert m.root_value() > 0.5           # root knows it is winning
    assert root.N.sum() == 200            # every simulation is accounted for


def test_mcts_finds_king_capture_in_one():
    # King on the throne with three attackers around it; a4 (0,6) slides
    # down to (5,6) closing the fourth side.
    st = board({(5, 5): "K", (4, 5): "A", (6, 5): "A", (5, 4): "A",
                (0, 6): "A", (9, 2): "D"}, ATT)
    win = make_action(sq(0, 6), sq(5, 6))
    assert apply(st, win).result.winner == ATT
    m = MCTS(UniformEval(), rng=np.random.default_rng(0))
    m.set_root(st)
    m.run(300)
    acts, probs = m.root_policy(0.0)
    assert int(acts[np.argmax(probs)]) == win


def test_mcts_batched_matches_serial():
    """Leaf batching (virtual loss) must not change the found tactic."""
    for bs in (1, 16):
        m = MCTS(UniformEval(), batch_size=bs, rng=np.random.default_rng(1))
        m.set_root(escape_in_one())
        root = m.run(150)
        acts, probs = m.root_policy(0.0)
        assert int(acts[np.argmax(probs)]) == make_action(sq(2, 0), sq(0, 0))
        assert root.N.sum() == 150


def test_root_policy_temperature():
    m = MCTS(UniformEval(), rng=np.random.default_rng(0))
    m.set_root(escape_in_one())
    m.run(100)
    acts, p1 = m.root_policy(1.0)
    assert abs(p1.sum() - 1.0) < 1e-9
    np.testing.assert_allclose(p1, m.root.N / m.root.N.sum(), atol=1e-9)
    _, p0 = m.root_policy(0.0)
    assert np.count_nonzero(p0) == 1 and p0.max() == 1.0


def test_advance_reuses_subtree():
    m = MCTS(UniformEval(), rng=np.random.default_rng(0))
    st = initial_state()
    m.set_root(st)
    m.run(50)
    acts, probs = m.root_policy(0.0)
    a = int(acts[np.argmax(probs)])
    m.advance(a)
    assert m.root.state.board_b == apply(st, a).board_b
    m.run(50)                             # search continues from the new root
    assert m.root.N.sum() >= 50


def test_dirichlet_noise_perturbs_root_priors():
    m1 = MCTS(UniformEval(), rng=np.random.default_rng(0))
    m1.set_root(initial_state())
    m1.run(1, noise=False)
    m2 = MCTS(UniformEval(), rng=np.random.default_rng(0))
    m2.set_root(initial_state())
    m2.run(1, noise=True)
    assert not np.allclose(m1.root.P, m2.root.P)
    assert abs(m2.root.P.sum() - 1.0) < 1e-4


# --- self-play -------------------------------------------------------------

@pytest.fixture(scope="module")
def one_game():
    cfg = SelfPlayConfig(sims=12, batch_size=4, temp_plies=6, move_limit=40)
    return play_game(tiny_evaluator(), cfg, np.random.default_rng(0)), cfg


def test_selfplay_samples_wellformed(one_game):
    rec, cfg = one_game
    assert rec.result is not None and rec.plies == len(rec.samples)
    assert rec.samples[0].board_b == initial_state().board_b
    w = rec.result.winner
    for s in rec.samples:
        assert abs(float(s.probs.sum()) - 1.0) < 1e-6
        assert len(s.acts) == len(s.probs)
        if w is None:
            assert s.z == -cfg.draw_penalty
        else:
            assert s.z == (1.0 if s.to_move == w else -1.0)


def test_sample_planes_matches_engine_encode(one_game):
    rec, _ = one_game
    np.testing.assert_array_equal(sample_planes(rec.samples[0]),
                                  encode(initial_state()))


# --- training --------------------------------------------------------------

def test_training_reduces_loss(one_game):
    rec, _ = one_game
    torch.manual_seed(0)
    cfg = TrainConfig(channels=16, blocks=1, batch_size=32, lr=3e-3)
    trainer = Trainer(cfg, device=torch.device("cpu"))
    buffer = ReplayBuffer(10_000)
    buffer.add_game(rec)
    before = trainer.train_steps(buffer, 2)
    trainer.train_steps(buffer, 60)
    after = trainer.train_steps(buffer, 2)
    assert after["policy_loss"] < before["policy_loss"]
    assert after["value_loss"] < before["value_loss"]


# --- arena -----------------------------------------------------------------

def test_arena_paired_match():
    cfg = ArenaConfig(sims=8, batch_size=4, temp_plies=4, move_limit=30)
    res = play_match(tiny_evaluator(0), tiny_evaluator(1), pairs=1, cfg=cfg,
                     seed=0)
    assert res.games == 2
    assert len(res.a_as_att) == 1 and len(res.a_as_def) == 1
    assert 0.0 <= res.score_a <= 2.0
    assert all(l > 0 for l in res.lengths)


def test_elo_diff():
    assert elo_diff(0.5) == 0.0
    assert elo_diff(0.75) > 100
    assert elo_diff(0.25) == -elo_diff(0.75)
    assert elo_diff(1.0) == 800.0
