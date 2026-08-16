"""Phase 3 tests: agents, the checkpoint registry, level persistence and
the within-side calibration pipeline.

The heuristic tests use hand-built positions with one obviously best move,
so a failure points at the evaluation rather than at search noise.  The
calibration tests run the real pipeline on throwaway 8-channel
checkpoints -- the *numbers* are meaningless there (every checkpoint holds
the same weights), but the schedule, tallying, ladder selection and
levels.json round-trip are exercised end to end.
"""

import json

import numpy as np
import pytest
import torch

from tafl.agents import (ATT, DEF, HeuristicAgent, Levels, MCTSAgent,
                         PolicyAgent, RandomAgent, build_levels,
                         clear_evaluator_cache, default_levels, describe_spec,
                         heuristic_value, king_corner_distance,
                         king_escape_threats, list_checkpoints, load_levels,
                         make_agent, side_id, side_name, spec)
from tafl.calibrate import (CalibrationConfig, build_jobs, calibrate,
                            candidate_pool, format_table, margins,
                            panel_specs, select_ladder, spread_checkpoints,
                            tally)
from tafl.engine import (apply, initial_state, legal_actions, make_action,
                         state_from_ascii)
from tafl.eval import (AgentMatchConfig, Job, play_agent_game, run_jobs,
                       score_for)
from tafl.net import PolicyValueNet, save_checkpoint


def sq(r, c):
    return r * 11 + c


def board(pieces, to_move=ATT):
    g = [["." for _ in range(11)] for _ in range(11)]
    for (r, c), ch in pieces.items():
        g[r][c] = ch
    return state_from_ascii("\n".join(" ".join(row) for row in g), to_move)


def rng(seed=0):
    return np.random.default_rng(seed)


@pytest.fixture
def models(tmp_path):
    """A models dir with two throwaway checkpoints plus best.pt."""
    torch.manual_seed(0)
    net = PolicyValueNet(channels=8, blocks=1)
    d = tmp_path / "models"
    d.mkdir()
    for it in (2, 4):
        save_checkpoint(net, d / f"ckpt_{it:04d}.pt", iter=it, games=it * 10)
        (d / f"ckpt_{it:04d}.json").write_text(json.dumps(
            {"iter": it, "total_games": it * 10, "gate_elo": 12.5 * it,
             "saved_at": 1.0}))
    save_checkpoint(net, d / "best.pt", iter=4, games=40)
    (d / "best.json").write_text(json.dumps(
        {"iter": 4, "total_games": 40, "gate_elo": 50.0, "saved_at": 2.0}))
    clear_evaluator_cache()
    yield d
    clear_evaluator_cache()


# --- sides ------------------------------------------------------------------

def test_side_id_and_name():
    assert side_id("attacker") == ATT and side_id("DEF") == DEF
    assert side_id(ATT) == ATT
    assert side_name(DEF) == "defender"
    with pytest.raises(ValueError):
        side_id("sideways")


# --- heuristic evaluation ---------------------------------------------------

def test_start_position_is_materially_even():
    """24 attackers vs 12 defenders is a balanced *start*, so the material
    term must be zero there (defenders weighted double)."""
    st = initial_state()
    assert king_corner_distance(st) == 10
    assert king_escape_threats(st) == 0
    assert heuristic_value(st) == pytest.approx(0.35 * 10)


def test_escape_threats_counted_per_open_corner():
    st = board({(0, 5): "K"}, DEF)          # top row, both corners open
    assert king_escape_threats(st) == 2
    blocked = board({(0, 5): "K", (0, 2): "A"}, DEF)
    assert king_escape_threats(blocked) == 1


def test_terminal_values_dominate():
    st = board({(2, 0): "K", (3, 0): "A"}, DEF)
    won = apply(st, make_action(sq(2, 0), sq(0, 0)))
    assert won.result.winner == DEF
    assert heuristic_value(won) < -1000


def test_heuristic_defender_escapes():
    st = board({(2, 0): "K", (3, 0): "A", (9, 9): "A"}, DEF)
    a = HeuristicAgent(rng=rng())
    assert a.select_action(st) == make_action(sq(2, 0), sq(0, 0))


def test_heuristic_attacker_captures_the_king():
    st = board({(5, 5): "K", (4, 5): "A", (6, 5): "A", (5, 4): "A",
                (0, 6): "A", (9, 2): "D"}, ATT)
    a = HeuristicAgent(rng=rng())
    assert a.select_action(st) == make_action(sq(0, 6), sq(5, 6))


def test_heuristic_attacker_takes_material():
    """Only one move captures anything; nothing else changes the eval."""
    st = board({(3, 0): "A", (3, 5): "D", (3, 6): "A", (8, 8): "K"}, ATT)
    a = HeuristicAgent(rng=rng())
    chosen = a.select_action(st)
    assert chosen == make_action(sq(3, 0), sq(3, 4))
    assert len(apply(st, chosen).defs) == 0


def test_heuristic_is_deterministic_per_seed():
    st = initial_state()
    picks = {HeuristicAgent(rng=rng(7)).select_action(st) for _ in range(3)}
    assert len(picks) == 1


# --- randomness dial --------------------------------------------------------

def test_randomness_dial_replaces_choices_with_random_ones():
    st = board({(2, 0): "K", (3, 0): "A", (9, 9): "A"}, DEF)
    escape = make_action(sq(2, 0), sq(0, 0))
    sure = HeuristicAgent(rng=rng(1))
    assert all(sure.select_action(st) == escape for _ in range(5))
    noisy = HeuristicAgent(randomness=1.0, rng=rng(1))
    picks = {noisy.select_action(st) for _ in range(20)}
    assert len(picks) > 1
    assert picks <= set(int(a) for a in legal_actions(st))


def test_random_agent_plays_legal_moves():
    st = initial_state()
    a = RandomAgent(rng=rng())
    legal = set(int(x) for x in legal_actions(st))
    assert all(a.select_action(st) in legal for _ in range(10))


# --- net agents -------------------------------------------------------------

def tiny_net_agents(models, **kw):
    ev_spec = spec("policy", "best.pt", **kw)
    return make_agent(ev_spec, models, rng=rng())


def test_policy_agent_greedy_is_deterministic(models):
    st = initial_state()
    a = tiny_net_agents(models)
    picks = {a.select_action(st) for _ in range(5)}
    assert len(picks) == 1
    assert picks.pop() in set(int(x) for x in legal_actions(st))


def test_policy_agent_temperature_varies(models):
    st = initial_state()
    a = make_agent(spec("policy", "best.pt", temperature=1.0), models,
                   rng=rng(3))
    assert len({a.select_action(st) for _ in range(20)}) > 1


def test_mcts_agent_reuses_its_tree(models):
    st = initial_state()
    a = make_agent(spec("mcts", "best.pt", sims=8), models, rng=rng())
    a.new_game()
    action = a.select_action(st)
    st2 = apply(st, action)
    a.observe(action)
    assert a.mcts.root is not None
    assert a.mcts.root.state.hash == st2.hash      # subtree kept
    reply = a.select_action(st2)
    assert reply in set(int(x) for x in legal_actions(st2))


def test_mcts_agent_recovers_from_an_unknown_move(models):
    """A move played outside the agent's tree must reset the root, not
    silently search from a stale position."""
    st = initial_state()
    a = make_agent(spec("mcts", "best.pt", sims=8), models, rng=rng())
    a.new_game()
    a.select_action(st)
    a.mcts.root = None
    a.observe(int(legal_actions(st)[0]))
    assert a.mcts.root is None
    assert a.select_action(st) in set(int(x) for x in legal_actions(st))


def test_evaluator_cache_is_shared_between_agents(models):
    a = make_agent(spec("mcts", "best.pt", sims=4), models, rng=rng())
    b = make_agent(spec("policy", "best.pt"), models, rng=rng())
    assert a.mcts.evaluator is b.evaluator


def test_mcts_agent_rejects_zero_sims(models):
    """sims=0 would leave the root visitless and root_policy degenerate
    (argmax over zeros = always the first legal move)."""
    with pytest.raises(ValueError, match="sims"):
        make_agent(spec("mcts", "best.pt", sims=0), models)


def test_make_agent_dispatch(models):
    assert isinstance(make_agent(spec("random"), models), RandomAgent)
    assert isinstance(make_agent(spec("heuristic"), models), HeuristicAgent)
    assert isinstance(make_agent(spec("policy", "best.pt"), models),
                      PolicyAgent)
    assert isinstance(make_agent(spec("mcts", "best.pt", sims=2), models),
                      MCTSAgent)
    # agent type inferred from the sims budget when it is left out
    assert isinstance(make_agent({"checkpoint": "best.pt", "sims": 2}, models),
                      MCTSAgent)
    assert isinstance(make_agent({"checkpoint": "best.pt"}, models),
                      PolicyAgent)
    with pytest.raises(ValueError):
        make_agent(spec("mcts"), models)
    with pytest.raises(ValueError):
        make_agent({"agent": "telepathy"}, models)


def test_describe_spec_is_readable():
    s = spec("mcts", "ckpt_0006.pt", sims=200)
    assert s["label"] == "ckpt_0006/200sims"
    assert describe_spec(spec("policy", "best.pt", randomness=0.5,
                              temperature=1.0)) == "best/policy/T1/rnd0.5"


# --- checkpoint registry ----------------------------------------------------

def test_list_checkpoints_orders_weakest_first(models):
    ck = list_checkpoints(models)
    assert [c.name for c in ck] == ["ckpt_0002.pt", "ckpt_0004.pt", "best.pt"]
    assert ck[0].iter == 2 and ck[0].games == 20 and ck[0].elo == 25.0
    assert ck[-1].name == "best.pt" and ck[-1].elo == 50.0


def test_checkpoint_meta_falls_back_to_the_weights_file(models):
    (models / "ckpt_0002.json").unlink()
    info = {c.name: c for c in list_checkpoints(models)}["ckpt_0002.pt"]
    assert info.iter == 2 and info.games == 20   # read from the .pt itself


# --- levels persistence -----------------------------------------------------

def test_default_levels_cover_both_sides(models):
    lv = default_levels(models)
    assert lv.n_levels == 5
    for side in (ATT, DEF):
        assert len(lv.specs(side)) == 5
        assert lv.data["calibrated"] is False


def test_levels_roundtrip_and_agents_play(models):
    path = models / "levels.json"
    default_levels(models).save(path)
    lv = Levels.load(path)
    st = initial_state()
    legal = set(int(x) for x in legal_actions(st))
    for side in ("attacker", "defender"):
        for level in range(1, 6):
            a = lv.agent(side, level, rng=rng(level))
            a.new_game()
            assert a.select_action(st) in legal
            assert lv.label(side, level)
    with pytest.raises(ValueError):
        lv.spec(ATT, 6)


def test_load_levels_falls_back_when_uncalibrated(models):
    lv = load_levels(models / "levels.json", models_dir=models)
    assert lv.data["calibrated"] is False
    lv.save(models / "levels.json")
    assert load_levels(models / "levels.json").data["calibrated"] is False


def test_build_levels_document_shape():
    ladders = {"attacker": [spec("random")] * 5,
               "defender": [spec("random")] * 5}
    doc = build_levels(ladders, {"attacker": [], "defender": []},
                       {"calibrated": True})
    assert doc["version"] == 1 and doc["n_levels"] == 5
    assert doc["calibrated"] is True and "created" in doc


# --- the agent match driver -------------------------------------------------

def test_play_agent_game_terminates_and_scores():
    cfg = AgentMatchConfig(move_limit=40, opening_plies=4)
    result, plies = play_agent_game(HeuristicAgent(rng=rng(1)),
                                    RandomAgent(rng=rng(2)), cfg)
    assert result is not None and 0 < plies <= 40
    assert score_for(ATT, result.winner) + score_for(DEF, result.winner) == 1.0


def test_score_for():
    assert score_for(ATT, ATT) == 1.0
    assert score_for(ATT, DEF) == 0.0
    assert score_for(DEF, None) == 0.5


def test_run_jobs_multiprocess(models):
    cfg = AgentMatchConfig(move_limit=30, opening_plies=4)
    jobs = [Job((ATT, i), spec("heuristic"), spec("random"), seed=i)
            for i in range(4)]
    res = run_jobs(jobs, models, cfg, workers=2)
    assert len(res) == 4
    assert {r["key"] for r in res} == {(ATT, i) for i in range(4)}
    assert all(0 < r["plies"] <= 30 for r in res)


def test_run_jobs_raises_when_a_worker_dies(models, monkeypatch):
    """A crashed worker must surface as an error, not hang the run
    (the overnight gate and the calibration both drain this queue)."""
    import tafl.eval as ev
    monkeypatch.setattr(ev, "WORKER_POLL_SEC", 0.2)
    cfg = AgentMatchConfig(move_limit=30, opening_plies=4)
    jobs = [Job((ATT, i), spec("policy", "missing.pt"), spec("random"), seed=i)
            for i in range(2)]
    with pytest.raises(RuntimeError, match="worker"):
        run_jobs(jobs, models, cfg, workers=2)


def test_jobs_are_paired_across_candidates():
    """Two candidates must meet each panel opponent from identical seeds."""
    cfg = CalibrationConfig(games=2)
    jobs = build_jobs([spec("random"), spec("heuristic")],
                      [spec("random"), spec("heuristic")], cfg)
    att = [j for j in jobs if j.key[0] == ATT]
    seeds0 = [j.seed for j in att if j.key[1] == 0]
    seeds1 = [j.seed for j in att if j.key[1] == 1]
    assert seeds0 == seeds1 and len(seeds0) == 4
    assert len(jobs) == 2 * 2 * 2 * 2      # sides x candidates x panel x games


# --- ladder selection -------------------------------------------------------

def test_select_ladder_is_monotone_and_spread():
    scores = [0.0, 0.05, 0.1, 0.3, 0.32, 0.5, 0.7, 0.9, 1.0]
    picked = select_ladder(scores, 5)
    got = [scores[i] for i in picked]
    assert len(set(picked)) == 5
    assert got == sorted(got)
    assert got[0] == 0.0 and got[-1] == 1.0


def test_select_ladder_uses_measured_not_assumed_order():
    """A candidate that measures weaker than its predecessor must not be
    placed above it just because it was expected to be stronger."""
    scores = [0.1, 0.9, 0.2, 0.8, 0.5, 0.35, 0.65]
    got = [scores[i] for i in select_ladder(scores, 5)]
    assert got == sorted(got)


def test_select_ladder_with_too_few_candidates():
    picked = select_ladder([0.2, 0.8], 5)
    assert len(picked) == 5 and picked == sorted(picked)


def test_margins():
    assert margins([0.1, 0.3, 0.35]) == [None, 0.2, 0.05]


def test_tally_counts_wins_draws_losses():
    from tafl.engine import Result
    results = [{"key": (ATT, 0), "winner": ATT, "reason": "escape", "plies": 10},
               {"key": (ATT, 0), "winner": DEF, "reason": "escape", "plies": 20},
               {"key": (ATT, 0), "winner": None, "reason": "move limit",
                "plies": 30}]
    st = tally(results, 1)[(ATT, 0)]
    assert st["w"] == 1 and st["l"] == 1 and st["d"] == 1
    assert st["score_rate"] == pytest.approx(0.5)
    assert st["avg_len"] == pytest.approx(20.0)
    assert Result(ATT, "escape").winner == ATT


def test_spread_checkpoints_samples_the_whole_curve(models):
    ck = list_checkpoints(models)
    assert spread_checkpoints(ck, 4) == ["ckpt_0002.pt", "ckpt_0002.pt",
                                         "ckpt_0004.pt", "best.pt"]
    many = ck * 5                                   # 15 "checkpoints"
    picked = spread_checkpoints(many, 4)
    assert len(picked) == 4 and picked[0] == many[0].name
    assert picked[-1] == many[-1].name
    with pytest.raises(ValueError):
        spread_checkpoints([], 4)


# --- calibration end to end -------------------------------------------------

def test_calibrate_end_to_end(models):
    cfg = CalibrationConfig(games=1, workers=1, panel_sims=2,
                            sims_ladder=(0, 2), randomness_ladder=(0.8, 0.5,
                                                                   0.25, 0.1),
                            match=AgentMatchConfig(move_limit=24,
                                                   opening_plies=4))
    ck = list_checkpoints(models)
    assert len(candidate_pool(ck, cfg)) >= 5
    assert len(panel_specs(ck, cfg)) == 4

    doc = calibrate(models, cfg)
    assert doc["calibrated"] is True
    assert doc["method"]["games_played"] > 0
    assert [c["name"] for c in doc["checkpoints"]][-1] == "best.pt"
    for side in ("attacker", "defender"):
        ladder = doc["levels"][side]
        assert [s["level"] for s in ladder] == [1, 2, 3, 4, 5]
        rates = [s["panel_score"] for s in ladder]
        assert rates == sorted(rates)          # monotone by construction
        assert ladder[0]["margin"] is None
        assert all(s["record"]["games"] > 0 for s in ladder)
        assert len(doc["candidates"][side]) == len(candidate_pool(ck, cfg))
    assert isinstance(format_table(doc), str)

    # the calibrated document is usable exactly like any levels.json
    path = models / "levels.json"
    Levels(doc, models).save(path)
    lv = load_levels(path)
    st = initial_state()
    legal = set(int(x) for x in legal_actions(st))
    for side in (ATT, DEF):
        for level in range(1, 6):
            assert lv.agent(side, level, rng=rng()).select_action(st) in legal
