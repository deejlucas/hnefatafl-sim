# Hnefatafl RL Project Plan

Build a reinforcement-learning system that learns to play **Fetlar Hnefatafl** (11x11) via
AlphaZero-style self-play, persists five difficulty levels per side, analyzes side balance,
and lets a human play against any trained AI in a browser.

**Stack:** Python + PyTorch (Apple MPS GPU), NumPy game engine, FastAPI + vanilla JS web UI.
**Training budget:** overnight (~8–12h) on this machine, until skill plateaus.

---

## Rules reference (Fetlar, from aagenielsen.dk/fetlar_rules_en.php)

- 11x11 board. 24 attackers (black), 12 defenders + 1 king (white). Attackers move first.
- All pieces move like rooks (any number of vacant squares orthogonally). No diagonal moves.
- **Restricted squares:** the central throne and the 4 corners. Only the king may occupy them
  (pieces may pass through the empty throne). They are *hostile*: they can substitute for an
  enemy piece in a capture.
- **Capture:** a piece is captured when the mover sandwiches it between two enemies (or an
  enemy and a hostile square) orthogonally. Moving *into* a sandwich is safe. Multiple
  captures in one move are possible.
- **King capture:** king is captured only when surrounded on all 4 orthogonal sides by
  attackers, or on 3 sides with the throne as the 4th.
- **Win — defenders:** king reaches any corner square.
- **Win — attackers:** capture the king, or form an unbroken ring encircling king + all
  defenders (no defender can ever escape).
- A player with no legal move loses. Perpetual repetition → draw (we will implement a
  threefold-repetition draw and a move-limit draw to keep self-play games finite).

Phase 1 includes re-verifying edge cases against the source page and its linked
clarifications before freezing the engine.

---

## Phase 1 — Game engine (`tafl/engine.py`)

Goal: a correct, fast, RL-friendly implementation of Fetlar rules.

- Board as NumPy int8 array (0 empty, 1 attacker, 2 defender, 3 king) plus side-to-move,
  move counter, and a Zobrist hash history for repetition detection.
- Core API designed for RL loops:
  - `legal_moves(state) -> list[Move]` and a fixed **action encoding**:
    `action = from_square * 40 + move_index` where move_index encodes direction (4) ×
    distance (1–10) → action space 121 × 40 = 4840. Legal-move mask as a bool vector.
  - `apply(state, action) -> state` (captures, terminal detection), immutable-style for
    MCTS tree reuse.
  - `encode(state) -> tensor` — planes for attacker/defender/king/side-to-move/move-count.
  - Encirclement win detected by flood fill from board edges.
- Performance target: ≥20k moves/sec in pure NumPy (vectorized move gen); profile and
  optimize hot paths since self-play throughput bounds training quality.
- **Tests first-class:** unit tests for every rule edge case (hostile squares, throne
  interactions, king capture variants, edge behavior, shieldwall-absence, repetition,
  no-move loss, encirclement) plus random-playout invariant tests. The engine must be
  trusted before any training starts.

## Phase 2 — RL framework (`tafl/net.py`, `tafl/mcts.py`, `tafl/selfplay.py`, `tafl/train.py`)

AlphaZero-style, sized for a laptop:

- **Network:** small ResNet (~6 residual blocks, 64–96 channels) with policy head
  (4840 logits, masked) and value head (tanh, from side-to-move perspective). One shared
  network plays both sides — it learns both roles from self-play; "defender AI" and
  "attacker AI" are the same net queried from different sides.
- **MCTS:** PUCT with Dirichlet noise at root, temperature schedule (τ=1 early moves →
  greedy), batched network evaluation for speed. ~100–200 simulations/move during
  self-play (tuned to hit the overnight budget).
- **Self-play loop:** multiple worker processes generate games into a replay buffer
  (positions, MCTS visit-count policies, game outcomes). Draw outcomes scored 0, with a
  small penalty option if draws dominate.
- **Training loop:** sample from replay buffer, loss = policy cross-entropy + value MSE +
  L2. Checkpoint every N games.
- **Gating/evaluation:** periodically pit the latest net against the previous best
  (fixed-seed match, both colors); track Elo over training to detect the plateau.

## Phase 3 — Persistence & difficulty levels (`tafl/agents.py`, `models/`)

- Checkpoints saved as `models/ckpt_<iter>.pt` with metadata JSON (training games seen,
  eval Elo, timestamp).
- **Difficulty levels 1–5** (per side, as requested) defined after training from the Elo
  curve, combining two dials:
  - which checkpoint (early = weaker understanding),
  - MCTS simulation budget at play time (e.g., level 1 = raw policy sampling from an early
    checkpoint; level 5 = final checkpoint + 400+ sims).
- Persisted as `models/levels.json` mapping `(side, level) → {checkpoint, sims,
  temperature}`.
- **Asymmetry-aware calibration.** The game is structurally asymmetric, so raw win rates
  between mixed-side opponents conflate strength with side advantage. Two safeguards:
  - All strength-comparison matches (training gates, balance study) are *paired and
    color-balanced*: equal games with each configuration on each side.
  - Difficulty ladders are calibrated *within a side*: Attacker level N must clearly
    outscore Attacker level N-1 against an identical fixed panel of defender opponents
    of assorted strengths (and symmetrically for defender levels). No cross-side win-rate
    target is used, so the systematic side advantage cancels out of the ladder.
  - The equal-strength color-balanced baseline from the Phase 5 balance study defines the
    expected score for a "fair fight," which the ladder margins are interpreted against.
- A `HeuristicAgent` (material + king-distance-to-corner evaluation, 1-ply) as a fixed
  external baseline for measuring absolute progress.

## Phase 4 — Run training to plateau

- Kick off the overnight run as a background process with logging (games played, loss
  curves, Elo vs previous checkpoints and vs the heuristic baseline, draw rate,
  win rate by side, average game length).
- Plateau criterion: new checkpoints stop beating the reigning best (<55% over ~100-game
  gated matches) for several consecutive evaluations.
- Monitor periodically; adjust sims/network size only if throughput or learning is
  clearly off in the first hour.

## Phase 5 — Balance report (`report.md`)

Answer: **is there a systematic advantage for one side?**

- Evidence: side win rates across self-play at each training stage (bias early vs late),
  final-net symmetric matches (same strength both sides, many games, both from fixed and
  varied openings), win rates when strong plays weak on each side, average game length and
  victory type (corner escape vs king capture vs encirclement).
- Compare against empirical human data: aagenielsen.dk's measured balances
  (https://aagenielsen.dk/tafl_balances.php) show Fetlar 11x11 at ~142 defender wins per
  100 attacker wins (defenders favored), with defenders also able to force draws
  disproportionately. Our self-play result either corroborates this independently or
  reveals a gap between human play and near-optimal play — either is a finding.
- Prior art context for the report: classical minimax engines exist (OpenTafl and its
  Computer Tafl Open tournaments); forum-documented ML attempts (2007 evolutionary paper,
  2017 supervised proposal, 2019 AlphaZero proposal) never produced a working engine, so
  a completed self-play RL treatment of Fetlar appears to be novel.
- Include training curves and methodology, limitations (compute-bounded strength).

## Phase 6 & 7 — Human play web app (`app/`)

- **Backend:** FastAPI server. Endpoints: new game (side + difficulty 1–5), game state,
  legal moves for a square, submit move, AI reply (runs MCTS at the level's sim budget),
  resign/restart. Loads agents per `levels.json`.
- **Frontend:** single-page vanilla JS/CSS board — click piece → highlighted legal
  destinations → click to move; capture/last-move highlighting; king/throne/corner
  styling; game-over banner with victory type; side + difficulty picker on a start
  screen; move log.
- I'll verify the full flow in the browser pane (play games as both sides at multiple
  levels) before handing it over.

## Project layout

```
tafl/            engine.py, net.py, mcts.py, selfplay.py, train.py, agents.py, eval.py
tests/           test_engine.py, test_mcts.py, test_levels.py
models/          checkpoints + levels.json
app/             server.py, static/ (index.html, board.js, style.css)
scripts/         run_training.py, calibrate_levels.py, balance_matches.py
report.md        findings write-up
plan.md          this file
```

## Order of execution & checkpoints with you

1. Engine + tests (Phase 1) — I'll confirm rule edge cases against the source site.
2. RL stack + short smoke-training run (~15 min) to validate learning happens (Phase 2–3).
3. Launch overnight training (Phase 4) — you leave it running; I'll set up logging so
   progress is inspectable anytime.
4. Calibrate levels, run balance matches, write report (Phases 3/5).
5. Build and test the web app (Phases 6–7).

## Risks & mitigations

- **Self-play throughput too low** → shrink network/sims, batch MCTS leaves, consider
  torch.compile; engine is profiled early for this reason.
- **Draw-heavy or degenerate self-play** (attackers shuffling forever) → move-limit +
  repetition draws, small draw penalty, temperature tuning.
- **Weak low levels not fun / strong levels too similar** → level calibration phase
  explicitly measures inter-level win rates and adjusts checkpoint/sim pairs.
- **MPS quirks in PyTorch** → fall back to CPU for inference if MPS gives wrong/slow
  results; small net keeps CPU viable.
