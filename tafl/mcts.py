"""PUCT Monte-Carlo tree search over the Fetlar engine.

AlphaZero-style search:

 - selection by Q + c_puct * P * sqrt(N_parent) / (1 + N_child), where every
   Q is stored from the perspective of the player to move at the *parent*
   (values flip sign at each ply on backup);
 - leaves are evaluated by a `NetEvaluator`-style callable in mini-batches:
   up to `batch_size` leaves are collected per pass using a virtual loss
   (N += 1, W -= 1 along the path) so one network call serves several
   simulations, which is what makes a Python-loop MCTS fast enough;
 - Dirichlet noise mixed into the root priors when `run(noise=True)`
   (self-play exploration; arena play runs without it);
 - the tree is reused across moves via `advance(action)`.

Edge statistics live in NumPy arrays on the parent node, indexed in step
with `node.acts` (the engine's legal action ids for that position).
"""

from __future__ import annotations

import math

import numpy as np

from .engine import MOVE_LIMIT, State, apply


def terminal_value(state: State) -> float:
    """Game outcome from the perspective of the side to move at `state`."""
    w = state.result.winner
    if w is None:
        return 0.0
    return 1.0 if w == state.to_move else -1.0


class Node:
    __slots__ = ("state", "acts", "P", "N", "W", "children")

    def __init__(self, state: State):
        self.state = state
        self.acts = None       # np.int64 legal action ids, set on expansion
        self.P = None          # priors over acts
        self.N = None          # edge visit counts (float32)
        self.W = None          # edge total value, parent's perspective
        self.children = None   # dict edge-index -> Node

    @property
    def expanded(self) -> bool:
        return self.acts is not None

    def expand(self, acts: np.ndarray, priors: np.ndarray):
        self.acts = acts
        self.P = priors.astype(np.float32, copy=True)
        n = len(acts)
        self.N = np.zeros(n, dtype=np.float32)
        self.W = np.zeros(n, dtype=np.float32)
        self.children = {}


class MCTS:
    def __init__(self, evaluator, c_puct: float = 1.5,
                 dirichlet_alpha: float = 0.1, dirichlet_eps: float = 0.25,
                 batch_size: int = 8, move_limit: int = MOVE_LIMIT,
                 rng: np.random.Generator | None = None):
        self.evaluator = evaluator
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.batch_size = batch_size
        self.move_limit = move_limit
        self.rng = rng or np.random.default_rng()
        self.root: Node | None = None

    # -- tree management ---------------------------------------------------

    def set_root(self, state: State):
        self.root = Node(state)

    def advance(self, action: int):
        """Move the root to the child reached by `action`, keeping its subtree."""
        root = self.root
        if root is None or not root.expanded:
            raise ValueError("advance() before search")
        idx = np.nonzero(root.acts == action)[0]
        if len(idx) != 1:
            raise ValueError(f"action {action} not legal at root")
        child = root.children.get(int(idx[0]))
        if child is None:
            child = Node(apply(root.state, action, self.move_limit))
        self.root = child

    # -- search ------------------------------------------------------------

    def run(self, n_sims: int, noise: bool = False) -> Node:
        root = self.root
        if root is None:
            raise ValueError("set_root() first")
        if root.state.result is not None:
            raise ValueError("root is terminal")
        if not root.expanded:
            (acts, p, _v), = self.evaluator.evaluate([root.state])
            root.expand(acts, p)
        if noise and self.dirichlet_eps > 0:
            d = self.rng.dirichlet([self.dirichlet_alpha] * len(root.acts))
            root.P = ((1 - self.dirichlet_eps) * root.P
                      + self.dirichlet_eps * d.astype(np.float32))

        sims = 0
        while sims < n_sims:
            batch = []   # (leaf, path) awaiting network evaluation
            while len(batch) < self.batch_size and sims < n_sims:
                sims += 1
                node, path = root, []
                while node.expanded and node.state.result is None:
                    i = self._select(node)
                    node.N[i] += 1.0       # virtual loss (undone on backup)
                    node.W[i] -= 1.0
                    path.append((node, i))
                    child = node.children.get(i)
                    if child is None:
                        child = Node(apply(node.state, int(node.acts[i]),
                                           self.move_limit))
                        node.children[i] = child
                    node = child
                if node.state.result is not None:
                    self._backup(path, terminal_value(node.state))
                else:
                    batch.append((node, path))
            if batch:
                results = self.evaluator.evaluate([n.state for n, _ in batch])
                for (node, path), (acts, p, v) in zip(batch, results):
                    if not node.expanded:   # may repeat within one batch
                        node.expand(acts, p)
                    self._backup(path, v)
        return root

    def _select(self, node: Node) -> int:
        q = np.divide(node.W, node.N, out=np.zeros_like(node.W),
                      where=node.N > 0)
        u = (self.c_puct * math.sqrt(node.N.sum() + 1.0)
             * node.P / (1.0 + node.N))
        return int(np.argmax(q + u))

    @staticmethod
    def _backup(path, v: float):
        """Propagate leaf value `v` (leaf's side-to-move perspective) up the
        path, flipping sign each ply; +1.0 undoes the virtual loss on W."""
        for node, i in reversed(path):
            v = -v
            node.W[i] += v + 1.0

    # -- results -----------------------------------------------------------

    def root_policy(self, temperature: float = 1.0):
        """(acts, probs) from root visit counts; τ→0 means argmax."""
        root = self.root
        n = root.N.astype(np.float64)
        if temperature < 0.05:
            # Break ties at random: np.argmax always returns the lowest index,
            # so a thinly searched root (sims below the legal-move count) picks
            # by action id rather than by search, and the game loops.
            p = np.zeros_like(n)
            tied = np.flatnonzero(n == n.max())
            p[int(self.rng.choice(tied))] = 1.0
        else:
            p = n ** (1.0 / temperature)
            s = p.sum()
            p = p / s if s > 0 else np.full_like(n, 1.0 / len(n))
        return root.acts, p

    def root_value(self) -> float:
        """Mean backed-up value at the root (root side-to-move perspective)."""
        root = self.root
        n = root.N.sum()
        return float(root.W.sum() / n) if n > 0 else 0.0
