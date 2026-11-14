"""Support enumeration for mixed Bayes-Nash equilibria of games whose agents all have two actions.

An *agent* is a (candidate, type) pair; its two actions are "silent" (0) and "tell" (1).  A behaviour
strategy is one probability p_c of telling per agent.  The payoff difference of agent c = (i, theta),

    f_c = E[u | tell] - E[u | silent],

depends only on the other *candidates'* strategies (never on i's other types), and is multilinear in their
type probabilities, because a candidate's pure strategy s = sum_theta a_theta 2^theta is drawn with
independent per-type probabilities.  f_c is read from the interim value tables V_i of AgentFormGame.

Procedure (exact up to the estimated table, which is treated as the game):

 1. Iterated strict dominance.  An action is removed from an agent if it loses against EVERY pure
    profile of the others that survived so far.  Payoffs are multilinear in the others' mixing, so
    this is domination against every mixed strategy too, and a removed action is played in no
    equilibrium, pure or mixed.
 2. Support enumeration over the agents that still have two actions: each is silent (0), tells (1) or
    mixes.  For every support the indifference conditions f_c = 0 of the mixing agents are solved
    and the candidate is verified against ALL deviations of ALL agents with the full payoff function
    (a mixed profile is an equilibrium iff every agent is indifferent on its support and does not
    gain outside it).
 3. How the indifference conditions are solved decides what can be claimed:
      k = 0 mixing agents  pure profile: checked directly (must equal enumerate_pure_bne).
      k = 1                f_c does not involve p_c: an exact tie (|f_c| < tie_tol) or nothing.
      k = 2                two agents of different candidates give two equations that are each
                           linear in the other agent's probability: solved exactly (multi-start
                           root finding converges to the unique solution in one Newton step).
      k >= 3               multilinear system: multi-start root finding, NOT exhaustive.
    `complete` is True only if no support with k >= 3 had to be solved.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import root

from .game import AgentFormGame


class TwoActionGame:
    """Cells = agents; cand_cells[j] = cell indices of candidate j's types (bit theta of the strategy
    index is the action of type theta); f[c] = payoff difference (tell - silent) of cell c as an array
    indexed by the pure strategy index of each OTHER candidate, in the order of others[c]."""

    def __init__(self, cand_cells: Sequence[Sequence[int]], f: Dict[int, np.ndarray], others: Dict[int, List[int]]):
        self.cand_cells = [list(c) for c in cand_cells]
        self.n_cells = sum(len(c) for c in self.cand_cells)
        self.cand_of = {}
        for j, cs in enumerate(self.cand_cells):
            for c in cs:
                self.cand_of[c] = j
        self.f = f
        self.others = others

    @classmethod
    def from_agent_form(cls, game: AgentFormGame) -> "TwoActionGame":
        if any(game.K[i] != 2 for i in game.active):
            raise ValueError("support enumeration is implemented for games in which every agent has 2 actions")
        cand_cells, f, others = [], {}, {}
        cand_pos = {i: p for p, i in enumerate(game.active)}
        for i in game.active:
            cand_cells.append([c for c, ta in enumerate(game.type_agents) if ta.player == i])
        for c, ta in enumerate(game.type_agents):
            i, th = ta.player, ta.theta
            others[c] = [cand_pos[j] for j in game.others[i]]
            diff = game.V[i][:, th, 1] - game.V[i][:, th, 0]              # (n_others,), first other least significant
            shape = [game.n_strat[j] for j in game.others[i]][::-1]
            f[c] = diff.reshape(shape).transpose(tuple(range(len(shape)))[::-1]) if shape else diff
        return cls(cand_cells, f, others)

    # ------------------------------------------------------------------ payoffs
    @staticmethod
    def strategy_weights(p: np.ndarray) -> np.ndarray:
        """Weights of the 2^t pure strategies of a candidate whose types tell with probabilities p."""
        t = len(p)
        w = np.ones(1 << t)
        for s in range(1 << t):
            for th in range(t):
                w[s] *= p[th] if (s >> th) & 1 else 1.0 - p[th]
        return w

    def f_value(self, c: int, P: np.ndarray) -> float:
        """f_c at the behaviour-strategy profile P (probability of telling per cell)."""
        val = self.f[c]
        for j in self.others[c][::-1]:                                       # contract the last axis first
            val = val @ self.strategy_weights(P[self.cand_cells[j]])
        return float(val)

    def f_all(self, P: np.ndarray) -> np.ndarray:
        return np.array([self.f_value(c, P) for c in range(self.n_cells)])

    # ------------------------------------------------------------------ dominance
    def surviving_strategies(self, j: int, allowed: List[set]) -> List[int]:
        cs = self.cand_cells[j]
        return [s for s in range(1 << len(cs)) if all(((s >> th) & 1) in allowed[c] for th, c in enumerate(cs))]

    def iterated_dominance(self, tol: float = 1e-12):
        allowed = [{0, 1} for _ in range(self.n_cells)]
        removed = []
        changed = True
        while changed:
            changed = False
            for c in range(self.n_cells):
                if len(allowed[c]) < 2:
                    continue
                surv = [self.surviving_strategies(j, allowed) for j in self.others[c]]
                vals = self.f[c][np.ix_(*surv)] if surv else self.f[c]
                if (vals < -tol).all():
                    allowed[c].discard(1); removed.append((c, 1)); changed = True
                elif (vals > tol).all():
                    allowed[c].discard(0); removed.append((c, 0)); changed = True
        return allowed, removed


@dataclass
class SupportEnumResult:
    n_cells: int
    dominated: List[tuple]                    # (cell, removed action)
    n_free: int                               # agents with two actions left after dominance
    n_supports: int                           # support profiles examined
    n_supports_k_ge_3: int                    # of which needed the (non-exhaustive) multi-start solver
    max_k: int                                # most mixing agents in an examined support
    pure: List[np.ndarray] = field(default_factory=list)     # pure equilibria (cell actions)
    mixed: List[Dict] = field(default_factory=list)          # {"p": per-cell tell probs, "k", "residual"}
    exact_ties: int = 0                       # supports with k=1 and |f_c| ~ 0 (knife-edge, reported not counted)

    @property
    def complete(self) -> bool:
        return self.n_supports_k_ge_3 == 0


def _is_equilibrium(tg: TwoActionGame, P: np.ndarray, mixing: Sequence[int], tol: float) -> bool:
    f = tg.f_all(P)
    for c in range(tg.n_cells):
        if c in mixing:
            if abs(f[c]) > tol:
                return False
        elif P[c] == 1.0 and f[c] < -tol:
            return False
        elif P[c] == 0.0 and f[c] > tol:
            return False
    return True


def support_enumeration(game_or_tg, n_starts: int = 40, seed: int = 0, tol: float = 1e-9,
                        tie_tol: float = 1e-12, max_free: int = 12) -> SupportEnumResult:
    tg = game_or_tg if isinstance(game_or_tg, TwoActionGame) else TwoActionGame.from_agent_form(game_or_tg)
    rng = np.random.default_rng(seed)
    allowed, removed = tg.iterated_dominance()
    fixed = {c: next(iter(a)) for c, a in enumerate(allowed) if len(a) == 1}
    free = [c for c, a in enumerate(allowed) if len(a) == 2]
    if len(free) > max_free:
        raise ValueError(f"{len(free)} agents left after dominance: 3^{len(free)} supports is too many")
    res = SupportEnumResult(tg.n_cells, removed, len(free), 0, 0, 0)
    seen = set()
    for choice in product((0, 1, 2), repeat=len(free)):                      # 0 silent, 1 tell, 2 mix
        res.n_supports += 1
        P = np.zeros(tg.n_cells)
        for c, a in fixed.items():
            P[c] = a
        mixing = []
        for c, ch in zip(free, choice):
            if ch == 2:
                mixing.append(c)
            else:
                P[c] = ch
        k = len(mixing)
        res.max_k = max(res.max_k, k)
        if k == 0:
            if _is_equilibrium(tg, P, mixing, tol):
                res.pure.append(P.astype(int))
            continue

        def resid(x):
            Q = P.copy()
            Q[mixing] = x
            return np.array([tg.f_value(c, Q) for c in mixing])

        if k == 1:
            # f_c does not depend on p_c: either an exact tie (any p_c is consistent) or no equilibrium here
            if abs(resid(np.array([0.5]))[0]) < tie_tol:
                res.exact_ties += 1
            continue
        if k >= 3:
            res.n_supports_k_ge_3 += 1
        starts = [np.full(k, 0.5)] + [rng.random(k) for _ in range(n_starts)]
        for x0 in starts:
            sol = root(resid, x0, method="hybr", tol=1e-13)
            x = sol.x
            if not sol.success or np.abs(resid(x)).max() > tol or (x <= 1e-9).any() or (x >= 1 - 1e-9).any():
                continue
            Q = P.copy()
            Q[mixing] = x
            key = tuple(np.round(Q, 6))
            if key in seen or not _is_equilibrium(tg, Q, mixing, tol):
                continue
            seen.add(key)
            res.mixed.append({"p": Q.tolist(), "k": k, "residual": float(np.abs(resid(x)).max())})
    return res


def dominance_reduction(game: AgentFormGame, tol: float = 1e-12):
    """Iterated strict dominance (by pure actions) for agents with any number of actions.

    Returns (allowed, removed): allowed[c] is the set of actions of cell c (ordered as game.type_agents)
    that survive, removed the list of (cell, action).  An action is removed if some surviving action of the
    same agent is strictly better against EVERY surviving pure profile of the other candidates; by
    multilinearity that holds against every mixed profile too, so no equilibrium (pure or mixed) uses a
    removed action.  If every agent keeps one action the game is dominance-solvable and that profile is
    its only equilibrium of any kind."""
    cells = game.type_agents
    allowed = [set(range(ta.n_actions)) for ta in cells]
    cell_of = {(ta.player, ta.theta): c for c, ta in enumerate(cells)}
    removed = []
    changed = True
    while changed:
        changed = False
        for c, ta in enumerate(cells):
            if len(allowed[c]) < 2:
                continue
            i, th = ta.player, ta.theta
            idx = np.zeros(1, dtype=np.int64)
            mult = 1
            for j in game.others[i]:
                Kj = game.K[j]
                surv = np.array([s for s in range(game.n_strat[j])
                                 if all(((s // Kj ** t) % Kj) in allowed[cell_of[(j, t)]] for t in range(3))], dtype=np.int64)
                idx = np.add.outer(idx, surv * mult).ravel()
                mult *= game.n_strat[j]
            vals = game.V[i][idx, th, :]                                     # (surviving profiles, K_i)
            for a in sorted(allowed[c]):
                if any(b != a and (vals[:, b] > vals[:, a] + tol).all() for b in allowed[c]):
                    allowed[c].discard(a)
                    removed.append((c, a))
                    changed = True
                    break
    return allowed, removed
