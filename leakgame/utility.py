"""Utility types.  Every utility is linear in the hiring indicators x and in the
pairwise products x_i x_j, so expected utilities are linear in the simulated
tables P[x_j] and P[x_i x_j]:

  SELF         u_i = x_i - c|S_i|
  FRIEND(a)    u_i = SELF + alpha * sum_{j in F(i)} x_j
  CLAN(a,b)    u_i = FRIEND - beta * sum_{j not in F(i) U {i}} x_j
  FAIR(a,b)    u_i = SELF - (1/(n-1)) sum_j [ a (x_j - x_i)^+ + b (x_i - x_j)^+ ]   (Fehr-Schmidt)
               with (x_j - x_i)^+ = x_j - x_i x_j  and (x_i - x_j)^+ = x_i - x_i x_j.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import List, Sequence

import numpy as np

from .config import GameConfig


@dataclass(frozen=True)
class Utility:
    kind: str = "FRIEND"
    alpha: float = 0.2
    beta: float = 0.0
    a: float = 0.0
    b: float = 0.0

    def label(self) -> str:
        if self.kind == "SELF":
            return "SELF"
        if self.kind == "FRIEND":
            return f"FRIEND({self.alpha:g})"
        if self.kind == "CLAN":
            return f"CLAN({self.alpha:g},{self.beta:g})"
        return f"FAIR({self.a:g},{self.b:g})"


SELF = Utility("SELF", 0.0)


def FRIEND(alpha: float) -> Utility:
    return Utility("FRIEND", alpha)


def CLAN(alpha: float, beta: float) -> Utility:
    return Utility("CLAN", alpha, beta)


def FAIR(a: float, b: float) -> Utility:
    return Utility("FAIR", 0.0, 0.0, a, b)


def weights(u: Utility, cfg: GameConfig, i: int):
    """Return (w_lin, w_pair): E u_i = sum_j w_lin[j] P[x_j] + sum_{j!=i} w_pair[j] P[x_i x_j] - c|S_i|."""
    n = cfg.n
    w_lin = np.zeros(n)
    w_pair = np.zeros(n)
    friends = cfg.friends(i)
    if u.kind == "FAIR":
        w_lin[i] = 1.0 - u.b
        for j in range(n):
            if j != i:
                w_lin[j] = -u.a / (n - 1)
                w_pair[j] = (u.a + u.b) / (n - 1)
        return w_lin, w_pair
    w_lin[i] = 1.0
    if u.kind in ("FRIEND", "CLAN"):
        for j in friends:
            w_lin[j] += u.alpha
    if u.kind == "CLAN":
        for j in range(n):
            if j != i and j not in friends:
                w_lin[j] -= u.beta
    return w_lin, w_pair


def realized_utility(u: Utility, cfg: GameConfig, i: int, x: np.ndarray, n_told: np.ndarray) -> np.ndarray:
    """Realised u_i from hiring indicators x (..., n) and the number of friends told (...)."""
    w_lin, w_pair = weights(u, cfg, i)
    val = x @ w_lin + x[..., i] * (x @ w_pair) - cfg.c * n_told
    return val


def reward_range(u: Utility, cfg: GameConfig, i: int):
    """Exact [min, max] of u_i over all hiring vectors and own actions (used to normalise rewards to [0,1])."""
    w_lin, w_pair = weights(u, cfg, i)
    xs = np.array(list(product([0, 1], repeat=cfg.n)), dtype=float)
    vals = xs @ w_lin + xs[:, i] * (xs @ w_pair)
    lo = vals.min() - cfg.c * len(cfg.later_friends(i))
    hi = vals.max()
    return float(lo), float(hi)


def homogeneous(u: Utility, n: int) -> List[Utility]:
    return [u] * n


def population_label(pop: Sequence[Utility]) -> str:
    labels = [u.label() for u in pop]
    if len(set(labels)) == 1:
        return labels[0]
    return "mixed[" + ",".join(labels) + "]"
