"""Game configuration: candidates, company, environment and friendship network.

Notation follows the report: n candidates interview in the fixed order 1..n
(0-indexed internally), a symmetric 0/1 matrix W records friendships, the
company has K seats, a bank of Q questions and asks m per interview.  Scores
are Y_j = s_j + (Delta * k_j + sum_q eta_jq) / m with eta ~ N(0, sigma^2).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Tuple

import numpy as np
from scipy.special import ndtri

ENVS = ("E0", "E1", "E2")
RULES = ("SEATS", "BAR")
TYPE_NAMES = ("weak", "average", "strong")


def two_groups(n: int = 6) -> np.ndarray:
    """Friend groups {1,3,5,...} and {2,4,6,...} (positions by parity)."""
    W = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(n):
            if i != j and (i % 2) == (j % 2):
                W[i, j] = 1
    return W


def pairs(n: int = 6) -> np.ndarray:
    """Disjoint friend pairs {1,2},{3,4},{5,6}."""
    W = np.zeros((n, n), dtype=np.int8)
    for i in range(0, n - 1, 2):
        W[i, i + 1] = W[i + 1, i] = 1
    return W


def star(n: int = 6, hub: int = 2) -> np.ndarray:
    """One hub (default: position 3, the median interviewee) friends with everyone."""
    W = np.zeros((n, n), dtype=np.int8)
    for j in range(n):
        if j != hub:
            W[hub, j] = W[j, hub] = 1
    return W


def complete(n: int = 6) -> np.ndarray:
    W = np.ones((n, n), dtype=np.int8)
    np.fill_diagonal(W, 0)
    return W


NETWORKS = {
    "two_groups": two_groups,
    "pairs": pairs,
    "star": star,
    "complete": complete,
}


@dataclass(frozen=True)
class GameConfig:
    n: int = 6
    K: int = 2
    m: int = 3
    Q: int = 15
    delta: float = 1.0
    sigma: float = 1.0
    c: float = 0.001
    env: str = "E1"
    network: str = "two_groups"
    W: Tuple[Tuple[int, ...], ...] = field(default=None)  # filled in __post_init__

    def __post_init__(self):
        if self.W is None:
            W = NETWORKS[self.network](self.n)
            object.__setattr__(self, "W", tuple(tuple(int(x) for x in row) for row in W))
        assert self.env in ENVS

    # ---- derived quantities -------------------------------------------------
    @property
    def Wmat(self) -> np.ndarray:
        return np.array(self.W, dtype=np.int8)

    def friends(self, i: int) -> List[int]:
        return [j for j in range(self.n) if self.W[i][j]]

    def later_friends(self, i: int) -> List[int]:
        """L(i): friends who interview after i (the only people i can tell)."""
        return [j for j in range(self.n) if self.W[i][j] and j > i]

    @property
    def free_entries(self) -> List[Tuple[int, int]]:
        """Ordered list of (i, j) with W_ij = 1 and i < j: the free entries of S."""
        return [(i, j) for i in range(self.n) for j in range(i + 1, self.n) if self.W[i][j]]

    @property
    def n_free(self) -> int:
        return len(self.free_entries)

    @property
    def tau(self) -> float:
        """BAR threshold tau_K = sqrt(1 + sigma^2/m) * Phi^{-1}(1 - K/n)."""
        return float(np.sqrt(1.0 + self.sigma ** 2 / self.m) * ndtri(1.0 - self.K / self.n))

    @property
    def leak_p(self) -> float:
        """Per-question probability that a shared question reappears (E1/E2: m/Q, E0: 1)."""
        return 1.0 if self.env == "E0" else self.m / self.Q

    def with_(self, **kw) -> "GameConfig":
        if "network" in kw and "W" not in kw:
            kw["W"] = None
        return replace(self, **kw)


# Tercile boundaries of N(0,1): types are weak (< -0.4307), average, strong (> 0.4307).
TERCILE_EDGES = np.array([ndtri(1.0 / 3.0), ndtri(2.0 / 3.0)])


def type_of(skill: np.ndarray) -> np.ndarray:
    """theta_i = tercile of s_i: 0 weak, 1 average, 2 strong."""
    return np.searchsorted(TERCILE_EDGES, skill).astype(np.int8)
