"""Vectorised simulation of one interview "day" (one copy of the stage game).

Randomness that does not depend on the sharing matrix S is drawn once per day
(`DayDraws`) so that every S (and both hiring rules) can be evaluated on the
same draws: common random numbers across arms, which is what makes the
estimated payoff *differences* precise.

Leak environments (k_j = number of j's m questions seen in advance):
  E0  every shared question reappears:        k_j = m * 1[n_j > 0]
  E1  independent per-(teller, slot) leaks:   slot q of j is leaked iff some
      teller i with S_ij = 1 has bern[i, j, q] = 1,  bern ~ Bernoulli(m/Q);
      hence k_j ~ Bin(m, 1 - (1 - m/Q)^{n_j}).
  E2  actual question sets A_i ~ Unif([Q] choose m): k_j = |A_j ∩ ∪_{i: S_ij=1} A_i|.
All three are monotone in S (Lemma 1) because the draws are fixed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import GameConfig


@dataclass
class DayDraws:
    skills: np.ndarray            # (B, n) skills s_i
    noise: np.ndarray             # (B, n) marking noise term (1/m) sum_q eta_jq  ~ N(0, sigma^2/m)
    bern: np.ndarray | None       # (B, n, n, m) bool, E1 only: teller i covers slot q of j
    sets: np.ndarray | None       # (B, n) uint64 bitmasks, E2 only: question sets A_i


def draw_question_sets(rng: np.random.Generator, B: int, n: int, Q: int, m: int) -> np.ndarray:
    """Uniform m-subsets of {0..Q-1} as 64-bit bitmasks, shape (B, n).  Requires Q <= 64."""
    assert Q <= 64, "E2 question sets are stored as 64-bit masks"
    # the m smallest of Q i.i.d. uniforms are a uniformly random m-subset
    u = rng.random((B, n, Q))
    idx = np.argpartition(u, m, axis=-1)[..., :m]          # (B, n, m) distinct question ids
    masks = np.bitwise_or.reduce((np.uint64(1) << idx.astype(np.uint64)), axis=-1)
    return masks.astype(np.uint64)


def draw_days(cfg: GameConfig, B: int, rng: np.random.Generator, skills: np.ndarray | None = None) -> DayDraws:
    n, m = cfg.n, cfg.m
    if skills is None:
        skills = rng.standard_normal((B, n))
    noise = rng.standard_normal((B, n)) * (cfg.sigma / np.sqrt(m))
    bern = sets = None
    if cfg.env == "E1":
        bern = rng.random((B, n, n, m)) < (m / cfg.Q)
    elif cfg.env == "E2":
        sets = draw_question_sets(rng, B, n, cfg.Q, m)
    return DayDraws(skills=skills, noise=noise, bern=bern, sets=sets)


def popcount(x: np.ndarray) -> np.ndarray:
    return np.bitwise_count(x).astype(np.int16)


# --------------------------------------------------------------------------- #
# Leak counts for a *list* of sharing matrices evaluated on the same days.
# --------------------------------------------------------------------------- #
def leak_counts_table(cfg: GameConfig, d: DayDraws, S: np.ndarray) -> np.ndarray:
    """S: (M, n, n) int/bool.  Returns k: (B, M, n) number of leaked questions per candidate."""
    n, m = cfg.n, cfg.m
    M = S.shape[0]
    B = d.skills.shape[0]
    Sb = S.astype(bool)
    if cfg.env == "E0":
        n_tellers = Sb.sum(axis=1)                                   # (M, n)
        k = (m * (n_tellers > 0)).astype(np.int16)                   # (M, n)
        return np.broadcast_to(k[None], (B, M, n)).copy()
    if cfg.env == "E1":
        leaked = np.zeros((B, M, n, m), dtype=bool)
        for i in range(n):
            if not Sb[:, i, :].any():
                continue
            # S[M, i, j] & bern[B, i, j, q]
            leaked |= Sb[None, :, i, :, None] & d.bern[:, None, i, :, :]
        return leaked.sum(axis=-1).astype(np.int16)                  # (B, M, n)
    # E2
    union = np.zeros((B, M, n), dtype=np.uint64)
    for i in range(n):
        if not Sb[:, i, :].any():
            continue
        union |= np.where(Sb[None, :, i, :], d.sets[:, None, i, None], np.uint64(0))
    return popcount(union & d.sets[:, None, :])                      # (B, M, n)


# --------------------------------------------------------------------------- #
# Leak counts for one sharing matrix per day (Level-2 learning loop).
# --------------------------------------------------------------------------- #
def leak_counts_batch(cfg: GameConfig, d: DayDraws, S: np.ndarray):
    """S: (B, n, n).  Returns k: (B, n) and kij: (B, n, n) where kij[b, i, j] is the number of
    j's questions that were among those i told j (0 if S_ij = 0).  kij is what a teller learns
    from her friend afterwards ("did my questions come up?"); it is the feedback of the
    model-based learner."""
    n, m = cfg.n, cfg.m
    Sb = S.astype(bool)
    if cfg.env == "E0":
        kij = (m * Sb).astype(np.int16)
        k = (m * (Sb.sum(axis=1) > 0)).astype(np.int16)
        return k, kij
    if cfg.env == "E1":
        cover = Sb[:, :, :, None] & d.bern                            # (B, i, j, q)
        kij = cover.sum(axis=-1).astype(np.int16)                     # (B, i, j)
        leaked = cover.any(axis=1)                                    # (B, j, q)
        return leaked.sum(axis=-1).astype(np.int16), kij
    # E2
    inter = popcount(d.sets[:, :, None] & d.sets[:, None, :])         # (B, i, j) |A_i ∩ A_j|
    kij = np.where(Sb, inter, 0).astype(np.int16)
    union = np.bitwise_or.reduce(np.where(Sb, d.sets[:, :, None], np.uint64(0)), axis=1)  # (B, j)
    k = popcount(union & d.sets)
    return k, kij


# --------------------------------------------------------------------------- #
# Scores, hiring and company value.
# --------------------------------------------------------------------------- #
def scores(cfg: GameConfig, d: DayDraws, k: np.ndarray) -> np.ndarray:
    """Y_j = s_j + (Delta k_j)/m + noise_j; k may carry an extra matrix axis (B, M, n)."""
    if k.ndim == 3:
        return d.skills[:, None, :] + (cfg.delta / cfg.m) * k + d.noise[:, None, :]
    return d.skills + (cfg.delta / cfg.m) * k + d.noise


def hire_seats(Y: np.ndarray, K: int) -> np.ndarray:
    """Top-K scores are hired (ties have probability zero)."""
    n = Y.shape[-1]
    thresh = np.partition(Y, n - K, axis=-1)[..., n - K]
    return Y >= thresh[..., None]


def hire_bar(Y: np.ndarray, tau: float) -> np.ndarray:
    return Y >= tau


def hire(cfg: GameConfig, Y: np.ndarray, rule: str) -> np.ndarray:
    return hire_seats(Y, cfg.K) if rule == "SEATS" else hire_bar(Y, cfg.tau)


def company_value(cfg: GameConfig, d: DayDraws, x: np.ndarray) -> np.ndarray:
    """V = sum_{j in H} (s_j - tau_K)."""
    s = d.skills[:, None, :] if x.ndim == 3 else d.skills
    return ((s - cfg.tau) * x).sum(axis=-1)


def simulate_day(cfg: GameConfig, d: DayDraws, S: np.ndarray):
    """One day for a batch with one sharing matrix per day.  Returns dict with k, kij, Y, x (per rule), V."""
    k, kij = leak_counts_batch(cfg, d, S)
    Y = scores(cfg, d, k)
    out = {"k": k, "kij": kij, "Y": Y}
    for rule in ("SEATS", "BAR"):
        x = hire(cfg, Y, rule)
        out["x_" + rule] = x
        out["V_" + rule] = company_value(cfg, d, x)
    return out
