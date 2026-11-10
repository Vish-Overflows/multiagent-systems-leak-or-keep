"""Verification checks that the simulator must pass before any experiment:
  C1  worked example of the introduction (0.71, 0.92, 0.71)
  C2  Lemma 1: monotonicity of Y_j in S_ij and no effect on other scores (exact, CRN)
  C3  Proposition 3: leak formula 1 - (1 - m/Q)^{n_j} in E1 and E2, and the bank-size bound
  C4  Proposition 2: under BAR own hiring is independent of own row (exact) and friends' gains
      do not depend on own type (up to Monte Carlo error)
  C5  Proposition 1: complete network, SEATS -> no sharing is the unique BNE (n=4 full solve)
      and every unit of sharing has negative gain at n=6 (local deviation check)
  C6  nobody shares at alpha = 0 (SELF utility), both rules, all environments
"""
from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Dict, List

import numpy as np

from .config import GameConfig
from .env import DayDraws, draw_days, draw_question_sets, leak_counts_batch, leak_counts_table, scores
from .game import AgentFormGame
from .solvers import enumerate_pure_bne
from .tables import build_table, type_profiles
from .utility import FRIEND, SELF, Utility, homogeneous, weights


# ----------------------------------------------------------------------------- C1
def worked_example(n_days: int = 400_000, seed: int = 0) -> Dict:
    """Aarav(21), Meera(22), Kabir(20), Diya(19); Q=10, m=3, a known question adds 4 marks;
    Aarav tells Kabir and Diya.  Exact: P(friend sees >=1 of Aarav's questions) = 1 - C(7,3)/C(10,3),
    P(someone overtakes Aarav) = 1 - (C(7,3)/C(10,3))^2;  under a bar at 20, P(Diya in) = 0.7083."""
    cfg = GameConfig(n=4, K=2, m=3, Q=10, delta=12.0, sigma=0.0, c=0.0, env="E2", network="complete")
    rng = np.random.default_rng(seed)
    skills = np.tile(np.array([21.0, 22.0, 20.0, 19.0]), (n_days, 1))
    d = draw_days(cfg, n_days, rng, skills=skills)
    d.noise[:] = 0.0
    S = np.zeros((n_days, 4, 4), dtype=np.int8)
    S[:, 0, 2] = 1
    S[:, 0, 3] = 1
    k, kij = leak_counts_batch(cfg, d, S)
    Y = scores(cfg, d, k)
    p_none = comb(7, 3) / comb(10, 3)
    kabir_sees = (kij[:, 0, 2] > 0).mean()
    diya_sees = (kij[:, 0, 3] > 0).mean()
    overtaken = ((Y[:, 2] > Y[:, 0]) | (Y[:, 3] > Y[:, 0])).mean()
    aarav_hired_seats = (np.argsort(-Y, axis=1)[:, :2] == 0).any(axis=1).mean()
    diya_in_bar = (Y[:, 3] >= 20.0).mean()
    se = lambda p: float(np.sqrt(p * (1 - p) / n_days))
    return {
        "P(friend sees >=1 question)": {"exact": 1 - p_none, "sim_kabir": float(kabir_sees), "sim_diya": float(diya_sees), "se": se(1 - p_none)},
        "P(someone overtakes Aarav)": {"exact": 1 - p_none ** 2, "sim": float(overtaken), "se": se(1 - p_none ** 2)},
        "P(Aarav hired | shares, SEATS)": {"exact": p_none ** 2, "sim": float(aarav_hired_seats), "se": se(p_none ** 2)},
        "P(Diya hired | shares, BAR at 20)": {"exact": 1 - p_none, "sim": float(diya_in_bar), "se": se(1 - p_none)},
        "n_days": n_days,
    }


# ----------------------------------------------------------------------------- C2
def lemma1_monotonicity(n_days: int = 5000, seed: int = 0) -> Dict:
    """For random S and random free entry (i,j) with S_ij = 0: Y_j(S + ij) >= Y_j(S), other Y unchanged."""
    out = {}
    rng = np.random.default_rng(seed)
    for env in ("E0", "E1", "E2"):
        cfg = GameConfig(env=env, network="complete")
        d = draw_days(cfg, n_days, rng)
        fe = cfg.free_entries
        S = (rng.random((n_days, cfg.n, cfg.n)) < 0.5)
        S = np.triu(S, 1).astype(np.int8)
        pick = rng.integers(0, len(fe), size=n_days)
        ii = np.array([fe[p][0] for p in pick]); jj = np.array([fe[p][1] for p in pick])
        S[np.arange(n_days), ii, jj] = 0
        S2 = S.copy(); S2[np.arange(n_days), ii, jj] = 1
        Y1 = scores(cfg, d, leak_counts_batch(cfg, d, S)[0])
        Y2 = scores(cfg, d, leak_counts_batch(cfg, d, S2)[0])
        dj = Y2[np.arange(n_days), jj] - Y1[np.arange(n_days), jj]
        mask = np.ones_like(Y1, dtype=bool); mask[np.arange(n_days), jj] = False
        others = np.abs((Y2 - Y1)[mask]).max()
        out[env] = {"min_dY_j": float(dj.min()), "frac_strict_increase": float((dj > 1e-12).mean()),
                    "max_|dY_other|": float(others), "pass": bool(dj.min() >= -1e-12 and others < 1e-12)}
    return out


# ----------------------------------------------------------------------------- C3
def prop3_leak_formula(n_days: int = 200_000, seed: int = 0, Q: int = 15, m: int = 3) -> Dict:
    """E[k_j] / m = 1 - (1 - m/Q)^{n_j} for n_j = 1, 2, 3 tellers (E1 and E2), and the bound Q >= m n_j / eps."""
    out = {}
    rng = np.random.default_rng(seed)
    for env in ("E1", "E2"):
        cfg = GameConfig(n=4, env=env, Q=Q, m=m, network="complete")
        d = draw_days(cfg, n_days, rng)
        res = {}
        for nj in (1, 2, 3):
            S = np.zeros((1, 4, 4), dtype=np.int8)
            for i in range(nj):
                S[0, i, 3] = 1
            k = leak_counts_table(cfg, d, S)[:, 0, 3]
            exact = 1 - (1 - m / Q) ** nj
            res[f"n_j={nj}"] = {"exact": exact, "sim": float(k.mean() / m),
                                 "se": float(k.std() / m / np.sqrt(n_days))}
        out[env] = res
    # bound: with Q >= m*n_j/eps the expected leaked fraction is <= eps
    bound = {}
    for nj, eps in ((1, 0.1), (2, 0.1), (3, 0.05)):
        Qb = int(np.ceil(m * nj / eps))
        frac = 1 - (1 - m / Qb) ** nj
        bound[f"n_j={nj},eps={eps}"] = {"Q": Qb, "leaked_fraction": frac, "holds": bool(frac <= eps)}
    out["bound"] = bound
    return out


# ----------------------------------------------------------------------------- C4
def prop2_bar_checks(table) -> Dict:
    """Under BAR: (a) P(x_i | theta, S) is exactly invariant to row i (same draws); (b) for FRIEND(alpha)
    the friends' gain from any own row is independent of own type theta_i (within MC error)."""
    cfg = table.cfg
    g = AgentFormGame(table, homogeneous(FRIEND(0.2), cfg.n), "BAR", build_V=False)
    Px = table.Px[:, :, 1, :].astype(float)
    # (a) own hiring invariant to own row: compare S and S with row i zeroed
    max_dev = 0.0
    for i in g.active:
        own_bits = np.bitwise_or.reduce(g.rowbits[i][1:]) if g.K[i] > 1 else 0
        Sidx = np.arange(g.M)
        base = Sidx & ~own_bits
        max_dev = max(max_dev, float(np.abs(Px[:, Sidx, i] - Px[:, base, i]).max()))
    # (b) friends' gain independent of own type
    gains = {}
    for i in g.active:
        friends = cfg.friends(i)
        Sidx = np.arange(g.M)
        own_bits = np.bitwise_or.reduce(g.rowbits[i][1:])
        base = Sidx & ~own_bits
        full = base | g.rowbits[i][-1]        # tell everyone in L(i)
        gain_tS = Px[:, full][:, :, friends].sum(-1) - Px[:, base][:, :, friends].sum(-1)   # (T, M)
        by_type = [gain_tS[g.profiles[:, i] == th].mean() for th in range(3)]
        gains[f"c{i + 1}"] = {"gain_by_type": [float(v) for v in by_type],
                              "spread": float(max(by_type) - min(by_type))}
    return {"max_|dP(own hire)| over own row": max_dev, "pass_a": bool(max_dev == 0.0), "friends_gain": gains}


# ----------------------------------------------------------------------------- C5
def prop1_complete_n4(n_days: int = 2000, seed: int = 0, alphas=(0.2, 0.5, 1.0)) -> Dict:
    """n = 4 complete network: full enumeration of pure BNEs under SEATS for several alpha."""
    cfg = GameConfig(n=4, K=2, env="E1", network="complete")
    tab = build_table(cfg, n_days=n_days, seed=seed, verbose=False)
    out = {}
    for al in alphas:
        for rule in ("SEATS", "BAR"):
            g = AgentFormGame(tab, homogeneous(FRIEND(al), 4), rule)
            res = enumerate_pure_bne(g)
            rates = [float(g.sharing_rate(p)[0]) for p in res.profiles]
            out[f"alpha={al},{rule}"] = {"n_bne": res.count, "sharing_rates": rates,
                                          "all_no_sharing": bool(res.count > 0 and max(rates) == 0.0)}
    return out


def local_gains_complete_n6(n_days: int = 1000, seed: int = 0, alpha: float = 1.0) -> Dict:
    """n = 6 complete network, SEATS, FRIEND(alpha <= 1): every nonzero row has negative gain for every
    type of every candidate when nobody else shares (and when everybody else shares).  Uses a table
    restricted to the needed matrices (single-row deviations from S = 0 and from S = all-ones)."""
    cfg = GameConfig(n=6, env="E1", network="complete")
    n = cfg.n
    rows = {i: [j for j in range(n) if j > i] for i in range(n)}
    S_list, index = [], {}

    def add(S):
        key = S.tobytes()
        if key not in index:
            index[key] = len(S_list)
            S_list.append(S.copy())
        return index[key]

    zero = np.zeros((n, n), dtype=np.int8)
    ones = np.triu(np.ones((n, n), dtype=np.int8), 1)
    for base in (zero, ones):
        add(base)
        for i in range(n):
            for r in range(0, 1 << len(rows[i])):
                S = base.copy()
                S[i, :] = 0
                for t, j in enumerate(rows[i]):
                    if (r >> t) & 1:
                        S[i, j] = 1
                add(S)
    S_arr = np.stack(S_list)
    tab = build_table(cfg, S_list=S_arr, n_days=n_days, seed=seed, verbose=False)
    profiles = type_profiles(n)
    out = {}
    for rule_i, rule in enumerate(("SEATS", "BAR")):
        Px = tab.Px[:, :, rule_i, :].astype(float)
        res = {}
        for base_name, base in (("others_silent", zero), ("others_share_all", ones)):
            worst = -np.inf
            n_pos = 0; n_tot = 0
            for i in range(n - 1):
                w_lin, _ = weights(FRIEND(alpha), cfg, i)
                # base for player i: base with row i zeroed
                Sb = base.copy(); Sb[i, :] = 0
                b_idx = index[Sb.tobytes()]
                for r in range(1, 1 << len(rows[i])):
                    S = Sb.copy()
                    for t, j in enumerate(rows[i]):
                        if (r >> t) & 1:
                            S[i, j] = 1
                    s_idx = index[S.tobytes()]
                    gain_t = (Px[:, s_idx] - Px[:, b_idx]) @ w_lin - cfg.c * bin(r).count("1")
                    for th in range(3):
                        gval = gain_t[profiles[:, i] == th].mean()
                        worst = max(worst, gval); n_tot += 1; n_pos += int(gval > 0)
            res[base_name] = {"max_gain": float(worst), "n_positive": n_pos, "n_total": n_tot}
        out[rule] = res
    out["alpha"] = alpha
    return out


# ----------------------------------------------------------------------------- C6
def no_sharing_at_alpha0(tables: Dict[str, object]) -> Dict:
    out = {}
    for env, tab in tables.items():
        for rule in ("SEATS", "BAR"):
            g = AgentFormGame(tab, homogeneous(SELF, tab.cfg.n), rule)
            res = enumerate_pure_bne(g)
            rates = [float(g.sharing_rate(p)[0]) for p in res.profiles]
            out[f"{env}/{rule}"] = {"n_bne": res.count, "max_sharing_rate": max(rates) if rates else None,
                                    "pass": bool(res.count == 1 and rates[0] == 0.0)}
    return out
