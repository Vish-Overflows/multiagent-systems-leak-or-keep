"""Analysis helpers shared by the X1/X2/X4 scripts."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .config import GameConfig
from .game import AgentFormGame, popcount_int
from .solvers import (best_response_dynamics, bootstrap_exploitability, br_dynamics_from_many_starts,
                      enumerate_pure_bne, replicator_dynamics)
from .tables import PayoffTable, build_table, type_profiles
from .utility import FRIEND, SELF, Utility, homogeneous, weights


def loss_gain_cells(game: AgentFormGame) -> Dict[str, Dict[str, float]]:
    """At the silent baseline (nobody else shares): own loss ell = P(hired | silent) - P(hired | tell all
    later friends) and friends' gain G = sum_{j in F(i)} [P(x_j | tell all) - P(x_j | silent)], per cell."""
    Px = game.table.Px[:, :, game.r, :].astype(float)
    out = {}
    for ta in game.type_agents:
        i, th = ta.player, ta.theta
        rows = game.profiles[:, i] == th
        full = game.rowbits[i][-1]
        ell = (Px[rows, 0, i] - Px[rows, full, i]).mean()
        friends = game.cfg.friends(i)
        G = (Px[rows, full][:, friends].sum(-1) - Px[rows, 0][:, friends].sum(-1)).mean()
        out[ta.name] = {"ell": float(ell), "G": float(G),
                        "alpha_star_local": float((ell + game.cfg.c) / G) if G > 0 else float("inf")}
    return out


def sweep_alpha(tab: PayoffTable, rule: str, alphas: Sequence[float], n_boot: int = 200, boot_seed: int = 0,
                do_replicator: bool = False, verbose: bool = False) -> Dict:
    """Pure BNEs of FRIEND(alpha) for every alpha on the grid, with BR dynamics, bootstrap eps and
    (optionally) replicator dynamics.  Uses affinity of utilities in alpha: two V tables only."""
    n = tab.cfg.n
    g0 = AgentFormGame(tab, homogeneous(SELF, n), rule)
    g1 = AgentFormGame(tab, homogeneous(FRIEND(1.0), n), rule)
    lg = loss_gain_cells(g0)
    cells = [ta.name for ta in g0.type_agents]
    records = []
    for al in alphas:
        g = AgentFormGame.affine(g0, g1, al, pop=homogeneous(FRIEND(al), n))
        res = enumerate_pure_bne(g)
        rec = {"alpha": float(al), "n_bne": res.count, "n_eps_bne_1e-3": res.eps_bne(1e-3),
               "min_exploitability_all": float(res.exploitability_all.min())}
        if res.count > 0:
            rates = np.array([float(g.sharing_rate(p)[0]) for p in res.profiles])
            cells_act = np.stack([g.cells_from_profile(p) for p in res.profiles])     # (n_bne, C)
            tells = popcount_int(cells_act)                                           # (n_bne, C)
            rec.update({"sharing_rate_min": float(rates.min()), "sharing_rate_max": float(rates.max()),
                        "sharing_rate_mean": float(rates.mean()),
                        "cell_tells_first": tells[0].tolist(),
                        "cell_shares_all_bne": (tells > 0).all(axis=0).tolist(),
                        "cell_shares_some_bne": (tells > 0).any(axis=0).tolist(),
                        "first_bne": g.describe_profile(res.profiles[0]),
                        "welfare_first": float(g.expected_utilities(res.profiles[0]).sum()),
                        "profiles": res.profiles.tolist()})
            brd = br_dynamics_from_many_starts(g, 8)
            rec["br_dynamics"] = {"n_converged": brd["n_converged"], "n_distinct": len(brd["distinct_fixed_points"])}
            if n_boot > 0:
                bs = bootstrap_exploitability(g, res.profiles[0], R=n_boot, seed=boot_seed)
                rec["bootstrap"] = {"eps95": bs.eps95, "frac_exact_bne": bs.frac_exact_bne, "R": n_boot}
        else:
            brd = br_dynamics_from_many_starts(g, 8)
            rec["br_dynamics"] = {"n_converged": brd["n_converged"], "n_distinct": len(brd["distinct_fixed_points"])}
        if do_replicator:
            rds = [replicator_dynamics(g, seed=s, max_iter=3000) for s in range(3)]
            rec["replicator"] = [{"converged": r.converged, "exploitability": r.exploitability, "is_pure": r.is_pure,
                                  "iterations": r.iterations,
                                  "matches_bne": bool(r.is_pure and res.count > 0 and
                                                      any((r.pure_profile == p).all() for p in res.profiles))}
                                 for r in rds]
        records.append(rec)
        if verbose:
            print(f"    {tab.cfg.network}/{tab.cfg.env}/{rule} alpha={al:.3f}: {res.count} BNE, "
                  f"share={rec.get('sharing_rate_mean', float('nan')):.3f}", flush=True)
    # alpha*: smallest alpha at which the cell tells >= 1 friend in every pure BNE
    alpha_star = {}
    for c, name in enumerate(cells):
        a_all = [r["alpha"] for r in records if r.get("cell_shares_all_bne") and r["cell_shares_all_bne"][c]]
        a_some = [r["alpha"] for r in records if r.get("cell_shares_some_bne") and r["cell_shares_some_bne"][c]]
        alpha_star[name] = {"alpha_star_all": min(a_all) if a_all else None,
                            "alpha_star_some": min(a_some) if a_some else None}
    return {"rule": rule, "network": tab.cfg.network, "env": tab.cfg.env, "cells": cells, "records": records,
            "loss_gain": lg, "alpha_star": alpha_star}


# ------------------------------------------------------------------ complete network (n = 6), local analysis
def complete_local_table(cfg: GameConfig, n_days: int = 1000, seed: int = 0):
    """Table restricted to single-row deviations from S = 0 and from S = all-ones."""
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
                S = base.copy(); S[i, :] = 0
                for t, j in enumerate(rows[i]):
                    if (r >> t) & 1:
                        S[i, j] = 1
                add(S)
    tab = build_table(cfg, S_list=np.stack(S_list), n_days=n_days, seed=seed, verbose=False)
    return tab, index, rows, zero, ones


def complete_local_sweep(cfg: GameConfig, alphas: Sequence[float], n_days: int = 1000, seed: int = 0) -> Dict:
    """For each alpha and rule: is 'nobody shares' a BNE?  is 'everybody shares' a BNE?  (checks every
    deviation of every type of every candidate against those two profiles)."""
    tab, index, rows, zero, ones = complete_local_table(cfg, n_days, seed)
    n = cfg.n
    profiles = type_profiles(n)
    out = {"alphas": [float(a) for a in alphas], "SEATS": [], "BAR": []}
    for rule_i, rule in enumerate(("SEATS", "BAR")):
        Px = tab.Px[:, :, rule_i, :].astype(float)
        for al in alphas:
            rec = {"alpha": float(al)}
            for base_name, base in (("no_share", zero), ("all_share", ones)):
                worst = -np.inf
                for i in range(n - 1):
                    w_lin, _ = weights(FRIEND(al), cfg, i)
                    Sb = base.copy(); Sb[i, :] = 0
                    own_rows = []
                    for r in range(0, 1 << len(rows[i])):
                        S = Sb.copy()
                        for t, j in enumerate(rows[i]):
                            if (r >> t) & 1:
                                S[i, j] = 1
                        own_rows.append((r, index[S.tobytes()]))
                    played_r = 0 if base is zero else (1 << len(rows[i])) - 1
                    eu = {r: Px[:, s] @ w_lin - cfg.c * bin(r).count("1") for r, s in own_rows}
                    for th in range(3):
                        mask = profiles[:, i] == th
                        base_val = eu[played_r][mask].mean()
                        worst = max(worst, max(eu[r][mask].mean() - base_val for r, _ in own_rows))
                rec[base_name + "_is_bne"] = bool(worst <= 0.0)
                rec[base_name + "_max_gain"] = float(worst)
            out[rule].append(rec)
    return out
