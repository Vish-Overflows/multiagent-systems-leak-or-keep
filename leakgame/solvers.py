"""Level-1 solvers on the agent-form game.

* enumerate_pure_bne       exhaustive check of every pure strategy profile
* best_response_dynamics   asynchronous best-response iteration from a start
* bootstrap_exploitability block bootstrap of the exploitability of a profile
* replicator_dynamics      mixed equilibria (agent form, Euler-discretised replicator)
* cce_lp                   extreme points of the coarse-correlated-equilibrium polytope
                           (max / min expected sharing, max welfare) by linear programming
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import linprog

from .game import AgentFormGame, popcount_int


# ----------------------------------------------------------------------------- pure BNE
@dataclass
class PureBNEResult:
    profiles: np.ndarray            # (B, n) pure BNEs
    exploitability_all: np.ndarray  # (N,) exploitability of every profile
    n_profiles: int
    seconds: float
    tol: float

    @property
    def count(self) -> int:
        return len(self.profiles)

    def eps_bne(self, eps: float) -> int:
        return int((self.exploitability_all <= eps).sum())


def enumerate_pure_bne(game: AgentFormGame, tol: float = 0.0, chunk: int = 1 << 18) -> PureBNEResult:
    t0 = time.time()
    allp = game.all_profiles()
    N = allp.shape[0]
    expl = np.zeros(N)
    for s in range(0, N, chunk):
        sig = allp[s:s + chunk]
        expl[s:s + chunk] = game.exploitability(sig)
    mask = expl <= tol
    return PureBNEResult(allp[mask], expl, N, time.time() - t0, tol)


def best_response_dynamics(game: AgentFormGame, start: np.ndarray, max_sweeps: int = 200,
                           order: Optional[List[int]] = None) -> Dict:
    """Asynchronous best-response dynamics over type-agents.  Returns the final profile,
    whether it converged (a fixed point = pure BNE) and the path length."""
    sig = np.array(start, dtype=np.int64).copy()
    cells = list(range(len(game.type_agents))) if order is None else list(order)
    history = [sig.copy()]
    for sweep in range(max_sweeps):
        changed = False
        for c in cells:
            ta = game.type_agents[c]
            i, th = ta.player, ta.theta
            Vi = game.interim_values(i, sig[None, :])[0, th, :]
            best = int(np.argmax(Vi))
            acts = game.actions_from_strategy(i, int(sig[i]))
            if Vi[best] > Vi[acts[th]] + 1e-15:
                acts = acts.copy()
                acts[th] = best
                sig[i] = game.strategy_from_actions(i, acts)
                changed = True
        history.append(sig.copy())
        if not changed:
            return {"profile": sig, "converged": True, "sweeps": sweep + 1, "history": history}
    return {"profile": sig, "converged": False, "sweeps": max_sweeps, "history": history}


def random_profile(game: AgentFormGame, rng: np.random.Generator) -> np.ndarray:
    return np.array([rng.integers(0, game.n_strat[i]) for i in range(game.n)], dtype=np.int64)


def br_dynamics_from_many_starts(game: AgentFormGame, n_starts: int = 8, seed: int = 0) -> Dict:
    rng = np.random.default_rng(seed)
    starts = [np.zeros(game.n, dtype=np.int64), np.array([s - 1 for s in game.n_strat], dtype=np.int64)]
    while len(starts) < n_starts:
        starts.append(random_profile(game, rng))
    results = [best_response_dynamics(game, st) for st in starts]
    finals = {tuple(r["profile"]) for r in results if r["converged"]}
    return {"results": results, "distinct_fixed_points": [np.array(f) for f in finals],
            "n_converged": sum(r["converged"] for r in results)}


# ----------------------------------------------------------------------------- bootstrap
@dataclass
class BootstrapResult:
    eps_point: float
    eps_samples: np.ndarray
    frac_exact_bne: float
    eps95: float
    per_cell_gain_mean: np.ndarray
    per_cell_gain_sd: np.ndarray


def bootstrap_exploitability(game: AgentFormGame, sig: np.ndarray, R: int = 1000, seed: int = 0) -> BootstrapResult:
    """Block-bootstrap the payoff table and recompute the exploitability of `sig`.
    eps95 is the 95th percentile: sig is an eps95-BNE with 95% bootstrap confidence."""
    rng = np.random.default_rng(seed)
    eps0, gains0 = game.profile_exploitability_direct(sig)
    eps = np.zeros(R)
    gains = np.zeros((R, len(gains0)))
    for r in range(R):
        Px = game.table.resample_blocks(rng)
        g = game.with_Px(Px, build_V=False)
        eps[r], gains[r] = g.profile_exploitability_direct(sig)
    return BootstrapResult(eps0, eps, float((eps <= 1e-12).mean()), float(np.quantile(eps, 0.95)),
                           gains.mean(axis=0), gains.std(axis=0))


def bootstrap_bne_set(game: AgentFormGame, R: int = 30, seed: int = 0) -> Dict:
    """Full re-enumeration of pure BNEs on bootstrap replicates of the table (slower)."""
    rng = np.random.default_rng(seed)
    counts, sets = [], []
    for r in range(R):
        Px = game.table.resample_blocks(rng)
        g = game.with_Px(Px, build_V=True)
        res = enumerate_pure_bne(g)
        counts.append(res.count)
        sets.append({tuple(p) for p in res.profiles})
    return {"counts": counts, "sets": sets}


# ----------------------------------------------------------------------------- replicator
@dataclass
class ReplicatorResult:
    strategies: List[np.ndarray]   # per type-agent mixed strategy
    exploitability: float
    iterations: int
    converged: bool
    is_pure: bool
    pure_profile: Optional[np.ndarray]


def _mixed_payoffs(game: AgentFormGame, x: List[np.ndarray]) -> List[np.ndarray]:
    """Expected payoff of every action of every type-agent given the mixed profile x
    (list over type_agents of (K,) distributions).  Returns list of (K,) arrays."""
    T, M, n = game.T, game.M, game.n
    cell = {(ta.player, ta.theta): c for c, ta in enumerate(game.type_agents)}
    # factor_j[t, S] = x_{j, theta_j(t)}(act_j(S))
    factors = {}
    for j in game.active:
        probs = np.stack([x[cell[(j, th)]] for th in range(3)])            # (3, K_j)
        factors[j] = probs[game.profiles[:, j][:, None], game.act_of_S[j][None, :]]   # (T, M)
    out = []
    for ta in game.type_agents:
        i, th = ta.player, ta.theta
        W = np.ones((T, M))
        for j in game.active:
            if j != i:
                W *= factors[j]
        rows = game.profiles[:, i] == th
        WE = (W * game.EU[i])[rows]                                           # (T/3, M)
        pay = np.zeros(game.K[i])
        for a in range(game.K[i]):
            pay[a] = WE[:, game.act_of_S[i] == a].sum() / rows.sum()
        out.append(pay)
    return out


def replicator_dynamics(game: AgentFormGame, x0: Optional[List[np.ndarray]] = None, step: float = 20.0,
                        max_iter: int = 4000, tol: float = 1e-10, seed: int = 0) -> ReplicatorResult:
    rng = np.random.default_rng(seed)
    if x0 is None:
        x = [rng.dirichlet(np.ones(ta.n_actions)) for ta in game.type_agents]
    else:
        x = [np.array(v, dtype=float) for v in x0]
    it = 0
    converged = False
    for it in range(1, max_iter + 1):
        pays = _mixed_payoffs(game, x)
        new = []
        delta = 0.0
        for xc, pc in zip(x, pays):
            avg = float(xc @ pc)
            xn = xc + step * xc * (pc - avg)
            xn = np.clip(xn, 1e-12, None)
            xn /= xn.sum()
            delta = max(delta, float(np.abs(xn - xc).max()))
            new.append(xn)
        x = new
        if delta < tol:
            converged = True
            break
    pays = _mixed_payoffs(game, x)
    expl = max(float(pc.max() - xc @ pc) for xc, pc in zip(x, pays))
    is_pure = all(xc.max() > 0.999 for xc in x)
    pure = game.profile_from_cells([int(np.argmax(xc)) for xc in x]) if is_pure else None
    return ReplicatorResult(x, expl, it, converged, is_pure, pure)


# ----------------------------------------------------------------------------- CCE LP
@dataclass
class CCEResult:
    objective: str
    value: float
    z: np.ndarray
    support: int
    sharing_rate: float
    welfare: float
    status: str
    seconds: float


def _cce_constraints(game: AgentFormGame, allp: np.ndarray) -> np.ndarray:
    """Dense A (n_constraints, N): coefficient of z(sigma) in the constraint
    E_z[U_i(sigma'_i, sigma_{-i}) - U_i(sigma)] <= 0, for every player i and pure deviation sigma'_i."""
    N = allp.shape[0]
    rows = []
    for i in game.active:
        Vi = game.interim_values(i, allp)                                   # (N, 3, K)
        played = np.zeros(N)
        for th in range(3):
            a = game._decode_action(i, allp[:, i], th)
            played += Vi[np.arange(N), th, a] / 3.0
        K = game.K[i]
        for sp in range(game.n_strat[i]):
            acts = game.actions_from_strategy(i, sp)
            dev = (Vi[:, 0, acts[0]] + Vi[:, 1, acts[1]] + Vi[:, 2, acts[2]]) / 3.0
            rows.append(dev - played)
    return np.stack(rows)


def cce_lp(game: AgentFormGame, objectives=("max_sharing", "min_sharing", "max_welfare"),
           slack: float = 0.0) -> List[CCEResult]:
    """Solve LPs over the CCE polytope.  `slack` > 0 gives the eps-CCE polytope."""
    allp = game.all_profiles()
    N = allp.shape[0]
    t0 = time.time()
    A = _cce_constraints(game, allp)
    share = game.sharing_rate(allp)
    welfare = game.expected_utilities(allp).sum(axis=1)
    results = []
    for obj in objectives:
        if obj == "max_sharing":
            c = -share
        elif obj == "min_sharing":
            c = share
        elif obj == "max_welfare":
            c = -welfare
        else:
            raise ValueError(obj)
        t1 = time.time()
        res = linprog(c, A_ub=A, b_ub=np.full(A.shape[0], slack), A_eq=np.ones((1, N)), b_eq=[1.0],
                      bounds=(0, None), method="highs")
        z = res.x if res.status == 0 else np.full(N, np.nan)
        results.append(CCEResult(obj, float(-res.fun if obj.startswith("max") else res.fun) if res.status == 0 else np.nan,
                                 z, int((z > 1e-9).sum()) if res.status == 0 else 0,
                                 float(z @ share) if res.status == 0 else np.nan,
                                 float(z @ welfare) if res.status == 0 else np.nan,
                                 res.message, time.time() - t1))
    return results


def cce_gap_of_distribution(game: AgentFormGame, profiles: np.ndarray, w: np.ndarray) -> float:
    """CCE gap (max deviation gain) of a distribution over pure profiles (Definition 4)."""
    w = w / w.sum()
    gap = 0.0
    for i in game.active:
        Vi = game.interim_values(i, profiles)
        played = np.zeros(profiles.shape[0])
        for th in range(3):
            a = game._decode_action(i, profiles[:, i], th)
            played += Vi[np.arange(len(w)), th, a] / 3.0
        dev = sum((w[:, None] * Vi[:, th, :]).sum(axis=0).max() for th in range(3)) / 3.0
        gap = max(gap, dev - float(w @ played))
    return gap
