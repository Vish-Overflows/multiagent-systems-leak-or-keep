"""Agent-form representation of the Bayesian stage game built from a payoff table.

Player i's action is a row of S: a subset of L(i) (later friends), encoded as a
bitmask a in {0, ..., 2^{|L(i)|}-1}.  A pure strategy is a map from the three
types to actions, encoded as sigma_i = a_weak + K_i a_avg + K_i^2 a_strong.
The central object is

    V_i[sigma_{-i}, theta_i, a] = E_{theta_{-i}}[ u_i(theta, S(a, sigma_{-i}(theta_{-i}))) ],

the interim expected utility of type theta_i of player i from action a when
the others play the pure strategy profile sigma_{-i}.  Everything else (best
responses, exploitability, pure BNE enumeration, CCE constraints, learning
metrics) is a lookup into V.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import GameConfig
from .tables import PayoffTable, type_profiles
from .utility import Utility, weights

RULE_IDX = {"SEATS": 0, "BAR": 1}


def popcount_int(a: np.ndarray) -> np.ndarray:
    return np.bitwise_count(np.asarray(a, dtype=np.uint64)).astype(np.int64)


@dataclass
class TypeAgent:
    player: int
    theta: int
    n_actions: int

    @property
    def name(self) -> str:
        return f"c{self.player + 1}/{['weak', 'avg', 'strong'][self.theta]}"


class AgentFormGame:
    def __init__(self, table: PayoffTable, pop: Sequence[Utility], rule: str, Px: np.ndarray | None = None,
                 build_V: bool = True):
        self.table = table
        self.cfg: GameConfig = table.cfg
        self.pop = list(pop)
        self.rule = rule
        self.r = RULE_IDX[rule]
        cfg = self.cfg
        n = cfg.n
        self.n = n
        self.profiles = type_profiles(n)                       # (T, n)
        self.T = len(self.profiles)
        self.L = [cfg.later_friends(i) for i in range(n)]
        self.K = [1 << len(self.L[i]) for i in range(n)]
        self.n_strat = [k ** 3 for k in self.K]
        self.active = [i for i in range(n) if self.K[i] > 1]
        fe = cfg.free_entries
        bitpos = {e: b for b, e in enumerate(fe)}
        # rowbits[i][a] = matrix-index contribution of action a of player i
        self.rowbits = []
        for i in range(n):
            rb = np.zeros(self.K[i], dtype=np.int64)
            for a in range(self.K[i]):
                for t, j in enumerate(self.L[i]):
                    if (a >> t) & 1:
                        rb[a] |= 1 << bitpos[(i, j)]
            self.rowbits.append(rb)
        # act_of_S[i][S] = action of player i encoded in matrix index S
        M = table.M
        self.M = M
        self.act_of_S = []
        for i in range(n):
            acts = np.zeros(M, dtype=np.int64)
            for t, j in enumerate(self.L[i]):
                acts |= ((np.arange(M) >> bitpos[(i, j)]) & 1) << t
            self.act_of_S.append(acts)
        self.n_told_of_S = np.stack([popcount_int(self.act_of_S[i]) for i in range(n)], axis=1)  # (M, n)
        self.type_agents: List[TypeAgent] = [TypeAgent(i, th, self.K[i]) for i in self.active for th in range(3)]
        self.EU = self._expected_utilities(Px if Px is not None else table.Px)   # list of (T, M)
        self.V: List[np.ndarray | None] = [None] * n
        self.others: List[List[int]] = [[j for j in self.active if j != i] for i in range(n)]
        if build_V:
            for i in self.active:
                self.V[i] = self._build_V(i)

    # ------------------------------------------------------------------ tables
    def _expected_utilities(self, Px: np.ndarray) -> List[np.ndarray]:
        """EU_i[t, S] for every player (uses this rule's hiring probabilities)."""
        cfg = self.cfg
        Px_r = Px[:, :, self.r, :].astype(np.float64)              # (T, M, n)
        Pxx_r = self.table.Pxx[:, :, self.r, :, :].astype(np.float64)
        out = []
        for i in range(self.n):
            w_lin, w_pair = weights(self.pop[i], cfg, i)
            eu = Px_r @ w_lin
            if np.any(w_pair != 0):
                eu = eu + Pxx_r[:, :, i, :] @ w_pair
            eu = eu - cfg.c * self.n_told_of_S[None, :, i]
            out.append(eu)
        return out

    def _decode_action(self, i: int, sigma: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Action of strategy sigma (array) at type theta (array), broadcasting."""
        K = self.K[i]
        return (sigma // (K ** theta)) % K

    def others_radix(self, i: int) -> List[int]:
        return [self.n_strat[j] for j in self.others[i]]

    def others_index(self, i: int, sig: np.ndarray) -> np.ndarray:
        """Mixed-radix index of sigma_{-i} from a (N, n) array of strategy indices."""
        idx = np.zeros(sig.shape[0], dtype=np.int64)
        mult = 1
        for j in self.others[i]:
            idx += sig[:, j] * mult
            mult *= self.n_strat[j]
        return idx

    def _decode_others(self, i: int, idx: np.ndarray) -> Dict[int, np.ndarray]:
        out = {}
        rem = idx.copy()
        for j in self.others[i]:
            out[j] = rem % self.n_strat[j]
            rem //= self.n_strat[j]
        return out

    def _build_V(self, i: int, EU_i: np.ndarray | None = None, chunk: int = 2048) -> np.ndarray:
        EU_i = self.EU[i] if EU_i is None else EU_i
        n_others = int(np.prod(self.others_radix(i))) if self.others[i] else 1
        K = self.K[i]
        V = np.zeros((n_others, 3, K), dtype=np.float64)
        theta_i = self.profiles[:, i]
        groups = [np.where(theta_i == th)[0] for th in range(3)]
        for start in range(0, n_others, chunk):
            idx = np.arange(start, min(n_others, start + chunk))
            sig = self._decode_others(i, idx)
            S_oth = np.zeros((len(idx), self.T), dtype=np.int64)
            for j, sj in sig.items():
                a_j = self._decode_action(j, sj[:, None], self.profiles[None, :, j])
                S_oth += self.rowbits[j][a_j]
            for a in range(K):
                vals = EU_i[np.arange(self.T)[None, :], S_oth + self.rowbits[i][a]]   # (C, T)
                for th in range(3):
                    V[idx, th, a] = vals[:, groups[th]].mean(axis=1)
        return V

    # --------------------------------------------------------------- profiles
    def all_profiles(self) -> np.ndarray:
        """(N, n) array of all pure strategy profiles (trivial players have index 0)."""
        grids = np.meshgrid(*[np.arange(self.n_strat[i]) for i in range(self.n)], indexing="ij")
        return np.stack([g.ravel() for g in grids], axis=1).astype(np.int64)

    @property
    def n_profiles(self) -> int:
        return int(np.prod(self.n_strat))

    def strategy_from_actions(self, i: int, actions: Sequence[int]) -> int:
        K = self.K[i]
        return int(actions[0] + K * actions[1] + K * K * actions[2])

    def actions_from_strategy(self, i: int, sigma: int) -> np.ndarray:
        return self._decode_action(i, np.int64(sigma), np.arange(3))

    def profile_from_cells(self, cells: Sequence[int]) -> np.ndarray:
        """Profile (n,) from the list of type-agent actions (ordered as self.type_agents)."""
        sig = np.zeros(self.n, dtype=np.int64)
        for c, ta in enumerate(self.type_agents):
            sig[ta.player] += cells[c] * (self.K[ta.player] ** ta.theta)
        return sig

    def cells_from_profile(self, sig: np.ndarray) -> np.ndarray:
        return np.array([self.actions_from_strategy(ta.player, sig[ta.player])[ta.theta] for ta in self.type_agents])

    # --------------------------------------------------------------- payoffs
    def interim_values(self, i: int, sig: np.ndarray) -> np.ndarray:
        """V_i[sigma_{-i}, :, :] for an (N, n) array of profiles -> (N, 3, K_i)."""
        return self.V[i][self.others_index(i, sig)]

    def deviation_gains(self, sig: np.ndarray) -> np.ndarray:
        """(N, n_type_agents): best deviation gain of every type-agent at each profile."""
        sig = np.atleast_2d(sig)
        gains = np.zeros((sig.shape[0], len(self.type_agents)))
        for c, ta in enumerate(self.type_agents):
            i, th = ta.player, ta.theta
            Vi = self.interim_values(i, sig)[:, th, :]                      # (N, K)
            played = self._decode_action(i, sig[:, i], th)
            gains[:, c] = Vi.max(axis=1) - Vi[np.arange(sig.shape[0]), played]
        return gains

    def exploitability(self, sig: np.ndarray) -> np.ndarray:
        return self.deviation_gains(sig).max(axis=1)

    def expected_utilities(self, sig: np.ndarray) -> np.ndarray:
        """Ex-ante U_i(sigma) for all players -> (N, n)."""
        sig = np.atleast_2d(sig)
        U = np.zeros((sig.shape[0], self.n))
        for i in range(self.n):
            if self.K[i] == 1:
                # trivial player: evaluate directly from EU over type profiles
                S_idx = np.zeros((sig.shape[0], self.T), dtype=np.int64)
                for j in self.active:
                    a_j = self._decode_action(j, sig[:, j][:, None], self.profiles[None, :, j])
                    S_idx += self.rowbits[j][a_j]
                U[:, i] = self.EU[i][np.arange(self.T)[None, :], S_idx].mean(axis=1)
            else:
                Vi = self.interim_values(i, sig)                                # (N, 3, K)
                for th in range(3):
                    played = self._decode_action(i, sig[:, i], th)
                    U[:, i] += Vi[np.arange(sig.shape[0]), th, played] / 3.0
        return U

    def sharing_counts(self, sig: np.ndarray) -> np.ndarray:
        """Expected number of tells per profile (expectation over types) -> (N,)."""
        sig = np.atleast_2d(sig)
        tot = np.zeros(sig.shape[0])
        for i in self.active:
            for th in range(3):
                tot += popcount_int(self._decode_action(i, sig[:, i], th)) / 3.0
        return tot

    def sharing_rate(self, sig: np.ndarray) -> np.ndarray:
        return self.sharing_counts(sig) / max(1, self.cfg.n_free)

    def describe_profile(self, sig: np.ndarray) -> Dict[str, List[int]]:
        """{'c1/weak': [3, 5], ...}: whom each type of each candidate tells (1-indexed)."""
        out = {}
        for ta in self.type_agents:
            a = int(self.actions_from_strategy(ta.player, int(sig[ta.player]))[ta.theta])
            out[ta.name] = [self.L[ta.player][t] + 1 for t in range(len(self.L[ta.player])) if (a >> t) & 1]
        return out

    def sharing_by_type(self, sig: np.ndarray) -> np.ndarray:
        """(n, 3) expected number of tells by candidate and type (nan for candidates with no later friends)."""
        out = np.full((self.n, 3), np.nan)
        for i in self.active:
            for th in range(3):
                out[i, th] = popcount_int(self._decode_action(i, np.int64(sig[i]), th))
        return out

    # ------------------------------------------------------- affine utilities
    @classmethod
    def affine(cls, g0: "AgentFormGame", g1: "AgentFormGame", lam: float, pop=None) -> "AgentFormGame":
        """Game whose utilities are (1-lam) u^0 + lam u^1 (utilities are affine in alpha and beta, so
        FRIEND(alpha) = affine(SELF, FRIEND(1), alpha)).  Reuses the V tables: no re-simulation."""
        g = cls.__new__(cls)
        g.__dict__.update(g0.__dict__)
        g.pop = pop if pop is not None else g0.pop
        g.EU = [(1 - lam) * a + lam * b for a, b in zip(g0.EU, g1.EU)]
        g.V = [None if a is None else (1 - lam) * a + lam * b for a, b in zip(g0.V, g1.V)]
        return g

    # ------------------------------------------------------- bootstrap support
    def with_Px(self, Px: np.ndarray, build_V: bool = False) -> "AgentFormGame":
        """Same game on a resampled hiring-probability table (block bootstrap)."""
        return AgentFormGame(self.table, self.pop, self.rule, Px=Px, build_V=build_V)

    def profile_interim_values(self, sig: np.ndarray) -> List[np.ndarray]:
        """For ONE profile: V_i[sigma_{-i}, :, :] for each active player computed directly from EU
        (does not need the full V tables; used for cheap bootstrap replicates)."""
        sig = np.asarray(sig).ravel()
        out = []
        S_base = {}
        for i in self.active:
            S_oth = np.zeros(self.T, dtype=np.int64)
            for j in self.active:
                if j == i:
                    continue
                a_j = self._decode_action(j, np.int64(sig[j]), self.profiles[:, j])
                S_oth += self.rowbits[j][a_j]
            Vi = np.zeros((3, self.K[i]))
            theta_i = self.profiles[:, i]
            for a in range(self.K[i]):
                vals = self.EU[i][np.arange(self.T), S_oth + self.rowbits[i][a]]
                for th in range(3):
                    Vi[th, a] = vals[theta_i == th].mean()
            out.append(Vi)
        return out

    def profile_exploitability_direct(self, sig: np.ndarray) -> Tuple[float, np.ndarray]:
        """Exploitability of one profile from EU directly; returns (eps, per type-agent gains)."""
        Vs = self.profile_interim_values(sig)
        gains = []
        for (i, Vi) in zip(self.active, Vs):
            acts = self.actions_from_strategy(i, int(sig[i]))
            for th in range(3):
                gains.append(Vi[th].max() - Vi[th, acts[th]])
        gains = np.array(gains)
        return float(gains.max()), gains
