"""Payoff tables P(j in H | theta, S) estimated by stratified Monte Carlo.

For every type profile theta (3^n of them, uniform prior because types are
terciles of i.i.d. skills) and every sharing matrix S in a list, we simulate
`n_days` days.  Within a type profile the *same* days are used for every S and
for both hiring rules (common random numbers).  Skills are drawn from the
truncated normal of each candidate's tercile.

Stored statistics (rule axis: 0 = SEATS, 1 = BAR):
  Px        (T, M, 2, n)      P(x_j = 1 | theta, S)
  Pxx       (T, M, 2, n, n)   P(x_i = 1, x_j = 1 | theta, S)   (needed for FAIR utilities)
  V         (T, M, 2)         E[ sum_{j in H} (s_j - tau_K) ]   (company value)
  Ek        (T, M, n)         E[k_j]                             (leak counts, Prop. 3 check)
  Px_blocks (T, M, 2, n, nblk) block means of x over nblk consecutive blocks of days
                              (for the block bootstrap of exploitability).
  V_blocks  (T, M, 2, nblk)    block means of the company value (bootstrap CI of Lambda).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri

from .config import GameConfig
from .env import company_value, draw_days, hire_bar, hire_seats, leak_counts_table, scores

TABLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "tables")


def type_profiles(n: int) -> np.ndarray:
    """(3^n, n) array; profile t has theta_i = (t // 3^i) % 3."""
    T = 3 ** n
    t = np.arange(T)
    return np.stack([(t // 3 ** i) % 3 for i in range(n)], axis=1).astype(np.int8)


def profile_index(theta: np.ndarray) -> np.ndarray:
    """Inverse of type_profiles: theta (..., n) -> t."""
    n = theta.shape[-1]
    pw = 3 ** np.arange(n)
    return (theta.astype(np.int64) * pw).sum(axis=-1)


def all_matrices(cfg: GameConfig) -> np.ndarray:
    """All 2^f sharing matrices; matrix index = bitmask over cfg.free_entries."""
    fe = cfg.free_entries
    M = 1 << len(fe)
    S = np.zeros((M, cfg.n, cfg.n), dtype=np.int8)
    for idx in range(M):
        for b, (i, j) in enumerate(fe):
            if (idx >> b) & 1:
                S[idx, i, j] = 1
    return S


def matrix_index(cfg: GameConfig, S: np.ndarray) -> int:
    idx = 0
    for b, (i, j) in enumerate(cfg.free_entries):
        if S[i, j]:
            idx |= 1 << b
    return idx


def skills_for_profile(theta: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Truncated-normal skills from uniforms U (N, n) for type profile theta (n,)."""
    lo = theta / 3.0
    return ndtri(lo + U * (1.0 / 3.0))


@dataclass
class PayoffTable:
    cfg: GameConfig
    S_list: np.ndarray
    Px: np.ndarray
    Pxx: np.ndarray
    V: np.ndarray
    Ek: np.ndarray
    Px_blocks: np.ndarray
    n_days: int
    seed: int
    build_seconds: float = 0.0
    V_blocks: np.ndarray | None = None

    @property
    def n_profiles(self) -> int:
        return self.Px.shape[0]

    @property
    def M(self) -> int:
        return self.Px.shape[1]

    def resample_blocks(self, rng: np.random.Generator) -> np.ndarray:
        """Block-bootstrap replicate of Px: resample blocks of days with replacement."""
        nblk = self.Px_blocks.shape[-1]
        pick = rng.integers(0, nblk, size=nblk)
        return self.Px_blocks[..., pick].mean(axis=-1)

    def resample_V_blocks(self, rng: np.random.Generator) -> np.ndarray:
        nblk = self.V_blocks.shape[-1]
        pick = rng.integers(0, nblk, size=nblk)
        return self.V_blocks[..., pick].mean(axis=-1)


def _cache_key(cfg: GameConfig, S_list: np.ndarray, n_days: int, seed: int, n_blocks: int) -> str:
    payload = {
        "cfg": {k: (v if not isinstance(v, tuple) else [list(r) for r in v]) for k, v in cfg.__dict__.items()},
        "S": hashlib.md5(np.ascontiguousarray(S_list).tobytes()).hexdigest(),
        "n_days": n_days, "seed": seed, "n_blocks": n_blocks, "v": 4,
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def build_table(cfg: GameConfig, S_list: np.ndarray | None = None, n_days: int = 2000, seed: int = 0,
                n_blocks: int = 20, cache_dir: str | None = TABLE_DIR, verbose: bool = True,
                chunk_days: int = 24000) -> PayoffTable:
    if S_list is None:
        S_list = all_matrices(cfg)
    S_list = np.asarray(S_list, dtype=np.int8)
    key = _cache_key(cfg, S_list, n_days, seed, n_blocks)
    path = os.path.join(cache_dir, f"table_{cfg.network}_{cfg.env}_n{cfg.n}_Q{cfg.Q}_{key}.npz") if cache_dir else None
    if path and os.path.exists(path):
        z = np.load(path)
        return PayoffTable(cfg, S_list, z["Px"], z["Pxx"], z["V"], z["Ek"], z["Px_blocks"], n_days, seed,
                           float(z["build_seconds"]), z["V_blocks"])

    t0 = time.time()
    n, K = cfg.n, cfg.K
    profiles = type_profiles(n)
    T, M = len(profiles), S_list.shape[0]
    assert n_days % n_blocks == 0
    rng = np.random.default_rng(seed)

    Px = np.zeros((T, M, 2, n), dtype=np.float32)
    Pxx = np.zeros((T, M, 2, n, n), dtype=np.float32)
    V = np.zeros((T, M, 2), dtype=np.float32)
    Ek = np.zeros((T, M, n), dtype=np.float32)
    Px_blocks = np.zeros((T, M, 2, n, n_blocks), dtype=np.float32)
    V_blocks = np.zeros((T, M, 2, n_blocks), dtype=np.float32)

    P = max(1, chunk_days // n_days)         # profiles per chunk
    for start in range(0, T, P):
        idx = np.arange(start, min(T, start + P))
        p = len(idx)
        U = rng.random((p, n_days, n))
        skills = np.stack([skills_for_profile(profiles[t], U[a]) for a, t in enumerate(idx)]).reshape(p * n_days, n)
        d = draw_days(cfg, p * n_days, rng, skills=skills)
        k = leak_counts_table(cfg, d, S_list)                        # (B, M, n)
        Y = scores(cfg, d, k)
        x = np.stack([hire_seats(Y, K), hire_bar(Y, cfg.tau)], axis=2)   # (B, M, 2, n) bool
        v = np.stack([company_value(cfg, d, x[:, :, 0]), company_value(cfg, d, x[:, :, 1])], axis=2)  # (B, M, 2)

        xf = x.reshape(p, n_days, M, 2, n).astype(np.float32)
        Px[idx] = xf.mean(axis=1)
        # pairwise joint hiring: batched matmul over the day axis
        xt = np.moveaxis(xf, 1, -1)                                  # (p, M, 2, n, N)
        Pxx[idx] = (xt @ np.swapaxes(xt, -1, -2)) / n_days
        V[idx] = v.reshape(p, n_days, M, 2).mean(axis=1)
        Ek[idx] = k.reshape(p, n_days, M, n).mean(axis=1)
        Px_blocks[idx] = np.moveaxis(xf.reshape(p, n_blocks, n_days // n_blocks, M, 2, n).mean(axis=2), 1, -1)
        V_blocks[idx] = np.moveaxis(v.reshape(p, n_blocks, n_days // n_blocks, M, 2).mean(axis=2), 1, -1)
        if verbose and (start // P) % 10 == 0:
            print(f"  [table {cfg.network}/{cfg.env} Q={cfg.Q}] profiles {start + p}/{T}  "
                  f"{time.time() - t0:6.1f}s", flush=True)

    secs = time.time() - t0
    tab = PayoffTable(cfg, S_list, Px, Pxx, V, Ek, Px_blocks, n_days, seed, secs, V_blocks)
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(path, Px=Px, Pxx=Pxx, V=V, Ek=Ek, Px_blocks=Px_blocks, V_blocks=V_blocks,
                            build_seconds=secs)
    if verbose:
        print(f"  built table {cfg.network}/{cfg.env} Q={cfg.Q}: T={T} profiles x M={M} matrices x "
              f"{n_days} days in {secs:.1f}s", flush=True)
    return tab
