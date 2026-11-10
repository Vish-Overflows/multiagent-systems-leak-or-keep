"""Statistics: percentile bootstrap CIs over seeds, paired sign-flip permutation tests on
common-random-number pairs, and Holm's step-down correction."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def bootstrap_ci(x: np.ndarray, B: int = 10000, level: float = 0.95, seed: int = 0, stat=np.mean):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    s = stat(x[idx], axis=1)
    a = (1 - level) / 2
    return (float(np.quantile(s, a)), float(np.quantile(s, 1 - a)))


def paired_test(x: np.ndarray, y: np.ndarray, B: int = 20000, seed: int = 0) -> Dict:
    """Paired comparison x - y: mean difference, bootstrap CI and two-sided sign-flip permutation p-value."""
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n == 0:
        return {"mean_diff": float("nan"), "ci95": (float("nan"), float("nan")), "p": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    obs = d.mean()
    signs = rng.choice([-1.0, 1.0], size=(B, n))
    null = (signs * d[None, :]).mean(axis=1)
    p = float((np.abs(null) >= abs(obs) - 1e-15).mean())
    lo, hi = bootstrap_ci(d, seed=seed)
    return {"mean_diff": float(obs), "ci95": (lo, hi), "p": max(p, 1.0 / B), "n": int(n)}


def holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def time_to_threshold(curve: np.ndarray, ck: np.ndarray, thr: float = 0.9) -> np.ndarray:
    """First checkpoint at which curve (B, n_ck) >= thr; +inf if never."""
    hit = curve >= thr
    first = np.where(hit.any(axis=1), ck[np.argmax(hit, axis=1)], np.inf)
    return first.astype(float)
