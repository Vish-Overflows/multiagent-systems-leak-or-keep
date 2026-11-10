"""Convenience builders shared by the experiment scripts."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .config import GameConfig
from .game import AgentFormGame
from .learning import LearningSetup
from .solvers import enumerate_pure_bne
from .tables import PayoffTable, build_table
from .utility import Utility

N_DAYS_DEFAULT = 2000


def build_setup(cfg: GameConfig, envs: Sequence[str], rules: Sequence[str], seeds: Sequence[int],
                pop: List[Utility], n_days: int = N_DAYS_DEFAULT, table_seed: int = 0, verbose: bool = True) -> LearningSetup:
    tables: Dict[str, PayoffTable] = {}
    games: Dict[tuple, AgentFormGame] = {}
    targets: Dict[tuple, np.ndarray] = {}
    for e in envs:
        tables[e] = build_table(cfg.with_(env=e), n_days=n_days, seed=table_seed, verbose=verbose)
        for r in rules:
            g = AgentFormGame(tables[e], pop, r)
            games[(e, r)] = g
            res = enumerate_pure_bne(g)
            if res.count == 0:
                raise RuntimeError(f"no pure BNE for {e}/{r}")
            targets[(e, r)] = np.stack([g.cells_from_profile(p) for p in res.profiles])
            if verbose:
                print(f"  BNE target {e}/{r}: {res.count} pure BNE(s); first: {g.describe_profile(res.profiles[0])}")
    return LearningSetup(cfg, list(envs), list(rules), list(seeds), pop, tables, games, targets)
