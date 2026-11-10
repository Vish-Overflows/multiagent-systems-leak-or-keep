"""X2 (core): utility types and mixed populations (two-groups network, E1, both rules).

  * SELF, FRIEND(0.2)                       baseline utilities
  * CLAN(0.2, beta), beta in a grid         H3: sharing should rise with beta under SEATS, not under BAR
  * FAIR(a, b) (Fehr-Schmidt)               type-dependent behaviour even under BAR
  * mixed populations                       friendly group vs selfish group; friendly/selfish within groups

Output: results/x2/x2.json
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from leakgame.config import GameConfig  # noqa: E402
from leakgame.game import AgentFormGame, popcount_int  # noqa: E402
from leakgame.solvers import bootstrap_exploitability, enumerate_pure_bne, br_dynamics_from_many_starts  # noqa: E402
from leakgame.tables import build_table  # noqa: E402
from leakgame.utility import CLAN, FAIR, FRIEND, SELF, homogeneous, population_label  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "x2")
BETAS = [round(b, 3) for b in np.arange(0.0, 0.5001, 0.05)]
FAIR_PARAMS = [(0.25, 0.1), (0.5, 0.25), (1.0, 0.5), (2.0, 0.5)]


def solve(tab, pop, rule, n_boot=200):
    g = AgentFormGame(tab, pop, rule)
    res = enumerate_pure_bne(g)
    rec = {"population": population_label(pop), "rule": rule, "n_bne": res.count,
           "cells": [ta.name for ta in g.type_agents]}
    if res.count:
        rates = [float(g.sharing_rate(p)[0]) for p in res.profiles]
        tells = np.stack([popcount_int(g.cells_from_profile(p)) for p in res.profiles])
        rec.update({"sharing_rate_min": min(rates), "sharing_rate_max": max(rates), "sharing_rate_mean": float(np.mean(rates)),
                    "cell_tells_first": tells[0].tolist(), "first_bne": g.describe_profile(res.profiles[0]),
                    "sharing_by_type_first": np.nan_to_num(g.sharing_by_type(res.profiles[0]), nan=-1).tolist(),
                    "welfare_first": float(g.expected_utilities(res.profiles[0]).sum()),
                    "utilities_first": g.expected_utilities(res.profiles[0])[0].tolist()})
        bs = bootstrap_exploitability(g, res.profiles[0], R=n_boot)
        rec["bootstrap"] = {"eps95": bs.eps95, "frac_exact_bne": bs.frac_exact_bne, "R": n_boot}
    brd = br_dynamics_from_many_starts(g, 8)
    rec["br_dynamics"] = {"n_converged": brd["n_converged"], "n_distinct": len(brd["distinct_fixed_points"])}
    return rec


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    cfg = GameConfig(network="two_groups", env="E1")
    tab = build_table(cfg, n_days=2000, seed=0, verbose=False)
    n = cfg.n
    out = {"config": {"network": "two_groups", "env": "E1"}, "baseline": [], "clan": [], "fair": [], "mixed": []}
    for rule in ("SEATS", "BAR"):
        for pop in (homogeneous(SELF, n), homogeneous(FRIEND(0.2), n)):
            out["baseline"].append(solve(tab, pop, rule))
            print("baseline", out["baseline"][-1]["population"], rule, out["baseline"][-1].get("sharing_rate_mean"), flush=True)
    # CLAN: utilities affine in beta -> two V tables per rule
    for rule in ("SEATS", "BAR"):
        g0 = AgentFormGame(tab, homogeneous(CLAN(0.2, 0.0), n), rule)
        g1 = AgentFormGame(tab, homogeneous(CLAN(0.2, 1.0), n), rule)
        for beta in BETAS:
            pop = homogeneous(CLAN(0.2, beta), n)
            g = AgentFormGame.affine(g0, g1, beta, pop=pop)
            res = enumerate_pure_bne(g)
            rates = [float(g.sharing_rate(p)[0]) for p in res.profiles]
            tells = np.stack([popcount_int(g.cells_from_profile(p)) for p in res.profiles]) if res.count else None
            rec = {"rule": rule, "beta": beta, "n_bne": res.count,
                   "sharing_rate_mean": float(np.mean(rates)) if rates else None,
                   "sharing_rate_min": min(rates) if rates else None, "sharing_rate_max": max(rates) if rates else None,
                   "cell_tells_first": tells[0].tolist() if res.count else None,
                   "alpha_eff": (0.2 + beta) / (1 + beta)}
            if res.count:
                bs = bootstrap_exploitability(g, res.profiles[0], R=100)
                rec["bootstrap"] = {"eps95": bs.eps95, "frac_exact_bne": bs.frac_exact_bne, "R": 100}
            out["clan"].append(rec)
            print("clan", rule, beta, rec["sharing_rate_mean"], rec["n_bne"], flush=True)
    # FAIR
    for rule in ("SEATS", "BAR"):
        for a, b in FAIR_PARAMS:
            rec = solve(tab, homogeneous(FAIR(a, b), n), rule)
            rec.update({"a": a, "b": b})
            out["fair"].append(rec)
            print("fair", rule, (a, b), rec.get("sharing_rate_mean"), rec["n_bne"], flush=True)
    # mixed populations
    pops = {
        "friendly group {1,3,5}, selfish group {2,4,6}": [FRIEND(0.2) if i % 2 == 0 else SELF for i in range(n)],
        "first in each group friendly, rest selfish": [FRIEND(0.2) if i < 2 else SELF for i in range(n)],
        "first in each group selfish, rest friendly": [SELF if i < 2 else FRIEND(0.2) for i in range(n)],
        "friendly group {1,3,5}, clannish(0.2,0.3) group {2,4,6}": [FRIEND(0.2) if i % 2 == 0 else CLAN(0.2, 0.3) for i in range(n)],
        "half FAIR(0.5,0.25) {1,3,5}, half FRIEND(0.2) {2,4,6}": [FAIR(0.5, 0.25) if i % 2 == 0 else FRIEND(0.2) for i in range(n)],
    }
    for name, pop in pops.items():
        for rule in ("SEATS", "BAR"):
            rec = solve(tab, pop, rule)
            rec["name"] = name
            out["mixed"].append(rec)
            print("mixed", name, rule, rec.get("sharing_rate_mean"), rec["n_bne"], flush=True)
    out["seconds"] = time.time() - t0
    with open(os.path.join(OUT, "x2.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"X2 done in {out['seconds']:.0f}s")


if __name__ == "__main__":
    main()
