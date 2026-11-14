"""Mixed equilibria beyond replicator dynamics: dominance reduction and support enumeration (X1 grid).

Replicator dynamics cannot find unstable mixed equilibria, so "no mixed equilibrium" needs another argument.
For every game of the X1 grid (network x leak model x rule x alpha, utilities FRIEND(alpha)):

  * all three networks: iterated strict dominance (leakgame.support_enum.dominance_reduction).  If every
    (candidate, type) keeps one action, that profile is the game's ONLY equilibrium, pure or mixed.  It is
    also checked to be an exact pure equilibrium of the table (exploitability 0).
  * pairs network (every agent has two actions): full support enumeration on what dominance leaves
    (leakgame.support_enum.support_enumeration), cross-checked against the exhaustively enumerated pure BNEs.

Two-groups and star games dominance does not solve are only bounded by the CCE linear programs (run_x1.py,
run_cce.py); nothing is claimed about their mixed equilibria beyond that.

Output: results/x1/x1_support.json
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from leakgame.config import GameConfig  # noqa: E402
from leakgame.game import AgentFormGame  # noqa: E402
from leakgame.solvers import enumerate_pure_bne  # noqa: E402
from leakgame.support_enum import dominance_reduction, support_enumeration  # noqa: E402
from leakgame.tables import build_table  # noqa: E402
from leakgame.utility import FRIEND, SELF, homogeneous  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "x1")
ALPHAS = [round(a, 3) for a in np.arange(0.0, 0.5001, 0.025)]
NETWORKS = ["pairs", "star", "two_groups"]
ENVS = ["E0", "E1", "E2"]
N_DAYS = 2000


def record(g, alpha, full_enum):
    allowed, removed = dominance_reduction(g)
    surv = [len(a) for a in allowed]
    solved = all(s == 1 for s in surv)
    rec = {"alpha": alpha, "survivors": surv, "dominance_solvable": solved,
           "residual_profiles": int(np.prod(surv))}
    if solved:
        prof = g.profile_from_cells([next(iter(a)) for a in allowed])
        eps = float(g.exploitability(prof)[0])
        rec["survivor_exploitability"] = eps
        assert eps <= 1e-12, "the dominance survivor must be an exact pure equilibrium of the table"
    if full_enum:
        t0 = time.time()
        r = support_enumeration(g)
        ref = {tuple(g.cells_from_profile(p)) for p in enumerate_pure_bne(g).profiles}
        got = {tuple(int(x) for x in p) for p in r.pure}
        rec["support"] = {"n_free": r.n_free, "n_supports": r.n_supports, "n_supports_k_ge_3": r.n_supports_k_ge_3,
                          "max_k": r.max_k, "complete": r.complete, "n_pure": len(r.pure), "n_mixed": len(r.mixed),
                          "exact_ties": r.exact_ties, "pure_matches_enumeration": got == ref,
                          "mixed": [{"p": m["p"], "k": m["k"],
                                     "sharing_rate": float(np.sum(m["p"]) / 3 / g.cfg.n_free)} for m in r.mixed],
                          "seconds": time.time() - t0}
        assert got == ref, "support enumeration disagrees with exhaustive enumeration of pure equilibria"
    return rec


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    out = {"alphas": ALPHAS, "games": {}}
    for net in NETWORKS:
        for env in ENVS:
            cfg = GameConfig(network=net, env=env)
            tab = build_table(cfg, n_days=N_DAYS, seed=0, verbose=False)
            for rule in ("SEATS", "BAR"):
                g0 = AgentFormGame(tab, homogeneous(SELF, cfg.n), rule)
                g1 = AgentFormGame(tab, homogeneous(FRIEND(1.0), cfg.n), rule)
                rows = []
                for al in ALPHAS:
                    g = AgentFormGame.affine(g0, g1, al, pop=homogeneous(FRIEND(al), cfg.n))
                    rows.append(record(g, al, full_enum=(net == "pairs")))
                out["games"][f"{net}/{env}/{rule}"] = rows
                solved = sum(r["dominance_solvable"] for r in rows)
                extra = ""
                if net == "pairs":
                    extra = (f"; mixed found in {sum(r['support']['n_mixed'] > 0 for r in rows)} games, "
                             f"max free agents {max(r['support']['n_free'] for r in rows)}, "
                             f"all complete: {all(r['support']['complete'] for r in rows)}")
                print(f"{net}/{env}/{rule}: dominance-solvable {solved}/{len(rows)}{extra}  ({time.time() - t0:.0f}s)", flush=True)
    out["seconds"] = time.time() - t0
    with open(os.path.join(OUT, "x1_support.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"support enumeration done in {out['seconds']:.0f}s")


if __name__ == "__main__":
    main()
