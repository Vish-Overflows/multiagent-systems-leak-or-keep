"""X1 (core): hiring rule x alpha in [0, 0.5] x network x environment, solved at Level 1.

For every (network, env) a payoff table is built once; for every rule and alpha on the grid the pure
BNEs are enumerated exhaustively (utilities are affine in alpha, so only two V tables are needed per
rule).  Each equilibrium gets best-response-dynamics confirmation and a block-bootstrap eps.  For the
two-groups network we also run replicator dynamics and solve the CCE linear programs (max / min
expected sharing) to obtain the range of sharing rates consistent with coarse correlated equilibrium.
The complete network (15 free entries, 2^15 matrices) is analysed by exhaustive *local* deviation
checks of the two candidate profiles (nobody shares / everybody shares) at n = 6, plus a full solve
at n = 4.

Output: results/x1/x1_<network>_<env>.json, results/x1/x1_complete_local.json, results/x1/x1_summary.json
"""
import json
import os
import sys
import time
from multiprocessing import get_context

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from leakgame.analysis import complete_local_sweep, sweep_alpha  # noqa: E402
from leakgame.config import GameConfig  # noqa: E402
from leakgame.game import AgentFormGame  # noqa: E402
from leakgame.solvers import cce_lp  # noqa: E402
from leakgame.tables import build_table  # noqa: E402
from leakgame.utility import FRIEND, homogeneous  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "x1")
ALPHAS = [round(a, 3) for a in np.arange(0.0, 0.5001, 0.025)]
ALPHAS_CCE = [round(a, 3) for a in np.arange(0.0, 0.5001, 0.05)]
NETWORKS = ["two_groups", "pairs", "star"]
ENVS = ["E0", "E1", "E2"]
N_DAYS = 2000


def task(args):
    network, env = args
    if os.path.exists(os.path.join(OUT, f"x1_{network}_{env}.json")):
        print(f"skip {network}/{env} (already done)", flush=True)
        return network, env
    t0 = time.time()
    cfg = GameConfig(network=network, env=env)
    tab = build_table(cfg, n_days=N_DAYS, seed=0, verbose=False)
    res = {"network": network, "env": env, "n_days": N_DAYS, "table_seconds": tab.build_seconds, "rules": {}}
    for rule in ("SEATS", "BAR"):
        res["rules"][rule] = sweep_alpha(tab, rule, ALPHAS, n_boot=200 if network == "two_groups" else 100,
                                         do_replicator=(network == "two_groups"), verbose=False)
    # CCE polytope range of sharing (E1 only; all networks small enough except two_groups -> coarser grid)
    if env == "E1":
        res["cce"] = {}
        grid = ALPHAS_CCE if network == "two_groups" else ALPHAS
        for rule in ("SEATS", "BAR"):
            rows = []
            for al in grid:
                g = AgentFormGame(tab, homogeneous(FRIEND(al), cfg.n), rule)
                cc = cce_lp(g, objectives=("max_sharing", "min_sharing"))
                rows.append({"alpha": al, "cce_share_max": cc[0].sharing_rate, "cce_share_min": cc[1].sharing_rate,
                             "support_max": cc[0].support, "support_min": cc[1].support,
                             "lp_seconds": cc[0].seconds + cc[1].seconds})
            res["cce"][rule] = rows
    res["seconds"] = time.time() - t0
    with open(os.path.join(OUT, f"x1_{network}_{env}.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"done {network}/{env} in {res['seconds']:.0f}s", flush=True)
    return network, env


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    tasks = [(net, env) for net in NETWORKS for env in ENVS]
    with get_context("spawn").Pool(processes=min(9, len(tasks))) as pool:
        for _ in pool.imap_unordered(task, tasks):
            pass
    # complete network: local analysis at n = 6 for the alpha grid
    comp = complete_local_sweep(GameConfig(network="complete", env="E1"), ALPHAS, n_days=1000)
    with open(os.path.join(OUT, "x1_complete_local.json"), "w") as f:
        json.dump(comp, f, indent=1, default=float)
    # summary: sharing rate curves
    summary = {"alphas": ALPHAS, "curves": {}}
    for net in NETWORKS:
        for env in ENVS:
            with open(os.path.join(OUT, f"x1_{net}_{env}.json")) as f:
                r = json.load(f)
            for rule in ("SEATS", "BAR"):
                recs = r["rules"][rule]["records"]
                summary["curves"][f"{net}/{env}/{rule}"] = {
                    "share_mean": [x.get("sharing_rate_mean") for x in recs],
                    "share_min": [x.get("sharing_rate_min") for x in recs],
                    "share_max": [x.get("sharing_rate_max") for x in recs],
                    "n_bne": [x["n_bne"] for x in recs],
                    "alpha_star": r["rules"][rule]["alpha_star"],
                    "loss_gain": r["rules"][rule]["loss_gain"],
                }
    summary["complete_local"] = comp
    summary["seconds"] = time.time() - t0
    with open(os.path.join(OUT, "x1_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(f"X1 done in {summary['seconds']:.0f}s")


if __name__ == "__main__":
    main()
