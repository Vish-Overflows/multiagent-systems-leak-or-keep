"""CCE linear programs for leak models E0 and E2 (run_x1.py solves them for E1 only).

For every network in (two_groups, pairs, star), env in (E0, E2), rule and alpha on the same grids as the E1
programs, solve two linear programs over the whole CCE polytope: maximise and minimise the expected share of
tells.  If the two optima coincide, every CCE (hence every mixed equilibrium and every limit of no-regret
learning) shares the same amount.

Two consistency checks run with every row, because every pure BNE is a CCE:
  * every pure equilibrium's sharing rate lies inside the [min, max] interval of the CCE programs;
  * the pure equilibria enumerated from the rebuilt table equal the ones stored by run_x1.py for the same
    (network, env, rule, alpha) -- same count, same sharing rates -- so this run uses the same table.

The two-groups programs are big (a dense 144 x 262,144 constraint matrix) and slow, so rows are saved as they
finish and an interrupted run resumes where it stopped.

Usage:  python experiments/run_cce.py [n_workers] [network:env ...]
        n_workers defaults to 1 (a two-groups LP needs about 4 GB); the optional cells (e.g. star:E0 pairs:E2)
        restrict the run, so that several processes can share the work.  A cell is skipped if its result
        file exists, so processes started on disjoint cells do not collide.
Output: results/x1/x1_cce_<network>_<env>.json
"""
import json
import os
import sys
import time
from multiprocessing import get_context

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from leakgame.config import GameConfig  # noqa: E402
from leakgame.game import AgentFormGame  # noqa: E402
from leakgame.solvers import cce_lp, enumerate_pure_bne  # noqa: E402
from leakgame.tables import build_table  # noqa: E402
from leakgame.utility import FRIEND, homogeneous  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "x1")
ALPHAS = [round(a, 3) for a in np.arange(0.0, 0.5001, 0.025)]
ALPHAS_CCE = [round(a, 3) for a in np.arange(0.0, 0.5001, 0.05)]      # two_groups: coarser grid, as for E1
NETWORKS = ["two_groups", "pairs", "star"]
ENVS = ["E0", "E2"]
N_DAYS = 2000
TOL = 1e-7


def path(network, env, partial=False):
    return os.path.join(OUT, f"x1_cce_{network}_{env}" + (".partial" if partial else "") + ".json")


def x1_record(network, env, rule, alpha):
    """The record run_x1.py stored for this (network, env, rule, alpha)."""
    with open(os.path.join(OUT, f"x1_{network}_{env}.json")) as f:
        recs = json.load(f)["rules"][rule]["records"]
    return min(recs, key=lambda r: abs(r["alpha"] - alpha))


def task(args):
    network, env = args
    final, part = path(network, env), path(network, env, partial=True)
    if os.path.exists(final):
        print(f"skip {network}/{env} (already done)", flush=True)
        return network, env
    t0 = time.time()
    cfg = GameConfig(network=network, env=env)
    tab = build_table(cfg, n_days=N_DAYS, seed=0, verbose=False)
    grid = ALPHAS_CCE if network == "two_groups" else ALPHAS
    done = {"SEATS": [], "BAR": []}
    if os.path.exists(part):
        with open(part) as f:
            done = json.load(f)["cce"]
    for rule in ("SEATS", "BAR"):
        for al in grid:
            if any(abs(r["alpha"] - al) < 1e-9 for r in done[rule]):
                continue
            g = AgentFormGame(tab, homogeneous(FRIEND(al), cfg.n), rule)
            cc = cce_lp(g, objectives=("max_sharing", "min_sharing"))
            hi, lo = cc[0].sharing_rate, cc[1].sharing_rate
            res = enumerate_pure_bne(g)
            rates = [float(g.sharing_rate(p)[0]) for p in res.profiles]
            row = {"alpha": al, "cce_share_max": hi, "cce_share_min": lo,
                   "support_max": cc[0].support, "support_min": cc[1].support,
                   "lp_seconds": cc[0].seconds + cc[1].seconds,
                   "lp_status": [cc[0].status, cc[1].status],
                   "n_bne": res.count,
                   "bne_share_min": min(rates) if rates else None, "bne_share_max": max(rates) if rates else None}
            # check 1: every pure BNE is a CCE
            if rates:
                assert lo - TOL <= min(rates) and max(rates) <= hi + TOL, \
                    f"pure BNE outside the CCE interval: {network}/{env}/{rule}/{al}: {rates} vs [{lo}, {hi}]"
            # check 2: same table as the committed X1 run
            ref = x1_record(network, env, rule, al)
            same = ref["n_bne"] == res.count and (
                res.count == 0 or (abs(ref["sharing_rate_min"] - min(rates)) < 1e-9
                                   and abs(ref["sharing_rate_max"] - max(rates)) < 1e-9))
            row["matches_x1_equilibria"] = bool(same)
            assert same, f"rebuilt table disagrees with run_x1.py at {network}/{env}/{rule}/{al}"
            done[rule].append(row)
            with open(part, "w") as f:
                json.dump({"network": network, "env": env, "cce": done}, f, default=float)
            print(f"  {network}/{env}/{rule} alpha={al:.3f}: CCE share [{lo:.3f}, {hi:.3f}]  "
                  f"{res.count} pure BNE in [{min(rates) if rates else float('nan'):.3f}, "
                  f"{max(rates) if rates else float('nan'):.3f}]  ({row['lp_seconds']:.0f}s)", flush=True)
    res = {"network": network, "env": env, "n_days": N_DAYS, "cce": done, "seconds": time.time() - t0}
    with open(final, "w") as f:
        json.dump(res, f, indent=1, default=float)
    os.remove(part)
    print(f"done {network}/{env} in {res['seconds']:.0f}s", flush=True)
    return network, env


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    tasks = [tuple(a.split(":")) for a in sys.argv[2:]] or [(net, env) for net in NETWORKS for env in ENVS]
    with get_context("spawn").Pool(processes=workers) as pool:
        for _ in pool.imap_unordered(task, tasks):
            pass
    print(f"CCE runs done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
