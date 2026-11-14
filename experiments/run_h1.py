"""H1 check that does not use the own-loss ell: how often does each type hold the last seat when nobody shares?

H1 says alpha* is highest for the types most likely to be pivotal.  The report measured "pivotal" with the own
loss ell, but alpha* = (ell + c)/G, so with G roughly constant the ordering of alpha* follows ell by construction.
Here pivotality is measured on fresh simulated days with no sharing at all (leakgame.analysis.pivotality_by_type).

Prediction written before the run: P(rank = K | type) increases weak < average < strong, the order of alpha*
(0.10 < 0.32 < 0.38); H1 is refuted if the order differs.

Output: results/h1/pivotality.json
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leakgame.analysis import pivotality_by_type  # noqa: E402
from leakgame.config import GameConfig  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "h1")


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    res = pivotality_by_type(GameConfig(), n_days=2_000_000, seed=0)
    res["seconds"] = time.time() - t0
    for key in ("P_rank_K", "P_rank_K1", "P_hired"):
        print(f"{key:10s}", "  ".join(f"{t}: {100 * p:5.2f}% (se {100 * s:.2f})" for t, p, s in zip(res["types"], res[key], res["se_" + key[2:]])))
    order = sorted(range(3), key=lambda i: res["P_rank_K"][i])
    res["order_matches_alpha_star"] = order == [0, 1, 2]
    print("order of P(rank = K):", [res["types"][i] for i in order], "-> matches the alpha* order weak < average < strong:", res["order_matches_alpha_star"])
    with open(os.path.join(OUT, "pivotality.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
