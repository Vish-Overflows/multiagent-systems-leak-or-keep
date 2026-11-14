"""X3 (core): six learners x hiring rule x environment, T = 2e5 rounds, 30 seeds (1,080 runs).

All 180 runs of one learner (30 seeds x 3 environments x 2 rules) are simulated as one vectorised
batch; runs sharing a seed share the environment draws (common random numbers).  The six learners
run in parallel processes.  Output: results/x3/x3_<learner>.npz
"""
import os
import sys
import time
from multiprocessing import get_context

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from leakgame.config import GameConfig  # noqa: E402
from leakgame.learning import run_learning  # noqa: E402
from leakgame.pipeline import build_setup  # noqa: E402
from leakgame.utility import FRIEND, homogeneous  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "x3")
LEARNERS = ["eps-greedy", "UCB1", "Thompson", "EXP3", "model-based", "model-ucb"]
T = 200_000
SEEDS = list(range(30))
ENVS = ["E0", "E1", "E2"]
RULES = ["SEATS", "BAR"]


def task(name):
    cfg = GameConfig(network="two_groups")
    setup = build_setup(cfg, ENVS, RULES, SEEDS, homogeneous(FRIEND(0.2), cfg.n), verbose=False)
    out = run_learning(name, setup, T=T, ckpt_every=1000, verbose=True)
    np.savez_compressed(os.path.join(OUT, f"x3_{name}.npz"), **{k: v for k, v in out.items()})
    return name, out["seconds"]


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    names = sys.argv[1:] or LEARNERS
    workers = int(os.environ.get("X3_WORKERS", len(names)))             # one learner needs ~1.5 GB while tables build
    with get_context("spawn").Pool(processes=workers) as pool:
        for name, secs in pool.imap_unordered(task, names):
            print(f"finished {name} in {secs:.0f}s", flush=True)
    print(f"X3 done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
