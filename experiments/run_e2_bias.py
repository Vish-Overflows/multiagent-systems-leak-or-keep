"""How biased is the counterfactual credit of Leak-aware UCB in leak model E2?  (Phase 1.6 of plan.md.)

In E1 the credited sample has exactly the distribution of a sample of the shorter arm.  In E2 the day is identical, but
keeping only the days on which the told question set A_i and the friend's set A_j are disjoint slightly changes the
distribution of A_j and therefore the chance that OTHER tellers' questions land.  The credited sample for the silent arm
has mean E[u_silent | A_i and A_j disjoint]; the unbiased value is E[u_silent].  The difference is the bias.

Configuration (two friend groups): candidate 1 (index 0) chooses between "tell candidate 5 (index 4)" and silence, while
candidate 3 (index 2) tells candidate 5 -- the case where candidate 5 has a second teller, i.e. the worst case for the bias.
Control: candidate 3 silent (candidate 5 has no other teller); there the bias must be exactly zero.
Utility of candidate 1: x_1 + alpha (x_3 + x_5) with alpha = 0.2 (her friends are 3 and 5); the message cost is refunded
by the learner, so it does not enter.

Prediction written before the run: |bias| is much smaller than the smallest payoff gap the learners must detect (0.36
percentage points); refuted if it exceeds half of that gap for any (type, rule).

Output: results/e2_bias/e2_bias.json
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from leakgame.config import GameConfig, type_of  # noqa: E402
from leakgame.env import draw_days, hire_bar, hire_seats, leak_counts_batch, scores  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "e2_bias")
ALPHA = 0.2
SMALLEST_GAP_PP = 0.36           # smallest best-vs-second-best payoff gap at equilibrium (E1, fixed seats), from x3_summary.json


def run(cfg, n_days, other_teller, seed=0, chunk=250_000):
    rng = np.random.default_rng(seed)
    n = cfg.n
    acc = {rule: {th: [0.0, 0.0, 0, 0] for th in range(3)} for rule in ("SEATS", "BAR")}   # sum u, sum u^2, n, n_cond (all days)
    cond = {rule: {th: [0.0, 0.0, 0] for th in range(3)} for rule in ("SEATS", "BAR")}
    done = 0
    while done < n_days:
        B = min(chunk, n_days - done)
        d = draw_days(cfg, B, rng)
        S_silent = np.zeros((B, n, n), dtype=bool)
        if other_teller:
            S_silent[:, 2, 4] = True
        S_tell = S_silent.copy()
        S_tell[:, 0, 4] = True
        k_s, _ = leak_counts_batch(cfg, d, S_silent)                       # the silent arm: what the learner wants to know
        _, kij = leak_counts_batch(cfg, d, S_tell)
        credited = kij[:, 0, 4] == 0                                       # the tell did not land: the day is the silent day
        Y = scores(cfg, d, k_s)
        theta0 = type_of(d.skills[:, 0])
        for rule in ("SEATS", "BAR"):
            x = (hire_seats(Y, cfg.K) if rule == "SEATS" else hire_bar(Y, cfg.tau)).astype(float)
            u = x[:, 0] + ALPHA * (x[:, 2] + x[:, 4])
            for th in range(3):
                m = theta0 == th
                a = acc[rule][th]; a[0] += u[m].sum(); a[1] += (u[m] ** 2).sum(); a[2] += m.sum()
                c = cond[rule][th]; mc = m & credited
                c[0] += u[mc].sum(); c[1] += (u[mc] ** 2).sum(); c[2] += mc.sum()
        done += B
    out = {}
    for rule in ("SEATS", "BAR"):
        for th, tn in enumerate(("weak", "average", "strong")):
            a, c = acc[rule][th], cond[rule][th]
            mean_all, mean_cond = a[0] / a[2], c[0] / c[2]
            sd = np.sqrt(a[1] / a[2] - mean_all ** 2)
            bias = mean_cond - mean_all
            se = sd * np.sqrt(1.0 / c[2] - 1.0 / a[2])                      # conservative: the credited days are a subset
            out[f"{rule}|{tn}"] = {"bias_pp": 100 * bias, "se_pp": 100 * se, "share_credited": c[2] / a[2], "mean_u_silent_pp": 100 * mean_all}
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = GameConfig(network="two_groups", env="E2")
    t0 = time.time()
    res = {"n_days": 4_000_000, "smallest_gap_pp": SMALLEST_GAP_PP,
           "second_teller": run(cfg, 4_000_000, other_teller=True), "no_other_teller": run(cfg, 4_000_000, other_teller=False, seed=1)}
    res["max_abs_bias_pp"] = max(abs(v["bias_pp"]) for v in res["second_teller"].values())
    res["max_abs_bias_over_gap"] = res["max_abs_bias_pp"] / SMALLEST_GAP_PP
    res["prediction_holds"] = bool(res["max_abs_bias_pp"] < 0.5 * SMALLEST_GAP_PP)
    res["seconds"] = time.time() - t0
    print(f"{'setting':22s} {'type':8s} bias (pp)   se (pp)  credited share")
    for label in ("second_teller", "no_other_teller"):
        for k, v in res[label].items():
            rule, tn = k.split("|")
            print(f"{label:16s} {rule:5s} {tn:8s} {v['bias_pp']:+8.4f}  {v['se_pp']:7.4f}   {100 * v['share_credited']:.1f}%")
    print(f"largest |bias| with a second teller: {res['max_abs_bias_pp']:.4f} pp = {100 * res['max_abs_bias_over_gap']:.1f}% of the smallest payoff gap "
          f"({SMALLEST_GAP_PP} pp); prediction (< half the gap) holds: {res['prediction_holds']}")
    with open(os.path.join(OUT, "e2_bias.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
