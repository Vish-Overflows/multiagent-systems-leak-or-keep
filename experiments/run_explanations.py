"""Tests of four explanations the report gave without testing them (Phase 1.5 of plan.md).

Predictions written before the run (refuted if the condition fails):
  (a) Average types learn slowest because weak types' rewards are much less noisy.
      Test: reward noise (sd of the normalised reward at the equilibrium profile) divided by the payoff gap between the
      best and second-best arm, per decision cell; (noise/gap)^2 is what a bandit has to average away.
      Prediction: (noise/gap)^2 is largest for the average types, and over the 12 cells of E1 its Spearman correlation
      with the final error rate is positive for UCB1 and Thompson sampling.
  (b) UCB1 has high regret because its confidence bonus keeps it exploring.
      Test: share of all pulls on non-equilibrium arms, E1.  Prediction: UCB1 > Thompson and > epsilon-greedy in both rules.
  (c) epsilon-greedy's remaining mistakes are on decisions where the wrong row costs little.
      Test: the error-weighted mean payoff gap of the cells where it ends wrong, divided by the mean gap over all cells.
      Prediction: ratio < 1.
  (d) Model-based UCB helps because it pools information across arms that share a friend.
      Test: relative regret reduction against UCB1 in cells with two later friends (4 arms) vs one later friend (2 arms).
      Prediction: larger in the 4-arm cells in at least 3 of the 4 (E1/E2 x rule) settings.

Outcome (see the log): (a) supported, (b) supported, (c) refuted (the mean gap of epsilon-greedy's mistakes is about 1.0 times the
mean gap over all cells; its low regret comes from pulling wrong arms less often, as (b) measures), (d) supported but modest.

Pull counts: the stored final_counts of the leak-aware learner ("model-based") include CREDITED observations (a tell that did
not land also counts for the arm without it), so they are not pulls and that learner is left out of (b) and (c).  Its settled
profile is unaffected: the profile from the credited counts equals the profile from the greedy means in 100% of runs.

Output: results/explanations/explanations.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from leakgame.config import GameConfig, type_of  # noqa: E402
from leakgame.env import draw_days, simulate_day  # noqa: E402
from leakgame.game import popcount_int  # noqa: E402
from leakgame.pipeline import build_setup  # noqa: E402
from leakgame.utility import FRIEND, homogeneous, realized_utility, reward_range  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "explanations")
X3 = os.path.join(ROOT, "results", "x3")
LEARNERS = ["eps-greedy", "UCB1", "Thompson", "EXP3", "model-based", "model-ucb"]
RULES = ["SEATS", "BAR"]
TYPES = ["weak", "average", "strong"]


def cell_reward_sd(cfg, g, rule, sig, n_days=400_000, seed=0):
    """sd of the normalised reward of every decision cell when everybody plays the profile sig."""
    rng = np.random.default_rng(seed)
    n = cfg.n
    d = draw_days(cfg, n_days, rng)
    theta = type_of(d.skills)
    S = np.zeros((n_days, n, n), dtype=bool)
    n_told = np.zeros((n_days, n))
    for i in g.active:
        acts = g.actions_from_strategy(i, int(sig[i]))                       # action per type
        a = acts[theta[:, i]]
        for t, j in enumerate(g.L[i]):
            S[:, i, j] = ((a >> t) & 1).astype(bool)
        n_told[:, i] = popcount_int(a)
    out = simulate_day(cfg, d, S)
    x = out["x_" + rule].astype(float)
    sd = {}
    for c, ta in enumerate(g.type_agents):
        i, th = ta.player, ta.theta
        lo, hi = reward_range(g.pop[i], cfg, i)
        u = realized_utility(g.pop[i], cfg, i, x, n_told[:, i])
        r = (u - lo) / (hi - lo)
        sd[c] = float(r[theta[:, i] == th].std())
    return sd


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = GameConfig(network="two_groups")
    setup = build_setup(cfg, ["E1", "E2"], RULES, [0], homogeneous(FRIEND(0.2), cfg.n), verbose=False)
    summ = json.load(open(os.path.join(X3, "x3_summary.json")))
    data = {n: np.load(os.path.join(X3, f"x3_{n}.npz"), allow_pickle=True) for n in LEARNERS}
    res = {"a": {}, "b": {}, "c": {}, "d": {}}

    for rule in RULES:
        env = "E1"
        g = setup.games[(env, rule)]
        cfg_e = cfg.with_(env=env)
        cells = g.type_agents
        eq = setup.targets[(env, rule)][0]                                 # unique pure equilibrium (cell actions)
        sig = g.profile_from_cells(eq)
        Vs = g.profile_interim_values(sig)                                  # per active player: (3, K) interim values
        pos = {i: p for p, i in enumerate(g.active)}
        arm_gap = []                                                       # per cell: gap of every arm to the equilibrium arm
        gap = []
        for c, ta in enumerate(cells):
            v = Vs[pos[ta.player]][ta.theta]
            arm_gap.append(v[eq[c]] - v)
            gap.append(np.sort(v)[-1] - np.sort(v)[-2])
        gap = np.array(gap)
        sd = cell_reward_sd(cfg_e, g, rule, sig)
        snr2 = np.array([(sd[c] / gap[c]) ** 2 for c in range(len(cells))])        # what a bandit has to average away
        by_type = {TYPES[th]: {"reward_sd": float(np.mean([sd[c] for c, ta in enumerate(cells) if ta.theta == th])),
                               "gap_pp": float(100 * np.mean([gap[c] for c, ta in enumerate(cells) if ta.theta == th])),
                               "noise_over_gap_sq": float(np.mean([snr2[c] for c, ta in enumerate(cells) if ta.theta == th]))}
                   for th in range(3)}
        # (a) rank correlation of (noise/gap)^2 with the final error rate over the cells, per learner
        corr = {}
        for name in LEARNERS:
            err = 1.0 - np.array(summ["cells"][f"{name}|{env}|{rule}"]["per_cell_correct"])
            rho, p = spearmanr(snr2, err)
            corr[name] = {"spearman": float(rho), "p": float(p),
                          "error_by_type": [float(np.mean(err[th::3])) for th in range(3)]}
        res["a"][f"{env}|{rule}"] = {"by_type": by_type, "per_cell_noise_over_gap_sq": snr2.tolist(), "per_cell_gap": gap.tolist(),
                                     "per_cell_reward_sd": [sd[c] for c in range(len(cells))], "correlation_with_error": corr,
                                     "average_is_hardest": bool(np.argmax([by_type[t]["noise_over_gap_sq"] for t in TYPES]) == 1)}

        # (b), (c) from the stored counts
        b, cc = {}, {}
        for name in LEARNERS:
            if name == "model-based":
                continue                                                   # counts include credited observations, not pulls
            d = data[name]
            envs, rules = list(d["envs"]), list(d["rules"])
            m = (d["env_idx"] == envs.index(env)) & (d["rule_idx"] == rules.index(rule))
            counts = d["final_counts"][m]                                  # (seeds, C, Kmax)
            tot = counts.sum(axis=2)
            wrong = np.stack([tot[:, c] - counts[:, c, eq[c]] for c in range(len(cells))], axis=1)
            share_wrong = float(wrong.sum() / tot.sum())
            cost = np.stack([(counts[:, c, :len(arm_gap[c])] * arm_gap[c][None]).sum(axis=1) for c in range(len(cells))], axis=1)
            b[name] = {"share_pulls_on_wrong_arms": share_wrong, "mean_cost_per_cell_from_wrong_pulls": float(cost.mean()),
                       "stored_regret": float(summ["cells"][f"{name}|{env}|{rule}"]["regret"]["mean"]),
                       "agreement": float(summ["cells"][f"{name}|{env}|{rule}"]["agree"]["mean"])}
            err = 1.0 - np.array(summ["cells"][f"{name}|{env}|{rule}"]["per_cell_correct"])
            cc[name] = {"mean_gap_of_mistakes_over_mean_gap": float((err * gap).sum() / max(err.sum(), 1e-12) / gap.mean()),
                        "n_cells_with_errors": int((err > 0.02).sum())}
        res["b"][f"{env}|{rule}"] = b
        res["c"][f"{env}|{rule}"] = cc

    # (d) pooling: relative regret reduction against UCB1, cells with 4 arms vs 2 arms
    for env in ("E1", "E2"):
        for rule in RULES:
            g = setup.games[(env, rule)]
            n_arms = np.array([ta.n_actions for ta in g.type_agents])
            row = {}
            for name in ("model-ucb", "model-based"):
                red = {}
                for key, mask in (("four_arm_cells", n_arms == 4), ("two_arm_cells", n_arms == 2)):
                    r = {}
                    for ln in (name, "UCB1"):
                        d = data[ln]
                        envs, rules = list(d["envs"]), list(d["rules"])
                        m = (d["env_idx"] == envs.index(env)) & (d["rule_idx"] == rules.index(rule))
                        r[ln] = d["regret_cell"][m][:, -1, :][:, mask].mean()
                    red[key] = float(1.0 - r[name] / r["UCB1"])
                row[name] = red
            res["d"][f"{env}|{rule}"] = row
    wins = sum(res["d"][k]["model-ucb"]["four_arm_cells"] > res["d"][k]["model-ucb"]["two_arm_cells"] for k in res["d"])
    res["d_summary"] = {"model_ucb_larger_in_four_arm_cells": int(wins), "of": len(res["d"])}

    with open(os.path.join(OUT, "explanations.json"), "w") as f:
        json.dump(res, f, indent=1)

    # ---- console digest
    print("(a) noise / gap by type (E1):")
    for rule in RULES:
        a = res["a"][f"E1|{rule}"]
        print(f"  {rule:5s} " + "  ".join(f"{t}: sd={a['by_type'][t]['reward_sd']:.3f} gap={a['by_type'][t]['gap_pp']:.2f}pp (sd/gap)^2={a['by_type'][t]['noise_over_gap_sq']:.0f}" for t in TYPES),
              "| average hardest:", a["average_is_hardest"])
        print("        Spearman((sd/gap)^2, error rate) per learner:", {k: round(v["spearman"], 2) for k, v in a["correlation_with_error"].items()})
        print("        final error rate by type, UCB1:", [round(x, 2) for x in a["correlation_with_error"]["UCB1"]["error_by_type"]])
    print("(b) share of pulls on non-equilibrium arms, and agreement (E1):")
    for rule in RULES:
        print(f"  {rule:5s} " + "  ".join(f"{n}: {100 * v['share_pulls_on_wrong_arms']:.0f}% (regret {v['stored_regret']:.0f}, agree {100 * v['agreement']:.0f}%)" for n, v in res["b"][f"E1|{rule}"].items()))
    print("(c) gap of the mistakes / mean gap (E1):")
    for rule in RULES:
        print(f"  {rule:5s} " + "  ".join(f"{n}: {v['mean_gap_of_mistakes_over_mean_gap']:.2f}" for n, v in res["c"][f"E1|{rule}"].items()))
    print("(d) relative regret reduction vs UCB1, four-arm cells | two-arm cells:")
    for k, v in res["d"].items():
        print(f"  {k:10s} model-ucb: {100 * v['model-ucb']['four_arm_cells']:.0f}% | {100 * v['model-ucb']['two_arm_cells']:.0f}%    leak-aware: {100 * v['model-based']['four_arm_cells']:.0f}% | {100 * v['model-based']['two_arm_cells']:.0f}%")
    print("(d) model-ucb reduction larger in four-arm cells in", wins, "of", len(res["d"]), "settings")


if __name__ == "__main__":
    main()
