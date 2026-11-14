"""Baselines the proposal promised: nobody shares, everybody shares, random sharing, the equilibrium.

For two friend groups, FRIEND(0.2), leak models E1 and E2, both hiring rules, the table gives the value of each of
the four baselines and of the profile each learner settled on (most-played arm of every decision cell, mean over the
30 seeds), as
  * welfare          sum over the six candidates of the ex-ante expected utility,
  * company value    E[V],  V = sum over hired j of (s_j - tau_K)   (nobody shares = the leak-free company),
  * share            fraction of possible tells made.
Under fixed seats exactly K candidates are always hired, so welfare only moves by the message cost and the comparison
is about the company; under a cut-off the number of hires moves with sharing.  Also reported: the average realised
utility per round of the learners over the last half of the run (from the stored learning curves).

Prediction written before the run: every learner's settled profile is worth more than random sharing and is close to
the equilibrium; refuted if any learner falls below random sharing in company value or welfare.
Outcome: true for welfare (what the players optimise) in every setting.  For company value the prediction was ill posed:
the company is not a player, and under a cut-off the equilibrium itself (everybody tells) is worth LESS to the company
than random sharing, so every learner is below random there; the right criterion for the company is "close to the
equilibrium", which holds.  The two checks are therefore stored separately.

Output: results/baselines/baselines.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from leakgame.analysis import matrix_distribution_values, profile_values  # noqa: E402
from leakgame.config import GameConfig  # noqa: E402
from leakgame.pipeline import build_setup  # noqa: E402
from leakgame.utility import FRIEND, homogeneous  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "baselines")
X3 = os.path.join(ROOT, "results", "x3")
LEARNERS = ["eps-greedy", "UCB1", "Thompson", "EXP3", "model-based", "model-ucb"]
ENVS = ["E1", "E2"]
RULES = ["SEATS", "BAR"]


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = GameConfig(network="two_groups")
    setup = build_setup(cfg, ["E0", "E1", "E2"], RULES, [0], homogeneous(FRIEND(0.2), cfg.n), verbose=False)
    res = {"alpha": 0.2, "games": {}}
    data = {n: np.load(os.path.join(X3, f"x3_{n}.npz"), allow_pickle=True) for n in LEARNERS}
    for env in ENVS:
        for rule in RULES:
            g = setup.games[(env, rule)]
            M = g.M
            rec = {"nobody": matrix_distribution_values(g, np.eye(M)[0]),
                   "everybody": matrix_distribution_values(g, np.eye(M)[M - 1]),
                   "random": matrix_distribution_values(g, np.ones(M))}
            eq = setup.targets[(env, rule)]
            assert eq.shape[0] == 1, "E1/E2 games have a unique pure equilibrium"
            rec["equilibrium"] = profile_values(g, g.profile_from_cells(eq[0]))
            rec["learners"] = {}
            for name in LEARNERS:
                d = data[name]
                envs, rules = list(d["envs"]), list(d["rules"])
                m = (d["env_idx"] == envs.index(env)) & (d["rule_idx"] == rules.index(rule))
                vals = [profile_values(g, g.profile_from_cells(row)) for row in d["final_counts"][m].argmax(axis=2)]
                rec["learners"][name] = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
                # realised utility per round over the last half of the run, from the cumulative averages
                ck = d["checkpoints"]
                iT, iH = len(ck) - 1, int(np.argmin(np.abs(ck - ck[-1] // 2)))
                um = d["util_mean"][m]
                last_half = (ck[iT] * um[:, iT] - ck[iH] * um[:, iH]) / (ck[iT] - ck[iH])
                rec["learners"][name]["realised_last_half"] = float(last_half.mean())
                rec["learners"][name]["realised_last_half_se"] = float(last_half.std(ddof=1) / np.sqrt(len(last_half)))
            res["games"][f"{env}|{rule}"] = rec
            print(f"{env} {rule}: " + "  ".join(f"{k} V={v['company_value']:.4f} W={v['welfare']:.4f} share={v['share']:.2f}"
                                                  for k, v in rec.items() if k != "learners"))
            for name, v in rec["learners"].items():
                print(f"    {name:12s} V={v['company_value']:.4f} W={v['welfare']:.4f} share={v['share']:.2f}  realised(last half)={v['realised_last_half']:.4f}")
            rec["learners_below_random_welfare"] = [n for n, v in rec["learners"].items()
                                                    if v["welfare"] < rec["random"]["welfare"] - 1e-9]
            rec["learners_below_random_company_value"] = [n for n, v in rec["learners"].items()
                                                          if v["company_value"] < rec["random"]["company_value"] - 1e-9]
            rec["equilibrium_below_random_company_value"] = rec["equilibrium"]["company_value"] < rec["random"]["company_value"] - 1e-9
            rec["max_company_value_gap_to_equilibrium"] = max(abs(v["company_value"] - rec["equilibrium"]["company_value"])
                                                              for v in rec["learners"].values())
            print("    below random sharing in welfare:", rec["learners_below_random_welfare"] or "none",
                  "| in company value:", len(rec["learners_below_random_company_value"]), "learners;",
                  "equilibrium itself below random in company value:", rec["equilibrium_below_random_company_value"])
    with open(os.path.join(OUT, "baselines.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
