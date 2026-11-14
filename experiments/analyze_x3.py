"""Statistics for X3: final metrics with 95% bootstrap CIs over seeds, rounds to 90% agreement,
paired BAR-vs-SEATS and model-based-vs-UCB1 comparisons on common random numbers, Holm correction.

Output: results/x3/x3_summary.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from leakgame.config import GameConfig  # noqa: E402
from leakgame.pipeline import build_setup  # noqa: E402
from leakgame.stats import bootstrap_ci, holm, paired_test, time_to_threshold  # noqa: E402
from leakgame.utility import FRIEND, homogeneous  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "x3")
LEARNERS = ["eps-greedy", "UCB1", "Thompson", "EXP3", "model-based", "model-ucb"]


def load(name):
    z = np.load(os.path.join(OUT, f"x3_{name}.npz"), allow_pickle=True)
    return {k: z[k] for k in z.files}


def sel(d, env, rule):
    envs = list(d["envs"]); rules = list(d["rules"])
    return (d["env_idx"] == envs.index(env)) & (d["rule_idx"] == rules.index(rule))


def main():
    data = {n: load(n) for n in LEARNERS if os.path.exists(os.path.join(OUT, f"x3_{n}.npz"))}
    names = list(data)
    d0 = data[names[0]]
    envs, rules = list(d0["envs"]), list(d0["rules"])
    ck = d0["checkpoints"]
    T = int(d0["T"])
    summary = {"learners": names, "envs": envs, "rules": rules, "T": T, "n_seeds": int(len(d0["seeds"])), "cells": {}, "tests": {},
               "checkpoints": ck.tolist(), "cell_names": [str(c) for c in d0["cell_names"]]}
    # equilibrium targets (same tables as the runs; cached) for the per-decision breakdown
    cfg = GameConfig(network="two_groups")
    setup = build_setup(cfg, envs, rules, [0], homogeneous(FRIEND(0.2), cfg.n), verbose=False)
    summary["n_equilibria"] = {f"{e}|{r}": int(setup.targets[(e, r)].shape[0]) for e in envs for r in rules}
    # size of the payoff gap a learner has to detect: best minus second-best action at equilibrium
    gaps = {}
    for e in envs:
        for r in rules:
            g = setup.games[(e, r)]
            sig = g.profile_from_cells(setup.targets[(e, r)][0])
            vals = []
            for Vi in g.profile_interim_values(sig):
                for th in range(3):
                    v = np.sort(Vi[th]); vals.append(float(v[-1] - v[-2]))
            gaps[f"{e}|{r}"] = {"min": min(vals), "median": float(np.median(vals)), "max": max(vals), "per_cell": vals}
    summary["payoff_gaps"] = gaps
    # ---- per (learner, env, rule) final metrics
    i_mid = int(np.argmin(np.abs(ck - 60000)))   # pilot horizon for comparability
    for name in names:
        d = data[name]
        for env in envs:
            for rule in rules:
                m = sel(d, env, rule)
                rec = {}
                for key, label in [("agree_greedy", "agree"), ("agree_mostplayed", "agree_mp"), ("regret_mean", "regret"),
                                   ("regret_max", "regret_max"), ("cce_gap", "cce_gap"), ("cce_gap_cell_mean", "cce_gap_cell")]:
                    x = d[key][m, -1]
                    rec[label] = {"mean": float(x.mean()), "ci95": bootstrap_ci(x), "sd": float(x.std(ddof=1))}
                    rec[label + "_curve_mean"] = d[key][m].mean(axis=0).tolist()
                    lo, hi = zip(*[bootstrap_ci(d[key][m, j], B=2000) for j in range(len(ck))])
                    rec[label + "_curve_lo"] = list(lo); rec[label + "_curve_hi"] = list(hi)
                rec["agree_at_60k"] = {"mean": float(d["agree_greedy"][m, i_mid].mean()), "ci95": bootstrap_ci(d["agree_greedy"][m, i_mid])}
                rec["agree_mp_at_60k"] = {"mean": float(d["agree_mostplayed"][m, i_mid].mean()), "ci95": bootstrap_ci(d["agree_mostplayed"][m, i_mid])}
                tt = time_to_threshold(d["agree_greedy"][m], ck, 0.9)
                rec["rounds_to_90"] = {"frac_reached": float(np.isfinite(tt).mean()),
                                       "median": float(np.median(tt)) if np.isfinite(np.median(tt)) else None,
                                       "median_reached": float(np.median(tt[np.isfinite(tt)])) if np.isfinite(tt).any() else None}
                tt2 = time_to_threshold(d["agree_greedy"][m], ck, 1.0)
                rec["rounds_to_100"] = {"frac_reached": float(np.isfinite(tt2).mean()),
                                        "median": float(np.median(tt2)) if np.isfinite(np.median(tt2)) else None}
                # CCE gap decrease: gap(T) vs gap(10^4)
                i10k = int(np.argmin(np.abs(ck - 10000)))
                rec["cce_gap_ratio_T_over_10k"] = float(d["cce_gap"][m, -1].mean() / max(d["cce_gap"][m, i10k].mean(), 1e-12))
                rec["cce_gap_at_10k"] = float(d["cce_gap"][m, i10k].mean())
                if name in ("model-based", "model-ucb"):
                    rec["leak_rate_hat"] = {"mean": float(np.nanmean(d["leak_rate_hat"][m, -1])), "sd": float(np.nanstd(d["leak_rate_hat"][m, -1]))}
                # per-decision breakdown: is the most-played action the equilibrium action?
                mp = d["final_counts"][m].argmax(axis=2)                          # (seeds, C)
                tg = setup.targets[(env, rule)]                                   # (n_eq, C)
                match = mp[:, None, :] == tg[None, :, :]                          # (seeds, n_eq, C)
                best = match.mean(axis=2).argmax(axis=1)
                per_cell = match[np.arange(len(mp)), best].mean(axis=0)
                rec["per_cell_correct"] = per_cell.tolist()
                rec["per_type_correct"] = [float(per_cell[th::3].mean()) for th in range(3)]
                # what could any candidate still gain by deviating from the profile the learners settled on?
                g = setup.games[(env, rule)]
                sigs = np.stack([g.profile_from_cells(row) for row in mp])
                ex = g.exploitability(sigs)
                rec["expl_mostplayed"] = {"mean": float(ex.mean()), "ci95": bootstrap_ci(ex), "max": float(ex.max()),
                                          "frac_exact_eq": float((ex <= 1e-12).mean())}
                summary["cells"][f"{name}|{env}|{rule}"] = rec
    # ---- H2-type paired test: BAR vs SEATS final agreement (per learner x env), Holm over 15
    tests = []
    for name in names:
        d = data[name]
        for env in envs:
            x = d["agree_greedy"][sel(d, env, "BAR"), -1]; y = d["agree_greedy"][sel(d, env, "SEATS"), -1]
            t = paired_test(x, y); t.update({"learner": name, "env": env, "what": "agree(BAR)-agree(SEATS)"}); tests.append(t)
    adj = holm([t["p"] for t in tests])
    for t, a in zip(tests, adj):
        t["p_holm"] = float(a)
    summary["tests"]["bar_vs_seats_agreement"] = tests
    # ---- H4: leak-aware (counterfactual credit) and model-based learners vs UCB1 (paired by seed/env/rule):
    #      agreement at 20k and final, regret, CCE gap; one Holm family per learner
    for mb_name, test_key in (("model-based", "model_based_vs_ucb1"), ("model-ucb", "model_ucb_vs_ucb1")):
      if mb_name in data and "UCB1" in data:
        tests = []
        i20 = int(np.argmin(np.abs(ck - 20000)))
        for env in envs:
            for rule in rules:
                mb, ub = data[mb_name], data["UCB1"]
                m1, m2 = sel(mb, env, rule), sel(ub, env, rule)
                for key, idx, what in [("agree_greedy", i20, "agree@20k(MB)-agree@20k(UCB1)"),
                                       ("agree_greedy", -1, "agree@T(MB)-agree@T(UCB1)"),
                                       ("regret_mean", -1, "regret(MB)-regret(UCB1)"),
                                       ("cce_gap", -1, "cce_gap(MB)-cce_gap(UCB1)")]:
                    t = paired_test(mb[key][m1, idx], ub[key][m2, idx]); t.update({"env": env, "rule": rule, "what": what}); tests.append(t)
        adj = holm([t["p"] for t in tests])
        for t, a in zip(tests, adj):
            t["p_holm"] = float(a)
        summary["tests"][test_key] = tests
        if mb_name == "model-based":
            # identical trajectories under E0?  (only the counterfactual-credit learner is UCB1 plus credit)
            m1 = sel(data["model-based"], "E0", "SEATS"); m2 = sel(data["UCB1"], "E0", "SEATS")
            summary["mb_equals_ucb1_under_E0"] = bool(np.allclose(data["model-based"]["agree_greedy"][m1], data["UCB1"]["agree_greedy"][m2]) and
                                                      np.allclose(data["model-based"]["regret_mean"][m1], data["UCB1"]["regret_mean"][m2]))
    # ---- H4: does the CCE gap fall?  paired gap(T) vs gap(10^4) for each learner (pooled over env/rule), Holm over learners
    tests = []
    i10k = int(np.argmin(np.abs(ck - 10000)))
    for name in names:
        d = data[name]
        t = paired_test(d["cce_gap"][:, i10k], d["cce_gap"][:, -1]); t.update({"learner": name, "what": "cce_gap(10k)-cce_gap(T)"}); tests.append(t)
    adj = holm([t["p"] for t in tests])
    for t, a in zip(tests, adj):
        t["p_holm"] = float(a)
    summary["tests"]["cce_gap_falls"] = tests
    # ---- learner ranking by final agreement pooled (for the text)
    summary["ranking_final_agree"] = sorted([(n, float(data[n]["agree_greedy"][:, -1].mean())) for n in names], key=lambda x: -x[1])
    summary["seconds_per_learner"] = {n: float(data[n]["seconds"]) for n in names}
    with open(os.path.join(OUT, "x3_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    # console digest
    print(f"{'learner':12s} {'env':3s} {'rule':5s} | agree(T) [CI]        agree@60k  mp@60k | regret(T)  cce_gap(T)   t90(frac, med)")
    for name in names:
        for env in envs:
            for rule in rules:
                r = summary["cells"][f"{name}|{env}|{rule}"]
                a = r["agree"]; t90 = r["rounds_to_90"]
                print(f"{name:12s} {env:3s} {rule:5s} | {a['mean']:.3f} [{a['ci95'][0]:.3f},{a['ci95'][1]:.3f}]  "
                      f"{r['agree_at_60k']['mean']:.3f}     {r['agree_mp_at_60k']['mean']:.3f}  | {r['regret']['mean']:8.2f}  "
                      f"{r['cce_gap']['mean']:.5f}   ({t90['frac_reached']:.2f}, {t90['median']})  "
                      f"expl={r['expl_mostplayed']['mean']:.5f} exact_eq={r['expl_mostplayed']['frac_exact_eq']:.2f} "
                      f"type_ok={[round(v, 2) for v in r['per_type_correct']]}")
    print("BAR vs SEATS agreement (Holm):")
    for t in summary["tests"]["bar_vs_seats_agreement"]:
        print(f"  {t['learner']:12s} {t['env']}: diff={t['mean_diff']:+.3f} CI=({t['ci95'][0]:+.3f},{t['ci95'][1]:+.3f}) p={t['p']:.4f} p_holm={t['p_holm']:.4f}")
    for test_key, label in (("model_based_vs_ucb1", "leak-aware (counterfactual credit)"), ("model_ucb_vs_ucb1", "model-based UCB")):
        if test_key in summary["tests"]:
            print(f"{label} vs UCB1 (Holm):")
            for t in summary["tests"][test_key]:
                print(f"  {t['env']} {t['rule']:5s} {t['what']:32s}: diff={t['mean_diff']:+.4f} p={t['p']:.4f} p_holm={t['p_holm']:.4f}")
    if "mb_equals_ucb1_under_E0" in summary:
        print("leak-aware == UCB1 under E0:", summary.get("mb_equals_ucb1_under_E0"))
    print("CCE gap falls (10k -> T), Holm:")
    for t in summary["tests"]["cce_gap_falls"]:
        print(f"  {t['learner']:12s}: diff={t['mean_diff']:+.5f} p={t['p']:.4f} p_holm={t['p_holm']:.4f}")


if __name__ == "__main__":
    main()
