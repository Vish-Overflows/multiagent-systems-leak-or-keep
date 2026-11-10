"""X4 (target): bank size Q swept in E2; company's hiring-value loss Lambda_R(Q).

  Lambda_R = E[V | no leaks] - E[V | equilibrium sharing],  V = sum_{j in H} (s_j - tau_K),
with the equilibrium S* re-solved for every Q (the leak probability m/Q changes the game).  We also
report the loss under "everybody shares" (an upper bound for any sharing pattern) and the relative
loss Lambda / E[V | no leaks].  95% CIs come from the block bootstrap of the table.

Output: results/x4/x4.json
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from leakgame.config import GameConfig  # noqa: E402
from leakgame.game import AgentFormGame, popcount_int  # noqa: E402
from leakgame.solvers import enumerate_pure_bne  # noqa: E402
from leakgame.tables import build_table  # noqa: E402
from leakgame.utility import FRIEND, homogeneous  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "x4")
QS = [4, 6, 9, 12, 15, 20, 25, 30, 40, 50, 64]
N_BOOT = 500


def value_of_profile(game: AgentFormGame, sig: np.ndarray, Vtab: np.ndarray) -> float:
    """E over types of V[theta, S*(theta)] for one pure profile (Vtab: (T, M) for this rule)."""
    S_idx = np.zeros(game.T, dtype=np.int64)
    for j in game.active:
        a_j = game._decode_action(j, np.int64(sig[j]), game.profiles[:, j])
        S_idx += game.rowbits[j][a_j]
    return float(Vtab[np.arange(game.T), S_idx].mean())


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    rows = []
    rng = np.random.default_rng(0)
    for Q in QS:
        cfg = GameConfig(network="two_groups", env="E2", Q=Q)
        tab = build_table(cfg, n_days=2000, seed=0, verbose=False)
        rec = {"Q": Q, "leak_p": cfg.leak_p}
        for rule_i, rule in enumerate(("SEATS", "BAR")):
            g = AgentFormGame(tab, homogeneous(FRIEND(0.2), cfg.n), rule)
            res = enumerate_pure_bne(g)
            Vtab = tab.V[:, :, rule_i].astype(float)
            v_noleak = float(Vtab[:, 0].mean())
            v_all = float(Vtab[:, -1].mean())
            star = res.profiles[0] if res.count else None
            v_star = value_of_profile(g, star, Vtab) if star is not None else float("nan")
            # bootstrap CI for Lambda at the (fixed) equilibrium profile
            lam_bs = []
            if star is not None:
                S_idx = np.zeros(g.T, dtype=np.int64)
                for j in g.active:
                    a_j = g._decode_action(j, np.int64(star[j]), g.profiles[:, j])
                    S_idx += g.rowbits[j][a_j]
                for _ in range(N_BOOT):
                    Vb = tab.resample_V_blocks(rng)[:, :, rule_i].astype(float)
                    lam_bs.append(float(Vb[:, 0].mean() - Vb[np.arange(g.T), S_idx].mean()))
                lam_bs = np.array(lam_bs)
            rec[rule] = {
                "n_bne": res.count,
                "sharing_rate": float(g.sharing_rate(star)[0]) if star is not None else None,
                "cell_tells": popcount_int(g.cells_from_profile(star)).tolist() if star is not None else None,
                "V_noleak": v_noleak, "V_eq": v_star, "V_allshare": v_all,
                "Lambda_eq": v_noleak - v_star, "Lambda_allshare": v_noleak - v_all,
                "Lambda_eq_rel": (v_noleak - v_star) / v_noleak, "Lambda_allshare_rel": (v_noleak - v_all) / v_noleak,
                "Lambda_eq_ci95": [float(np.quantile(lam_bs, 0.025)), float(np.quantile(lam_bs, 0.975))] if star is not None else None,
                "Lambda_eq_rel_ci95": [float(np.quantile(lam_bs, 0.025) / v_noleak), float(np.quantile(lam_bs, 0.975) / v_noleak)] if star is not None else None,
            }
        rows.append(rec)
        print(f"Q={Q:3d}: SEATS share={rec['SEATS']['sharing_rate']:.2f} Lambda={rec['SEATS']['Lambda_eq']:.4f} "
              f"({rec['SEATS']['Lambda_eq_rel']*100:.1f}%) | BAR share={rec['BAR']['sharing_rate']:.2f} "
              f"Lambda={rec['BAR']['Lambda_eq']:.4f} ({rec['BAR']['Lambda_eq_rel']*100:.1f}%)", flush=True)
    out = {"rows": rows, "seconds": time.time() - t0, "n_boot": N_BOOT}
    with open(os.path.join(OUT, "x4.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"X4 done in {out['seconds']:.0f}s")


if __name__ == "__main__":
    main()
