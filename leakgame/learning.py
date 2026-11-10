"""Level-2: repeated play of the stage game by bandit learners, vectorised over
independent runs.  A run is (seed, environment, hiring rule); all runs that
share a seed see the same environment draws (common random numbers), so SEATS
vs BAR and E0 vs E1 vs E2 can be compared pairwise.

Each round t:
  1. skills s^t ~ N(0,1) -> types theta^t;
  2. every decision cell (candidate, type) picks an arm (a row of S) from its policy; the
     realised cells' arms form S^t;  the full vector of picks is the strategy profile
     sigma^t used for the CCE-gap metric;
  3. the environment draws leaks, scores and hires;  each realised cell gets only its own
     realised utility u_i^t (and, for the model-based learner, k_j^{(i)});
  4. metrics are accumulated with the *expected* (table) payoffs:
       pseudo-regret_c(t) = max_a sum_{s<=t, c acted} [ubar_c(a, a^s_{-c}) - ubar_c(a^s)]
       CCE gap(t)        = max_i (1/3) sum_theta max_a (1/t) sum_{s<=t} [V_i(sigma^s_{-i}, theta, a)
                                                                  - V_i(sigma^s_{-i}, theta, sigma^s_i(theta))]
       agreement(t)      = fraction of cells whose current greedy arm equals the BNE arm
                           (max over pure BNEs if several).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .config import GameConfig, type_of
from .env import DayDraws, draw_question_sets, hire_seats, leak_counts_batch
from .game import AgentFormGame, popcount_int
from .learners import LEARNERS, ModelBased
from .tables import PayoffTable, profile_index
from .utility import Utility, reward_range, weights


@dataclass
class LearningSetup:
    cfg: GameConfig                       # network / parameters (env field ignored)
    envs: List[str]
    rules: List[str]
    seeds: List[int]
    pop: List[Utility]
    tables: Dict[str, PayoffTable]        # env -> table
    games: Dict[tuple, AgentFormGame]     # (env, rule) -> game with V tables
    targets: Dict[tuple, np.ndarray]      # (env, rule) -> (n_bne, C) BNE cell actions


def checkpoints(T: int, every: int = 1000) -> np.ndarray:
    pts = {100, 200, 500} | set(range(every, T + 1, every)) | {T}
    return np.array(sorted(p for p in pts if p <= T))


def run_learning(learner_name: str, setup: LearningSetup, T: int = 200_000, ckpt_every: int = 1000,
                 learner_seed: int = 12345, env_seed: int = 999, verbose: bool = True) -> Dict:
    cfg0 = setup.cfg
    n, m, K = cfg0.n, cfg0.m, cfg0.K
    envs, rules, seeds = setup.envs, setup.rules, setup.seeds
    n_seeds, n_env, n_rule = len(seeds), len(envs), len(rules)
    B = n_seeds * n_env * n_rule
    # batch layout: b = (seed_i * n_env + env_i) * n_rule + rule_i
    seed_idx = np.repeat(np.arange(n_seeds), n_env * n_rule)
    env_idx = np.tile(np.repeat(np.arange(n_env), n_rule), n_seeds)
    rule_idx = np.tile(np.arange(n_rule), n_seeds * n_env)
    combo = env_idx * n_rule + rule_idx
    combos = [(e, r) for e in envs for r in rules]
    is_seats = np.array([rules[r] == "SEATS" for r in rule_idx])[:, None]
    game0 = setup.games[combos[0]]
    active = game0.active
    cells = game0.type_agents
    C = len(cells)
    Kmax = max(game0.K)
    cell_idx = {i: np.array([c for c, ta in enumerate(cells) if ta.player == i]) for i in active}  # (3,) per player
    valid = np.zeros((C, Kmax), dtype=bool)
    for c, ta in enumerate(cells):
        valid[c, :ta.n_actions] = True
    cfg_env = {e: cfg0.with_(env=e) for e in envs}
    tau = cfg0.tau
    rowmask = {}
    for i in active:
        rm = np.zeros((game0.K[i], n), dtype=bool)
        for a in range(game0.K[i]):
            for tpos, j in enumerate(game0.L[i]):
                if (a >> tpos) & 1:
                    rm[a, j] = True
        rowmask[i] = rm
    Lpos = {i: {j: tpos for tpos, j in enumerate(game0.L[i])} for i in active}
    # utilities
    Wlin = np.zeros((n, n)); Wpair = np.zeros((n, n))
    lo = np.zeros(n); hi = np.ones(n)
    for i in range(n):
        Wlin[i], Wpair[i] = weights(setup.pop[i], cfg0, i)
        lo[i], hi[i] = reward_range(setup.pop[i], cfg0, i)
    # stacked tables over combos
    EU_stack = {i: np.stack([setup.games[cb].EU[i] for cb in combos]) for i in active}          # (NC, T, M)
    V_stack = {i: np.stack([setup.games[cb].V[i] for cb in combos]) for i in active}             # (NC, O, 3, K)
    targets = [setup.targets[cb] for cb in combos]                                                # list of (n_bne, C)
    n_bne_max = max(t.shape[0] for t in targets)
    target_arr = np.full((len(combos), n_bne_max, C), -1)
    for ci, t in enumerate(targets):
        target_arr[ci, :t.shape[0]] = t
    target_b = target_arr[combo]                                                                  # (B, n_bne_max, C)

    rng_env = np.random.default_rng(env_seed)
    rng_learn = np.random.default_rng(learner_seed)
    Learner = LEARNERS[learner_name]
    learner = Learner(B, C, Kmax, valid, rng_learn, horizon_per_cell=T / 3.0)
    is_mb = isinstance(learner, ModelBased)

    regret_sum = np.zeros((B, C, Kmax)); regret_played = np.zeros((B, C))
    cce_sum = np.zeros((B, C, Kmax)); cce_played = np.zeros((B, C))
    ck = checkpoints(T, ckpt_every)
    out = {k: np.zeros((B, len(ck))) for k in ["agree_greedy", "agree_mostplayed", "regret_mean", "regret_max",
                                                "cce_gap", "cce_gap_cell_mean", "leak_rate_hat", "util_mean"]}
    out["regret_cell"] = np.zeros((B, len(ck), C))
    util_acc = np.zeros(B)
    arangeB = np.arange(B)
    cell_theta = np.array([ta.theta for ta in cells])
    cell_player = np.array([ta.player for ta in cells])
    t0 = time.time()
    ck_pos = 0
    for t in range(1, T + 1):
        # ---- 1. types (shared across combos of a seed)
        skills_s = rng_env.standard_normal((n_seeds, n))
        noise_s = rng_env.standard_normal((n_seeds, n)) * (cfg0.sigma / np.sqrt(m))
        bern_s = rng_env.random((n_seeds, n, n, m)) < (m / cfg0.Q) if "E1" in envs else None
        sets_s = draw_question_sets(rng_env, n_seeds, n, cfg0.Q, m) if "E2" in envs else None
        skills = skills_s[seed_idx]; noise = noise_s[seed_idx]
        theta = type_of(skills)                                                  # (B, n)
        t_idx = profile_index(theta)
        # ---- 2. actions
        act = learner.choose()                                                   # (B, C)
        a_real = np.zeros((B, n), dtype=np.int64)
        S = np.zeros((B, n, n), dtype=bool)
        for i in active:
            a_real[:, i] = act[arangeB, cell_idx[i][theta[:, i]]]
            S[:, i, :] = rowmask[i][a_real[:, i]]
        # ---- 3. environment
        k = np.zeros((B, n), dtype=np.int16); kij = np.zeros((B, n, n), dtype=np.int16)
        for e, env in enumerate(envs):
            idx = np.where(env_idx == e)[0]
            d = DayDraws(skills[idx], noise[idx],
                         bern_s[seed_idx[idx]] if env == "E1" else None,
                         sets_s[seed_idx[idx]] if env == "E2" else None)
            k[idx], kij[idx] = leak_counts_batch(cfg_env[env], d, S[idx])
        Y = skills + (cfg0.delta / m) * k + noise
        x_seats = hire_seats(Y, K)
        x_bar = Y >= tau
        x = np.where(is_seats, x_seats, x_bar)
        xf = x.astype(float)
        n_told = popcount_int(a_real)
        u = xf @ Wlin.T + xf * (xf @ Wpair.T) - cfg0.c * n_told                  # (B, n)
        util_acc += u.sum(axis=1)
        # ---- 4. learner updates (realised cells only)
        bs, cs, arms, rs, credits, credit_rs = [], [], [], [], [], []
        for i in active:
            c = cell_idx[i][theta[:, i]]
            r = (u[:, i] - lo[i]) / (hi[i] - lo[i])
            bs.append(arangeB); cs.append(c); arms.append(a_real[:, i]); rs.append(r)
            if is_mb:
                inert = np.zeros(B, dtype=np.int64)
                for j in game0.L[i]:
                    told = S[:, i, j]
                    inert |= ((told & (kij[:, i, j] == 0)).astype(np.int64)) << Lpos[i][j]
                    if told.any():
                        learner.observe_leaks(arangeB[told], kij[told, i, j].astype(float))
                arm_b = a_real[:, i][:, None]
                ap = np.arange(Kmax)[None, :]
                credit = ((ap & ~arm_b) == 0) & (((arm_b ^ ap) & ~inert[:, None]) == 0) & valid[c]
                credits.append(credit)
                # the sub-arm would have saved the telling cost c per deleted tell (known to the candidate)
                saved = cfg0.c * (popcount_int(arm_b) - popcount_int(ap)) / (hi[i] - lo[i])
                credit_rs.append(r[:, None] + saved)
        b_all = np.concatenate(bs); c_all = np.concatenate(cs); a_all = np.concatenate(arms); r_all = np.concatenate(rs)
        if is_mb:
            learner.update(b_all, c_all, a_all, r_all, credit=np.concatenate(credits), credit_reward=np.concatenate(credit_rs))
        else:
            learner.update(b_all, c_all, a_all, r_all)
        # ---- 5. metrics
        # pseudo-regret: expected utility of every arm against realised others
        for i in active:
            S_oth = np.zeros(B, dtype=np.int64)
            for j in active:
                if j != i:
                    S_oth += game0.rowbits[j][a_real[:, j]]
            c = cell_idx[i][theta[:, i]]
            Ki = game0.K[i]
            vals = np.stack([EU_stack[i][combo, t_idx, S_oth + game0.rowbits[i][a]] for a in range(Ki)], axis=1)  # (B, Ki)
            regret_sum[arangeB[:, None], c[:, None], np.arange(Ki)[None, :]] += vals
            regret_played[arangeB, c] += vals[arangeB, a_real[:, i]]
        # CCE gap: full hypothetical profile sigma^t
        sig = np.zeros((B, n), dtype=np.int64)
        for j in active:
            Kj = game0.K[j]
            sig[:, j] = act[:, cell_idx[j][0]] + Kj * act[:, cell_idx[j][1]] + Kj * Kj * act[:, cell_idx[j][2]]
        for i in active:
            oth = game0.others_index(i, sig)
            Vi = V_stack[i][combo, oth]                                          # (B, 3, Ki)
            Ki = game0.K[i]
            cc = cell_idx[i]
            cce_sum[:, cc, :Ki] += Vi
            played = act[:, cc]                                                  # (B, 3)
            cce_played[:, cc] += Vi[arangeB[:, None], np.arange(3)[None, :], played]
        # ---- 6. checkpoints
        if ck_pos < len(ck) and t == ck[ck_pos]:
            g = learner.greedy(); mp = learner.most_played()
            out["agree_greedy"][:, ck_pos] = (g[:, None, :] == target_b).mean(axis=2).max(axis=1)
            out["agree_mostplayed"][:, ck_pos] = (mp[:, None, :] == target_b).mean(axis=2).max(axis=1)
            reg = np.where(valid[None], regret_sum, -np.inf).max(axis=2) - regret_played   # (B, C)
            out["regret_cell"][:, ck_pos] = reg
            out["regret_mean"][:, ck_pos] = reg.mean(axis=1)
            out["regret_max"][:, ck_pos] = reg.max(axis=1)
            gap_cell = (np.where(valid[None], cce_sum, -np.inf).max(axis=2) - cce_played) / t   # (B, C)
            gap_player = np.stack([gap_cell[:, cell_idx[i]].mean(axis=1) for i in active], axis=1)
            out["cce_gap"][:, ck_pos] = gap_player.max(axis=1)
            out["cce_gap_cell_mean"][:, ck_pos] = gap_cell.mean(axis=1)
            out["util_mean"][:, ck_pos] = util_acc / t
            if is_mb:
                out["leak_rate_hat"][:, ck_pos] = learner.leak_rate_estimate(m)
            ck_pos += 1
            if verbose and (ck_pos % 25 == 0 or t == T):
                el = time.time() - t0
                print(f"  [{learner_name}] t={t:7d}  agree={out['agree_greedy'][:, ck_pos-1].mean():.3f}  "
                      f"cce_gap={out['cce_gap'][:, ck_pos-1].mean():.5f}  {el:6.1f}s", flush=True)
    out.update({"checkpoints": ck, "seed_idx": seed_idx, "env_idx": env_idx, "rule_idx": rule_idx,
                "envs": np.array(envs), "rules": np.array(rules), "seeds": np.array(seeds),
                "learner": learner_name, "T": T, "seconds": time.time() - t0,
                "final_counts": learner.counts, "final_means": learner.means(),
                "cell_names": np.array([ta.name for ta in cells])})
    return out
