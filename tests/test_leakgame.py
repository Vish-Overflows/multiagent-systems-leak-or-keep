"""Fast unit tests (about 20 s).  Run with:  .venv/bin/python -m pytest -q"""
import numpy as np
import pytest

from leakgame.checks import lemma1_monotonicity, prop3_leak_formula, worked_example
from leakgame.config import GameConfig, type_of
from leakgame.env import draw_days, hire_bar, hire_seats, leak_counts_batch, leak_counts_table, scores
from leakgame.game import AgentFormGame
from leakgame.learners import EXP3, UCB1, EpsGreedy, ModelBased, Thompson
from leakgame.solvers import best_response_dynamics, cce_lp, enumerate_pure_bne
from leakgame.stats import bootstrap_ci, holm, paired_test
from leakgame.tables import all_matrices, build_table
from leakgame.utility import CLAN, FAIR, FRIEND, SELF, homogeneous, realized_utility, reward_range, weights


# ------------------------------------------------------------------ simulator
def test_free_entries_of_pilot_network():
    cfg = GameConfig()
    assert [(i + 1, j + 1) for i, j in cfg.free_entries] == [(1, 3), (1, 5), (2, 4), (2, 6), (3, 5), (4, 6)]
    assert cfg.later_friends(4) == [] and cfg.later_friends(5) == []


def test_types_are_terciles():
    s = np.random.default_rng(0).standard_normal(300_000)
    freq = np.bincount(type_of(s), minlength=3) / len(s)
    assert np.allclose(freq, 1 / 3, atol=0.005)


def test_seats_hires_exactly_K_and_bar_uses_threshold():
    cfg = GameConfig()
    Y = np.random.default_rng(1).standard_normal((1000, cfg.n))
    assert (hire_seats(Y, cfg.K).sum(axis=1) == cfg.K).all()
    assert (hire_bar(Y, cfg.tau) == (Y >= cfg.tau)).all()


def test_bar_hires_K_on_average_without_leaks():
    cfg = GameConfig(env="E1")
    d = draw_days(cfg, 200_000, np.random.default_rng(2))
    Y = scores(cfg, d, np.zeros((200_000, cfg.n), dtype=np.int16))
    assert abs(hire_bar(Y, cfg.tau).sum(axis=1).mean() - cfg.K) < 0.02


@pytest.mark.parametrize("env", ["E0", "E1", "E2"])
def test_table_and_batch_leak_counts_agree(env):
    cfg = GameConfig(env=env)
    d = draw_days(cfg, 500, np.random.default_rng(3))
    S_all = all_matrices(cfg)
    k_tab = leak_counts_table(cfg, d, S_all)                       # (B, M, n)
    for idx in (0, 5, 37, 63):
        k_b, _ = leak_counts_batch(cfg, d, np.broadcast_to(S_all[idx], (500, cfg.n, cfg.n)))
        assert (k_tab[:, idx, :] == k_b).all()


def test_lemma1_monotone_and_local():
    for env, r in lemma1_monotonicity(n_days=2000).items():
        assert r["pass"], env


def test_leak_formula():
    res = prop3_leak_formula(n_days=60_000)
    for env in ("E1", "E2"):
        for r in res[env].values():
            assert abs(r["sim"] - r["exact"]) < 5 * r["se"] + 1e-3
    assert all(b["holds"] for b in res["bound"].values())


def test_worked_example_matches_exact_values():
    r = worked_example(n_days=100_000)
    assert abs(r["P(friend sees >=1 question)"]["sim_kabir"] - 0.7083) < 0.01
    assert abs(r["P(someone overtakes Aarav)"]["sim"] - 0.9149) < 0.01


# ------------------------------------------------------------------ utilities
def test_realized_utilities_by_hand():
    cfg = GameConfig()                                  # groups {1,3,5}, {2,4,6}
    x = np.array([1, 0, 1, 1, 0, 0], dtype=float)       # hired: 1, 3, 4
    told = np.array(2.0)
    assert realized_utility(SELF, cfg, 0, x, told) == pytest.approx(1 - 2 * cfg.c)
    assert realized_utility(FRIEND(0.2), cfg, 0, x, told) == pytest.approx(1 + 0.2 * 1 - 2 * cfg.c)
    assert realized_utility(CLAN(0.2, 0.3), cfg, 0, x, told) == pytest.approx(1 + 0.2 - 0.3 * 1 - 2 * cfg.c)
    # Fehr-Schmidt: i hired, 3 of 5 others not hired -> advantageous inequality b * 3/5
    assert realized_utility(FAIR(0.5, 0.25), cfg, 0, x, told) == pytest.approx(1 - 0.25 * 3 / 5 - 2 * cfg.c)
    # i not hired, 3 others hired -> disadvantageous inequality a * 3/5
    assert realized_utility(FAIR(0.5, 0.25), cfg, 1, x, np.array(0.0)) == pytest.approx(-0.5 * 3 / 5)


def test_clan_is_rescaled_friend_under_seats():
    """Remark: with |H| = K, CLAN(a, b) = (1 + b) FRIEND((a + b)/(1 + b)) - b K   (cost-free part)."""
    cfg = GameConfig(c=0.0)
    rng = np.random.default_rng(4)
    a, b = 0.2, 0.3
    for _ in range(50):
        x = np.zeros(cfg.n)
        x[rng.choice(cfg.n, cfg.K, replace=False)] = 1
        for i in range(cfg.n):
            lhs = realized_utility(CLAN(a, b), cfg, i, x, np.array(0.0))
            rhs = (1 + b) * realized_utility(FRIEND((a + b) / (1 + b)), cfg, i, x, np.array(0.0)) - b * cfg.K
            assert lhs == pytest.approx(rhs)


def test_reward_range_bounds_all_outcomes():
    cfg = GameConfig()
    for u in (SELF, FRIEND(0.2), CLAN(0.2, 0.3), FAIR(1.0, 0.5)):
        lo, hi = reward_range(u, cfg, 0)
        for bits in range(1 << cfg.n):
            x = np.array([(bits >> j) & 1 for j in range(cfg.n)], dtype=float)
            for told in (0.0, 2.0):
                assert lo - 1e-12 <= realized_utility(u, cfg, 0, x, np.array(told)) <= hi + 1e-12


# ------------------------------------------------------------------ equilibrium solvers
@pytest.fixture(scope="module")
def small_table():
    return build_table(GameConfig(n=4, K=2, env="E1", network="complete"), n_days=600, seed=0,
                       cache_dir=None, verbose=False)


def test_nobody_shares_when_selfish(small_table):
    for rule in ("SEATS", "BAR"):
        g = AgentFormGame(small_table, homogeneous(SELF, 4), rule)
        res = enumerate_pure_bne(g)
        assert res.count == 1 and g.sharing_rate(res.profiles[0])[0] == 0.0


def test_prop1_no_sharing_in_a_clique_under_seats(small_table):
    g = AgentFormGame(small_table, homogeneous(FRIEND(0.5), 4), "SEATS")
    res = enumerate_pure_bne(g)
    assert res.count == 1 and g.sharing_rate(res.profiles[0])[0] == 0.0


def test_everyone_shares_under_bar(small_table):
    g = AgentFormGame(small_table, homogeneous(FRIEND(0.5), 4), "BAR")
    res = enumerate_pure_bne(g)
    assert res.count == 1 and g.sharing_rate(res.profiles[0])[0] == pytest.approx(1.0)


def test_best_response_dynamics_reaches_enumerated_equilibrium(small_table):
    g = AgentFormGame(small_table, homogeneous(FRIEND(0.5), 4), "BAR")
    res = enumerate_pure_bne(g)
    out = best_response_dynamics(g, np.zeros(4, dtype=np.int64))
    assert out["converged"] and (out["profile"] == res.profiles[0]).all()
    assert g.exploitability(out["profile"])[0] == 0.0


def test_exploitability_matches_direct_computation(small_table):
    g = AgentFormGame(small_table, homogeneous(FRIEND(0.3), 4), "SEATS")
    rng = np.random.default_rng(5)
    for _ in range(5):
        sig = np.array([rng.integers(0, s) for s in g.n_strat])
        assert g.exploitability(sig)[0] == pytest.approx(g.profile_exploitability_direct(sig)[0], abs=1e-12)


def test_cce_contains_the_equilibrium():
    cfg = GameConfig(n=4, K=2, env="E1", network="pairs")
    tab = build_table(cfg, n_days=600, seed=0, cache_dir=None, verbose=False)
    g = AgentFormGame(tab, homogeneous(FRIEND(0.5), 4), "BAR")
    eq_rate = g.sharing_rate(enumerate_pure_bne(g).profiles[0])[0]
    hi, lo = cce_lp(g, objectives=("max_sharing", "min_sharing"))
    assert lo.sharing_rate - 1e-9 <= eq_rate <= hi.sharing_rate + 1e-9


# ------------------------------------------------------------------ learners
@pytest.mark.parametrize("cls", [EpsGreedy, UCB1, Thompson, EXP3])
def test_learners_find_the_better_arm_of_a_stationary_bandit(cls):
    rng = np.random.default_rng(6)
    B, C, K, T = 20, 1, 2, 4000
    L = cls(B, C, K, np.ones((C, K), dtype=bool), rng, horizon_per_cell=T)
    p = np.array([0.4, 0.6])
    b = np.arange(B); c = np.zeros(B, dtype=int)
    for _ in range(T):
        arm = L.choose()[:, 0]
        L.update(b, c, arm, (rng.random(B) < p[arm]).astype(float))
    assert (L.most_played()[:, 0] == 1).mean() >= 0.9


def test_learners_never_pick_padded_arms():
    rng = np.random.default_rng(7)
    valid = np.array([[True, True, False, False], [True, True, True, True]])
    for cls in (EpsGreedy, UCB1, Thompson, EXP3, ModelBased):
        L = cls(8, 2, 4, valid, rng, horizon_per_cell=100)
        for _ in range(50):
            a = L.choose()
            assert (a[:, 0] < 2).all()
            L.update(np.repeat(np.arange(8), 2), np.tile(np.arange(2), 8), a.ravel(), rng.random(16))


def test_model_based_credits_only_named_arms():
    rng = np.random.default_rng(8)
    L = ModelBased(1, 1, 4, np.ones((1, 4), dtype=bool), rng)
    credit = np.array([[True, False, False, True]])      # played arm 3 (told both), both tells inert -> also arm 0
    L.update(np.array([0]), np.array([0]), np.array([3]), np.array([0.7]), credit=credit)
    assert L.counts[0, 0].tolist() == [1, 0, 0, 1] and L.sums[0, 0].tolist() == [0.7, 0, 0, 0.7]


# ------------------------------------------------------------------ statistics
def test_holm_adjustment():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert np.allclose(holm([0.5, 0.5]), [1.0, 1.0])


def test_paired_test_and_ci():
    rng = np.random.default_rng(9)
    x = rng.normal(1.0, 1.0, 30)
    t = paired_test(x + 0.8, x)
    assert t["mean_diff"] == pytest.approx(0.8) and t["p"] < 0.001
    lo, hi = bootstrap_ci(x)
    assert lo < x.mean() < hi
