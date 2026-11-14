"""Bandit learners for the Level-2 (unknown environment) game.

Every (candidate, type) pair -- a *decision cell* -- is an independent bandit
learner over the rows of S available to that candidate.  All learners are
vectorised over a batch axis B (independent runs) and a cell axis C, with
arms padded to K_max and masked by `valid`.  Rewards are the candidate's own
realised utility, normalised to [0, 1] with the (known) range of her utility.

  EpsGreedy   eps_t = min(1, t^{-1/3}) on the cell's own round count t
  UCB1        mean + sqrt(2 ln t / n_a)                         (Auer et al. 2002)
  Thompson    Gaussian posterior sampling, prior N(1/2, 1), noise var 1/4
  EXP3        exponential weights with mixing gamma, importance-weighted rewards
              (Auer et al. 2002b), gamma tuned to the expected horizon
  ModelBased  "counterfactual-credit" learner: uses the structure of the
              environment -- a tell matters only through questions that actually
              reappear -- plus the feedback k_j^{(i)} (how many of the questions i
              told j came up in j's interview).  When k_j^{(i)} = 0 the tell was
              inert and the outcome is *exactly* the one the arm without that tell
              would have produced, so the observation (plus the saved telling cost,
              which the candidate knows) is credited to every arm obtained by
              deleting inert tells.  Chooses with UCB1 on the augmented statistics;
              identical to UCB1 under E0 (nothing is inert).

              NOT a model-based planner: it does not estimate Delta and does not
              best-respond in an estimated game, as the project review proposed.  The
              running estimate p-hat = P(a shared question reappears) is logged
              (`leak_rate_estimate`) for reporting only and is never used to choose an
              arm; the whole advantage over UCB1 comes from the counterfactual credit.
              The registry key "model-based" is kept only so result files stay valid.
              Exactness: in E1 the credited sample has exactly the distribution of a
              sample of the shorter arm; in E2 the day is identical but selecting days
              on which the two question sets are disjoint very slightly changes the
              other leaks, so the credit is not exactly unbiased there.
"""
from __future__ import annotations

import numpy as np


class Learner:
    name = "base"

    def __init__(self, B: int, C: int, Kmax: int, valid: np.ndarray, rng: np.random.Generator,
                 horizon_per_cell: float = 1.0):
        self.B, self.C, self.Kmax = B, C, Kmax
        self.valid = valid.astype(bool)                       # (C, Kmax)
        self.Kc = self.valid.sum(axis=1)                      # (C,)
        self.rng = rng
        self.counts = np.zeros((B, C, Kmax))
        self.sums = np.zeros((B, C, Kmax))
        self.t = np.zeros((B, C))                             # own rounds per cell
        self.neg_inf = np.where(self.valid, 0.0, -np.inf)[None]   # (1, C, Kmax)

    # ---- helpers
    def means(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = np.where(self.counts > 0, self.sums / np.maximum(self.counts, 1), 0.5)
        return mu

    def _argmax_random_ties(self, score: np.ndarray) -> np.ndarray:
        score = np.where(self.valid[None], score + 1e-12 * self.rng.random(score.shape), -np.inf)
        return score.argmax(axis=-1)

    def greedy(self) -> np.ndarray:
        """Current greedy action per cell (used for the agreement-with-BNE metric)."""
        return self._argmax_random_ties(self.means())

    def most_played(self) -> np.ndarray:
        return self._argmax_random_ties(self.counts)

    def _uniform_valid(self, shape) -> np.ndarray:
        u = np.where(self.valid[None], self.rng.random(shape + (self.Kmax,)), -np.inf)
        return u.argmax(axis=-1)

    # ---- interface
    def choose(self) -> np.ndarray:
        raise NotImplementedError

    def update(self, b: np.ndarray, c: np.ndarray, arm: np.ndarray, r: np.ndarray, credit: np.ndarray | None = None):
        """b, c, arm, r: flat arrays of realised (batch, cell, arm, normalised reward)."""
        np.add.at(self.counts, (b, c, arm), 1.0)
        np.add.at(self.sums, (b, c, arm), r)
        np.add.at(self.t, (b, c), 1.0)


class EpsGreedy(Learner):
    name = "eps-greedy"

    def choose(self) -> np.ndarray:
        eps = np.minimum(1.0, np.power(self.t + 1.0, -1.0 / 3.0))       # (B, C)
        explore = self.rng.random((self.B, self.C)) < eps
        return np.where(explore, self._uniform_valid((self.B, self.C)), self.greedy())


class UCB1(Learner):
    name = "UCB1"

    def index(self) -> np.ndarray:
        t = np.maximum(self.t, 1.0)[..., None]
        with np.errstate(divide="ignore", invalid="ignore"):
            bonus = np.sqrt(2.0 * np.log(t) / self.counts)
        bonus = np.where(self.counts > 0, bonus, 1e6)       # unplayed arms first
        return self.means() + bonus

    def choose(self) -> np.ndarray:
        return self._argmax_random_ties(self.index())


class Thompson(Learner):
    name = "Thompson"
    prior_mean, prior_var, noise_var = 0.5, 1.0, 0.25

    def posterior(self):
        prec = 1.0 / self.prior_var + self.counts / self.noise_var
        mean = (self.prior_mean / self.prior_var + self.sums / self.noise_var) / prec
        return mean, 1.0 / prec

    def greedy(self) -> np.ndarray:
        mean, _ = self.posterior()
        return self._argmax_random_ties(mean)

    def choose(self) -> np.ndarray:
        mean, var = self.posterior()
        sample = mean + np.sqrt(var) * self.rng.standard_normal(mean.shape)
        return self._argmax_random_ties(sample)


class EXP3(Learner):
    name = "EXP3"

    def __init__(self, *args, horizon_per_cell: float = 1.0, **kw):
        super().__init__(*args, horizon_per_cell=horizon_per_cell, **kw)
        K = self.Kc.astype(float)
        g = max(1.0, horizon_per_cell)
        self.gamma = np.minimum(1.0, np.sqrt(K * np.log(np.maximum(K, 2)) / ((np.e - 1.0) * g)))   # (C,)
        self.logw = np.zeros((self.B, self.C, self.Kmax))
        self.last_p = None

    def probs(self) -> np.ndarray:
        lw = self.logw + self.neg_inf
        lw = lw - lw.max(axis=-1, keepdims=True)
        w = np.exp(lw)
        w = w / w.sum(axis=-1, keepdims=True)
        gam = self.gamma[None, :, None]
        p = (1.0 - gam) * w + gam / self.Kc[None, :, None]
        return np.where(self.valid[None], p, 0.0)

    def greedy(self) -> np.ndarray:
        return self._argmax_random_ties(self.logw)

    def choose(self) -> np.ndarray:
        p = self.probs()
        self.last_p = p
        cum = np.cumsum(p, axis=-1)
        u = self.rng.random((self.B, self.C, 1)) * cum[..., -1:]
        return np.minimum((u > cum).sum(axis=-1), self.Kmax - 1)

    def update(self, b, c, arm, r, credit=None):
        super().update(b, c, arm, r)
        p = self.last_p[b, c, arm]
        xhat = r / p
        gam = self.gamma[c]
        np.add.at(self.logw, (b, c, arm), gam * xhat / self.Kc[c])


class ModelBased(UCB1):
    """Counterfactual-credit learner (see module docstring; despite the registry name it is UCB1 with
    credit for inert tells, not a model-based planner).  `credit` is a (N, Kmax) boolean mask of the arms
    to which the observed reward applies; it always contains the played arm."""
    name = "model-based"

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.k_sum = np.zeros(self.B)      # leak-rate estimate (logged only, never used to choose): sum of k_j^{(i)} over tells
        self.k_n = np.zeros(self.B)

    def update(self, b, c, arm, r, credit=None, credit_reward=None):
        """credit: (N, Kmax) mask of arms the observation applies to; credit_reward: (N, Kmax) reward
        each of those arms would have received (differs from r only by the saved telling cost)."""
        np.add.at(self.t, (b, c), 1.0)
        if credit is None:
            np.add.at(self.counts, (b, c, arm), 1.0)
            np.add.at(self.sums, (b, c, arm), r)
            return
        if credit_reward is None:
            credit_reward = np.repeat(r[:, None], self.Kmax, axis=1)
        bb = np.repeat(b, self.Kmax)
        cc = np.repeat(c, self.Kmax)
        aa = np.tile(np.arange(self.Kmax), len(b))
        w = credit.astype(float).ravel()
        np.add.at(self.counts, (bb, cc, aa), w)
        np.add.at(self.sums, (bb, cc, aa), w * credit_reward.ravel())

    def observe_leaks(self, b: np.ndarray, kij_told: np.ndarray):
        """kij_told: observed k_j^{(i)} for each (batch, tell) pair."""
        np.add.at(self.k_sum, b, kij_told)
        np.add.at(self.k_n, b, 1.0)

    def leak_rate_estimate(self, m: int) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.k_n > 0, self.k_sum / np.maximum(self.k_n, 1) / m, np.nan)


class ModelUCB(Learner):
    """Model-based learner: learns how leaks work and plans with that model.

    Per decision cell it fits the additive model  r + c_n |a| = mu_0 + sum_{j in a} beta_j k_j + noise,
    where k_j is the number of the questions told to friend j that came up in j's interview (observed
    feedback, 0 for friends not told) and c_n the known telling cost in reward units.  mu_0 is the value of
    keeping quiet, beta_j the value (to the teller) of one leaked question reaching friend j -- the role
    Delta plays in the game, passed through the hiring rule.  The coefficients are fitted by ridge least
    squares on EVERY round, whatever arm was played.  The leak rate p = P(a told question comes up) is
    estimated from all observed k_j / m (pooled over the run), and the learner plans with it: the predicted
    value of arm a is

        mu_0 + sum_{j in a} beta_j * m * p_hat - c_n |a|,

    played optimistically (LinUCB bonus sqrt(2 ln t * phi' A^-1 phi), which reduces to the UCB1 bonus for
    an arm that is a unit vector).  The bonus is computed with an optimistic leak rate (p_hat plus a
    Hoeffding radius, 1 before any tell is seen): with the plug-in p_hat alone, one tell that happens not to
    land gives p_hat = 0, hence no leak component in the arm's features, zero exploration bonus on beta and
    an arm that is never tried again.  The model is additive across friends, so it cannot represent that a
    second teller of the same friend is redundant (E0); like UCB1 it learns each cell's marginal values.
    It sees the same feedback as the counterfactual-credit learner (`kij`)."""
    name = "model-ucb"
    needs_model_info = True

    def __init__(self, *args, arm_bits: np.ndarray, cost: np.ndarray, m: int, ridge: float = 1.0, **kw):
        super().__init__(*args, **kw)
        self.arm_bits = arm_bits.astype(float)                  # (C, Kmax, L): friend j is told by arm a
        self.n_told = self.arm_bits.sum(axis=2)                 # (C, Kmax)
        self.cost = cost                                        # (C,) telling cost in reward units
        self.m = m
        self.L = arm_bits.shape[2]
        D = 1 + self.L
        self.Ainv = np.broadcast_to(np.eye(D) / ridge, (self.B, self.C, D, D)).copy()
        self.bvec = np.zeros((self.B, self.C, D))
        self.k_sum = np.zeros(self.B)                           # pooled leak-rate estimate
        self.k_n = np.zeros(self.B)

    def leak_rate_estimate(self, m: int | None = None) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.k_n > 0, self.k_sum / np.maximum(self.k_n, 1) / self.m, np.nan)

    def _p_hat(self) -> np.ndarray:
        """Plug-in leak rate (0.5 before any tell has been observed)."""
        return np.where(self.k_n > 0, self.k_sum / np.maximum(self.k_n, 1) / self.m, 0.5)

    def _p_optimistic(self) -> np.ndarray:
        """Upper confidence bound on the leak rate from the m * k_n observed question slots (1 if none)."""
        slots = self.m * self.k_n
        t_run = np.maximum(self.t.sum(axis=1), 2.0)
        radius = np.sqrt(2.0 * np.log(t_run) / np.maximum(slots, 1.0))
        return np.where(self.k_n > 0, np.minimum(1.0, self._p_hat() + radius), 1.0)

    def _features(self, p: np.ndarray | None = None) -> np.ndarray:
        """phi(a) = (1, m p * [friend j told by a]) for every arm: (B, C, Kmax, 1 + L); p defaults to p_hat."""
        p = self._p_hat() if p is None else p
        phi = np.ones(self.arm_bits.shape[:2] + (1 + self.L,))[None].repeat(self.B, axis=0)
        phi[..., 1:] = self.arm_bits[None] * (self.m * p)[:, None, None, None]
        return phi

    def _predicted_means(self, phi: np.ndarray) -> np.ndarray:
        theta = np.einsum("bcij,bcj->bci", self.Ainv, self.bvec)                              # (B, C, D)
        gross = np.einsum("bcki,bci->bck", phi, theta)
        return gross - self.cost[None, :, None] * self.n_told[None]

    def means(self) -> np.ndarray:
        return self._predicted_means(self._features())

    def choose(self) -> np.ndarray:
        mu = self._predicted_means(self._features())
        phi = self._features(self._p_optimistic())                  # exploration: uncertainty about beta AND p
        width = np.einsum("bcki,bcij,bckj->bck", phi, self.Ainv, phi)
        t = np.maximum(self.t, 1.0)[..., None]
        score = mu + np.sqrt(2.0 * np.log(t) * np.maximum(width, 0.0))
        return self._argmax_random_ties(score)

    def update(self, b, c, arm, r, feats=None):
        """feats: (N, L) observed k_j^{(i)} of the told friends (0 for friends not told)."""
        np.add.at(self.counts, (b, c, arm), 1.0)
        np.add.at(self.sums, (b, c, arm), r)
        np.add.at(self.t, (b, c), 1.0)
        gross = r + self.cost[c] * self.n_told[c, arm]
        x = np.concatenate([np.ones((len(b), 1)), feats], axis=1)                              # (N, D)
        # Sherman-Morrison; at most one update per (run, cell) per round, so plain indexing is safe
        Ai = self.Ainv[b, c]                                                                   # (N, D, D)
        Ax = np.einsum("nij,nj->ni", Ai, x)
        denom = 1.0 + np.einsum("ni,ni->n", x, Ax)
        self.Ainv[b, c] = Ai - np.einsum("ni,nj->nij", Ax, Ax) / denom[:, None, None]
        self.bvec[b, c] += x * gross[:, None]
        told = self.arm_bits[c, arm]                                                           # (N, L)
        np.add.at(self.k_sum, b, (feats * told).sum(axis=1))
        np.add.at(self.k_n, b, told.sum(axis=1))


LEARNERS = {
    "eps-greedy": EpsGreedy,
    "UCB1": UCB1,
    "Thompson": Thompson,
    "EXP3": EXP3,
    "model-based": ModelBased,
    "model-ucb": ModelUCB,
}
