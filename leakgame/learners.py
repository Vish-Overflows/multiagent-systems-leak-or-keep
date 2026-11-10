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
              deleting inert tells.  It also estimates the leak
              rate p = P(a shared question reappears).  Chooses with UCB1 on the
              augmented statistics; identical to UCB1 under E0 (nothing is inert).
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
    """Counterfactual-credit learner (see module docstring).  `credit` is a (N, Kmax) boolean
    mask of the arms to which the observed reward applies; it always contains the played arm."""
    name = "model-based"

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.k_sum = np.zeros(self.B)      # leak-rate estimate: sum of k_j^{(i)} over tells
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


LEARNERS = {
    "eps-greedy": EpsGreedy,
    "UCB1": UCB1,
    "Thompson": Thompson,
    "EXP3": EXP3,
    "model-based": ModelBased,
}
