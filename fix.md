# Fix plan: items 5–7 of the review

## Status (what was actually done)

All three items are implemented and their results are in the repository. Where the outcome differed from what the
plan expected, or the plan was changed, it is listed here; the plan below is left as written.

| Item | Done | Result |
|---|---|---|
| 5a CCE programs for E0 and E2 | `experiments/run_cce.py`; 6 cells (about 78 min with two processes) | E1 and E2: one sharing level over all CCEs in 212 of 212 games. **E0: only 55 of 106**; in 48 of them the CCE interval is wider than the range spanned by the pure equilibria (up to 42 points; two groups, α = 0.2, cut-off: pure equilibria 67%, CCE 66% to 99%). The report's old sentence "no mixed or correlated equilibrium shares more or less" held only for E1/E2 and is restricted accordingly. Every pure equilibrium lies in its CCE interval and the rebuilt tables reproduce the X1 equilibria (both asserted per row). |
| 5b Support enumeration | `leakgame/support_enum.py`, `experiments/run_support.py` | The plan expected "most likely none found". **Three mixed equilibria were found** (pairs, E0, fixed seats, α = 0.05, 0.2, 0.325), one per game, next to two pure ones; each lies inside its CCE interval. Dominance reduction, added beyond the plan, solves 311 of 378 games on all three networks exactly. |
| 6 Fairness bootstrap | `PayoffTable.Pxx_blocks`, `resample_tables`, `with_Px(..., Pxx=)`; X2 rerun | Only 2 of 44 X2 records changed, both fairness games under a cut-off and only in their bootstrap fields (FAIR(0.5, 0.25): 49.5% → 52.0% exact; mixed FAIR/FRIEND: 99% → 100%). The understatement was real but small; no verdict changed. |
| 7a Honest naming | docstrings, report, README | The leak-aware learner is described as UCB1 with counterfactual credit; its logged leak-rate estimate is said not to be used. |
| 7b Model-based learner | `ModelUCB` in `leakgame/learners.py`; X3 run for it; `analyze_x3.py` has six learners | Reaches 90% agreement in 12,000 / 4,500 rounds (E1, seats / cut-off) against 27,000 / 7,000 for leak-aware UCB and 56,000 / 18,000 for UCB1, and lowers regret against UCB1 in all four E1/E2 settings. H4's "faster" half is now supported by a real model-based learner. |

Differences from the plan:
- The pairwise block table is 112 MB for two groups, not 56 MB.
- Item 7b: a unit test caught a flaw in the first version of the learner (one tell that happens not to land gave
  p̂ = 0, zero exploration bonus, and the arm was never tried again in 4 of 20 runs). The bonus now uses an optimistic
  leak rate. The first, flawed run was discarded and redone.
- Adding a sixth learner enlarges two Holm families (BAR vs SEATS, CCE-gap tests), so their corrected p-values
  changed (raw p-values and all per-learner cells did not; cells of the five existing learners agree to 1e-14).
- Unit tests went from 27 to 40, covering the learning loop, replicator dynamics, the sweep helper, support enumeration
  and the new learner, so the report's "every module is tested" is now true.
- The environment is a `uv` virtual environment (`uv venv .venv`; Python 3.10, NumPy 2.2, SciPy 1.15).

Not done:
- **`report/main.pdf` is not rebuilt.** There is no LaTeX on this machine. I tried to download the official tectonic
  release into `.tools/` (gitignored, as `run_all.sh` expects) and the download was refused, so I stopped. Build it with
  `cd report && tectonic main.tex` or on Overleaf, then check the page count (it was 8 pages and the text has grown).
- Support enumeration for two friend groups and star (agents with four or eight actions).

Also changed: the Contribution table's status column (Chaitanya: "support enumeration on the pairs network only";
Vishnu: "model-based learner in reduced form"). Check that you are happy with how it credits these tasks.

---

Items 1–4 and 8 are already done:
- `numpy>=2.0` in `pyproject.toml`.
- X4 headline reworded.
- Regret added to Table 3, and the CCE wording softened.
- H2/H4 verdicts judged by the proposal's refutation criteria.
- Near-equilibrium counts reported.

This file covers the three items that need new code and new runs:

| # | Item | New runs | Wall time (10-core laptop) |
|---|------|----------|----------------------------|
| 5a | CCE linear programs for E0 and E2 | 6 (network, env) cells | about 65 min |
| 5b | Support enumeration on the pairs network | 6 envs × rules, small | under 5 min |
| 6 | Fairness bootstrap resamples the pairwise table | X2 | about 10 min |
| 7 | Leak-aware learner: honest naming (7a), optional real model (7b) | 7a: none; 7b: one learner in X3 | 7a: 0; 7b: about 6 min |

---

## 0. Prerequisites (once)

1. Create a clean environment. The system Python has NumPy 1.26, which cannot run the code.
   ```
   python -m venv .venv
   .venv\Scripts\pip install -e ".[dev]"     (Windows)
   .venv/bin/pip install -e ".[dev]"         (Linux/macOS)
   ```
2. Run `python -m pytest -q`. All 27 tests must pass before anything else changes.
3. Payoff tables are not committed (`results/tables/`, 263 MB). They rebuild automatically, 5–40 s each. 5a and 5b reuse the X1 tables (same config, seed 0, 2,000 days), so they come out bit-identical to the ones behind the committed results.
4. Work on a branch (e.g. `fix/items-5-7`). Commit each item separately so any one can be reverted.

---

## 5a. CCE linear programs for E0 and E2

**Why.** The CCE result ("max and min sharing over all CCEs coincide in 106 of 106 games") exists only for E1. E0 is where all 63 multi-equilibrium games are. It's also where a mixed or correlated equilibrium could share a different amount than the pure ones. That makes it the most informative place to run the LP, and it's currently missing.

**Code**
- New script `experiments/run_cce.py`. Don't rerun `run_x1.py`: it skips finished cells and redoing everything costs about 40 min.
  - For each network in (two_groups, pairs, star) and each env in (E0, E2): build the table with `build_table(GameConfig(network, env), n_days=2000, seed=0)`. The settings match X1, so the cached table is reused.
  - For each rule and α on the same grids as E1 (step 0.05 for two_groups, 0.025 for pairs and star), call `cce_lp(game, objectives=("max_sharing", "min_sharing"))`.
  - Write `results/x1/x1_cce_<network>_<env>.json` with the same row format as the `"cce"` block in `x1_<network>_E1.json`.
  - Run the six cells in a process pool, as `run_x1.py` does.
- Memory: the two-groups LP builds a dense 144 × 262,144 constraint matrix (about 300 MB). Cap the pool at about 4 processes if RAM is under 16 GB.
- `experiments/make_figures.py`, `numbers_x1`:
  - Also load the new files.
  - Report `CceSingle`/`CceTot` separately for E0, E1 and E2.
  - For E0, add the largest width (max − min sharing over CCEs) and where it occurs.
  - New macros, e.g. `CceTotEzero`, `CceSingleEzero`, `CceMaxWidthEzero`, and the same for E2.

**Expected time.** The E1 runs took 18 min (two groups) + 13 min (star) + about 0 (pairs). E0 and E2 together should be about 2 × 31 min, run in parallel.

**Report changes**
- §5.1 "How solid…": replace "coincide in 106 of 106 games" with per-environment counts.
- If E0 widths are non-zero, say so plainly: correlated play could share between X% and Y%, and the pure equilibria cover Z–W% of that range. That would be a real finding for E0, not a failure.
- Deviations list, (iv): delete "the CCE programs were run for E1 only".

**Check.** Every pure equilibrium's sharing rate must lie inside the matching [min, max] CCE interval. Add this as an assertion in `run_cce.py`, because every BNE is a CCE.

---

## 5b. Support enumeration on the pairs network

**Why.** Replicator dynamics can't find unstable mixed equilibria, so "no mixed equilibria" is currently backed only by the CCE bound. The proposal promised support enumeration. The pairs network is small enough to do it properly:
- 3 active candidates (1, 3, 5), each with one later friend.
- That gives 9 (candidate, type) agents with 2 actions each.
- 512 pure profiles in total.

Two groups (4-action agents) and star (an 8-action hub) are too big to enumerate without pruning. They stay covered by the CCE bound only, and that limit gets stated.

**Method** (new module `leakgame/support_enum.py`)
1. **Iterated strict dominance**, exact. An action of agent c is strictly dominated if it's worse than the agent's other action against *every* pure profile of the other agents. Use the interim value tables `game.V`. Payoffs are multilinear in the others' mixed strategies, so domination against all pure profiles implies domination against all mixed ones. Remove dominated actions and repeat.
2. **Enumerate supports** over the surviving agents: each keeps {0}, {1} or {0,1}. That's at most 3^9 = 19,683 support profiles.
3. **Solve the indifference conditions** for the agents that mix (k of them):
   - k = 0: a pure profile. It is already checked by `enumerate_pure_bne`; use this as a consistency check.
   - k = 1: the agent's own probability doesn't enter its indifference condition. Accept only if the agent is exactly indifferent given the others' pure actions; record it as a continuum.
   - k = 2: each condition is linear in the *other* agent's probability, so solve the bilinear system in closed form.
   - k ≥ 3: multilinear system. Use `scipy.optimize.least_squares` with bounds [0, 1] from 50 random starts, then polish and deduplicate. Results from this step are labelled "found by numerical search", not "complete".
4. **Verify** every candidate: (a) all probabilities in (0, 1); (b) no profitable deviation to an action outside the support, using `_mixed_payoffs` from `solvers.py`, which already computes this.

**Runs.** Pairs network, E0/E1/E2 × SEATS/BAR × the X1 α grid. Output goes to `results/x1/x1_support_pairs.json`: per game, the number of mixed equilibria found, their sharing rates, and whether the k ≥ 3 search was needed.

**Tests** (`tests/test_leakgame.py`)
- A 2-agent game with known mixed equilibrium (a matching-pennies payoff oracle) returns p = 0.5.
- On the pairs table, the k = 0 supports reproduce exactly the profiles from `enumerate_pure_bne`.

**Report.** One sentence in §5.1. Either "on the pairs network, support enumeration finds no mixed equilibria in N of M games", or the list of games where it finds some, with their sharing compared to the CCE interval. Deviation (iii) becomes "support enumeration on the pairs network only".

**Risk.** Mixed equilibria with exact indifference are rare in an estimated table. The most likely result is "none found", which is still worth stating. If the k ≥ 3 search finds some, they must lie inside the CCE interval (5a check); a solution outside it means a bug.

---

## 6. Fairness bootstrap resamples the pairwise table

**Problem.** `AgentFormGame.with_Px()` swaps in a resampled P(x_j) table, but `_expected_utilities` still reads the original P(x_i x_j). Fairness utilities depend on the pairwise term, so their bootstrap understates uncertainty. This affects the 8 fairness rows of Table 2 and the "half FAIR / half FRIEND" population. Selfish, friendly and clannish utilities don't use the pairwise table, so they're unaffected.

**Code**
1. `leakgame/tables.py`
   - Also store block means of the pairwise table: `Pxx_blocks`, shape (T, M, 2, n, n, nblk). Only the n(n−1)/2 = 15 upper-triangle pairs are needed, which is about 56 MB float32 for two groups.
   - Bump the cache version `"v": 4` → `"v": 5` so old cached tables aren't reused.
   - `resample_blocks(rng)` returns `(Px, Pxx)` built from the **same** `pick` of blocks. This matters: the two tables must be resampled together, with the same days.
2. `leakgame/game.py`: `with_Px(Px, Pxx=None, build_V=False)` passes `Pxx` into `_expected_utilities`, which uses it instead of `self.table.Pxx` when given.
3. `leakgame/solvers.py`: `bootstrap_exploitability` and `bootstrap_bne_set` unpack both tables.
4. `run_x4.py` uses `resample_V_blocks` (company value) and doesn't change.

**Draw the same random numbers as before.** `resample_blocks` must still call `rng.integers` exactly once per replicate. Then every non-fairness bootstrap result (X1, and the friendly and clannish rows of X2) is bit-identical to the committed results. That's the regression test: rerun X2 and diff `results/x2/x2.json`. Only records whose utilities have a pairwise term may change.

**Runs.** `python experiments/run_x2.py` (about 10 min), then `make_figures.py`. X1, X3 and X4 don't need rerunning.

**Tests**
- With all blocks picked once each (identity resample), `resample_blocks` reproduces `Px` and `Pxx` exactly.
- For a fairness game, the bootstrap ε samples differ from a run that keeps `Pxx` fixed.
- For a friendly game, they are identical.

**Report.**
- Table 2 daggers are generated automatically from the bootstrap fraction.
- If more fairness rows become fragile, soften "Fairness-minded candidates behave the other way round" (§5.2). The FAIR(0.5, 0.25) cut-off row is already at 49.5%; if the strong-types-tell result now rests only on fragile rows, the text must say so.
- Add one line to the method: "the bootstrap resamples both tables jointly".

---

## 7. Leak-aware learner

**Problem.** The learner's leak-rate estimate p̂ is computed (`ModelBased.observe_leaks`) but never used to choose actions. The learner doesn't estimate Δ and doesn't best-respond. It is UCB1 plus counterfactual credit for tells that didn't land. On top of that:
- The report says "it recovers the leak rate", which is true but beside the point.
- The proposal's "model-based" label is still used in Table 4 and in the code.

### 7a. Honest description (recommended, no reruns)

- `leakgame/learners.py`: in the docstring, say that p̂ is logged for reporting only and is not used in decisions. Keep the registry key `"model-based"` so the result files (`x3_model-based.npz`) and the analysis keys stay valid. Add a comment explaining that.
- `report/main.tex`
  - Algorithm 1, line 5: mark p̂ as "(logged, not used for decisions)", or delete the line.
  - §5.3: replace "and it recovers the leak rate: p̂ = … (truth m/Q = 0.2)" with a neutral clause, or remove it.
  - §4, the leak-aware bullet: add one sentence on E2. There, the extra samples come from days when the told and received question sets were disjoint. That condition slightly changes the other leaks, so the credit is exact day by day but not exactly unbiased in distribution. In E1 it is exactly unbiased. Agreement is 100% in E2 anyway, so this is a stated limit, not a change to the results.
  - Table 4, H4: "model-based learners are faster" → evidence says "leak-aware (not model-based) learner".
- README: the same two wording changes.

### 7b. A real model-based learner (optional, only if time allows)

This uses p̂ and an estimated effect per leaked question, i.e. it learns the environment and plans with it, as the proposal intended.

- **Model**, per (candidate, type): the expected reward of arm a is

  μ(a) ≈ μ₀ + Σ_{j ∈ a} β_j · E[k_j^{(i)} | told] − c·|a|,  with E[k_j^{(i)} | told] = m·p̂.

  - μ₀ is the silent arm's mean.
  - β_j is the value to the teller of one leaked question reaching friend j. It plays the role of Δ, folded through the hiring rule.
  - β_j is estimated by least squares of the reward on the observed k_j^{(i)}, pooled over all arms that include j.
  - Choose arms with a LinUCB-style bonus on (μ₀, β).
  - The model is additive, so it ignores interactions between friends (e.g. the redundant second tell in E0). Say so, and keep the E0 comparison.
- **Code.** New class `ModelUCB` in `learners.py`. Register it as `"model-ucb"`. It gets the same feedback (`kij`) as the leak-aware learner, so `learning.py` needs only the `is_mb` branch generalised.
- **Runs.** `python experiments/run_x3.py model-ucb` (about 5 min: 180 runs, like the leak-aware rerun log). Then add `"model-ucb"` to `LEARNERS` in `analyze_x3.py`, rerun it, then run `make_figures.py`. The other learners' result files are untouched. All learners share `env_seed`, so the new learner's runs are paired with the existing ones.
- **Tests.** In a synthetic additive environment, `ModelUCB` finds the best arm and its p̂ converges.
- **Report.** Add it as a sixth row in Table 3 and Figure 4, and judge H4's "model-based learners are faster" against *this* learner. If it isn't faster than UCB1, H4's second half is refuted, and the report must say so.

---

## Order of execution and final regeneration

1. §0 prerequisites, then commit.
2. Item 6 (touches the shared table code; run tests and diff X2), then commit.
3. Item 5a and 5b (both read the tables; 5a's check also validates 5b), then commit.
4. Item 7a, then commit. Item 7b, if done, as a separate commit.
5. `python experiments/make_figures.py`
   - Check that `git diff report/generated/numbers.tex` changes only the macros these items touch.
   - Figures will redraw. Keep the redrawn PNG/PDF only for files whose data changed (Figure 4 if 7b is done); `git checkout` the rest to avoid matplotlib-version noise.
6. `python experiments/make_console_shots.py` if any logs changed.
7. Rebuild the PDF: `cd report && tectonic main.tex`, or `pdflatex main` twice, or Overleaf. No LaTeX is installed on this machine.
   - Check the page count. The report was 8 pages, and items 1–4 and 8 already added text. Trim if there is a page limit.
8. Update the README's "Known limits and deviations" list to match.

## Done when

- [ ] `pytest` passes, with the new tests, in a NumPy 2 environment.
- [ ] `results/x1/x1_cce_*_E0.json` and `*_E2.json` exist. Every pure equilibrium lies inside its CCE interval.
- [ ] `results/x1/x1_support_pairs.json` exists. Its k = 0 supports equal the enumerated pure equilibria.
- [ ] `results/x2/x2.json` differs from the committed version only in records with fairness utilities.
- [ ] The report and README no longer claim the learner "recovers the leak rate" or is "model-based" (7a), or a real model-based learner is reported against H4 (7b).
- [ ] `report/main.pdf` is rebuilt, and the deviations list matches what was actually done.
