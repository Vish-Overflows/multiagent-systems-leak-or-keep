# Phase 1 report: closing small gaps with data we already have

What was asked: do Phase 1 of `plan.md`. This file says, item by item, what I did, what the result was, what I
changed in the repository and in the report, what went wrong on the way, and how you can check each claim yourself.

**Result in one line:** all six items are done on branch `phase-1` (not merged into `main`; that is yours to review).
Four predictions held, one was **refuted** (1.5c), and one was **ill posed** (the company-value half of 1.4). In
both cases the report now says what was measured, not what I had guessed. Unit tests went from 40 to 42.

Commits on `phase-1`: `8adc1f5` (all of Phase 1), `6ceb6b6` (status section in `plan.md`), `4c70827` (the clean-clone
comparison tool; see `report_p0.md`).

---

## 1. Summary table

| Item | Prediction written beforehand | Outcome |
|---|---|---|
| 1.1 CCE-gap test per setting | significant in all 24 E1/E2 learner×setting pairs | **holds**: 36 of 36 significant (24 of 24 in E1/E2) |
| 1.2 Rounds to 90 % for E0 and E2 | (no prediction; a table) | done, appendix Table 5 |
| 1.3 H1 without using the own loss | P(last seat) increases weak < average < strong, the order of α\* | **holds**: 3.6 % / 17.9 % / 28.6 % |
| 1.4 Baselines | no learner below random sharing in company value or welfare | **welfare holds; company value ill posed**: under a cut-off every learner *and the equilibrium itself* is below random sharing |
| 1.5 Four "plausible, not tested" explanations | (a) average types hardest, (b) UCB1 explores most, (c) ε-greedy's mistakes are cheap, (d) pooling helps in 4-arm cells | (a) **holds**, (b) **holds**, (c) **REFUTED**, (d) **holds, modest** |
| 1.6 Bias of the counterfactual credit in E2 | \|bias\| < half the smallest payoff gap | **holds**: at most 0.08 pp (22 % of the gap) |

---

## 2. Item 1.1: the CCE-gap significance test

**Problem.** `analyze_x3.py` tested "the CCE gap falls" by pooling 180 runs per learner across six different games.
Runs that share a seed share their random draws, so they are not independent, and the six games are different
populations.

**What I did.** Added a second family of tests: for every learner × leak model × rule (36 settings), a paired
sign-flip test of gap at 10,000 rounds vs gap at 200,000 rounds, with the **30 seeds as the unit**, Holm-corrected over
all 36. The old pooled test is kept unchanged (a regression check confirmed that every other key of
`x3_summary.json` is identical to before; only the new key `tests.cce_gap_falls_by_setting` was added).

**Prediction.** Significant in all 24 E1/E2 pairs; refuted if any one is not.

**Result.** 36 of 36 significant after Holm (24 of 24 in E1/E2). The gap falls for every learner in every setting.

**Changes.** `experiments/analyze_x3.py` (new test block and console digest), `experiments/make_figures.py` (macros
`GapFallSetTests/Sig/...`), `report/main.tex` (§5.3 CCE-gap paragraph and the H4 row of Table 4 now quote the
per-setting counts instead of "6 of 6 tests"), `results/x3/x3_summary.json`, `results/logs/x3_analysis.log`.

---

## 3. Item 1.2: rounds to 90 % for all leak models

**What I did.** Table 3 shows rounds-to-90 % for E1 only, although all three leak models were already computed.
`make_figures.py` now writes `report/generated/tab_rounds90.tex`, shown as **Table 5 in a new Appendix A**. Table 3's
caption points to it.

**Worth knowing from the table.** In E0 under a cut-off almost no learner reaches 90 % ("never (3 %)" for UCB1 and EXP3,
"never (0 %)" for Model-based UCB). That is the E0 coordination problem already explained in the report, not a
learning failure.

---

## 4. Item 1.3: H1 without circularity

**Problem.** H1 says α\* is highest for the types most likely to be *pivotal*, but the report measured "pivotal" with
the own-loss ℓ, and α\* = (ℓ + c)/G. With G roughly constant, the ordering of α\* follows ℓ by construction.

**Idea.** A tell costs the teller her seat only if she holds the **last** seat (rank K) when a boosted friend overtakes
her. How often each type holds rank K when nobody shares can be measured without any sharing at all.

**What I did.**
- `leakgame/analysis.py`: new `pivotality_by_type(cfg, n_days, seed)`. It simulates days with no leaks (all candidates
  exchangeable, so they are pooled), ranks the scores, and returns P(rank = K), P(rank = K+1) and P(hired) per type,
  with standard errors.
- `experiments/run_h1.py` (2,000,000 days, 1 second) writes `results/h1/pivotality.json`.
- Unit test `test_pivotality_by_type_is_a_valid_distribution`: the mean over types of P(rank = K) must equal 1/n and the
  mean hire rate K/n, and the order must be weak < average < strong.

**Prediction.** P(rank = K) increases weak < average < strong; H1 refuted if the order differs from α\*.

**Result** (2 million days, standard errors ≤ 0.02 points):

| | weak | average | strong |
|---|---|---|---|
| P(holds the last seat) | **3.6 %** | **17.9 %** | **28.6 %** |
| P(first place outside) | 9.0 % | 23.6 % | 17.4 % |
| P(hired) | 4.5 % | 26.3 % | 69.2 % |
| α\* (from the report) | 0.10 | 0.32 | 0.38 |

The order matches. The sanity checks pass: the mean hire rate is 33.3 % = K/n. (My plan guessed "average ≈ strong";
strong is clearly the most pivotal.)

**Changes.** `report/main.tex`: one sentence in the "Why" paragraph of §5.1 and the evidence column of the H1 row in
Table 4, now citing the independent measurement.

---

## 4b. A note on how the measurement was done
All Phase 1 experiments are deterministic (fixed seeds) and reuse the payoff tables already cached in
`results/tables/`. None needed a new simulation of the payoff tables; the only new simulations are the 2 M and 4 M day
runs of 1.3 and 1.6 and the 400 k day reward simulation of 1.5a.

---

## 5. Item 1.4: baselines

The proposal promised five baselines (nobody shares, everybody shares, **random sharing**, the equilibrium, a
leak-free company). Random sharing never appeared, and learners were never compared with any baseline.

### What I did
- `leakgame/analysis.py`: `profile_values(game, sig)` and `matrix_distribution_values(game, w)` give, for a strategy
  profile or for a distribution over sharing matrices: **welfare** (sum of the six candidates' expected utilities),
  **company value** E[V], and the share of possible tells. "Random sharing" is the uniform distribution over all
  matrices (each possible tell independently with probability ½); "nobody shares" is the leak-free company.
- `experiments/run_baselines.py` evaluates the four baselines and the profile each learner settled on (most-played
  arm of every decision cell, averaged over 30 seeds; E1 and E2, both rules, two friend groups, α = 0.2). It also
  computes each learner's realised utility over the last half of the run from the stored learning curves.
- **X4 extended.** `experiments/run_x4.py` now also records `V_random`, `Lambda_random` and `Lambda_random_rel`. I
  re-ran X4 (14.5 min). Regression check: **every field that existed before is identical**; only three fields were
  added.
- Appendix Table 6 (`report/generated/tab_baselines.tex`) and a new paragraph in §5.3 ("What the settled play is worth").
- Unit test `test_baseline_values_match_the_tables`.

### Results (E1, two friend groups, α = 0.2; E2 is the same within about 0.006)

| Pattern | Seats: tells | Seats: company value | Cut-off: tells | Cut-off: welfare | Cut-off: company value |
|---|---|---|---|---|---|
| Nobody shares (leak-free company) | 0 % | 0.657 | 0 % | 2.799 | 0.894 |
| Random sharing | 50 % | 0.631 | 50 % | 3.067 | 0.833 |
| Everybody shares | 100 % | 0.614 | 100 % | 3.317 | 0.771 |
| **Equilibrium** | 33 % | 0.645 | 100 % | 3.317 | 0.771 |
| ε-greedy (settled) | 43 % | 0.638 | 87 % | 3.256 | 0.787 |
| UCB1 / Thompson (settled) | 38 % | 0.643 | 99 % | 3.314 | 0.772 |
| EXP3 (settled) | 41 % | 0.639 | 94 % | 3.288 | 0.778 |
| Leak-aware UCB / Model-based UCB | 33–34 % | 0.645 | 100 % | 3.317 | 0.771 |

Company loss at Q = 15 (X4, E2): random sharing costs **3.6 %** (seats) and **6.5 %** (cut-off); the equilibrium
costs 1.7 % and 13.1 %.

### The prediction, and why it failed in one half
I had written: "every learner's settled profile is worth more than random sharing and close to the equilibrium; refuted
if any learner falls below random sharing in company value or welfare."
- **Welfare** (what the players optimise): no learner is below random sharing in any of the 24 comparisons. Holds.
- **Company value under a cut-off:** all six learners are below random sharing, **and so is the equilibrium itself**
  (0.771 vs 0.833), because everybody telling hurts the company. Read literally the prediction is refuted, but for a
  reason that has nothing to do with learning: the company is not a player. The right criterion for the company was
  "close to the equilibrium", and that holds (learners lie within 0.016 of it). I recorded this in the script's
  docstring and stored the two checks separately in the JSON. The report says the same.

### Something the baselines showed
Under fixed seats exactly K candidates are hired whatever anyone shares, so welfare is the same for every pattern
(2.794–2.800; the spread is just the message cost of the tells). The only informative comparison there is company value, and the report and the
Table 6 caption say so.

---

## 6. Item 1.5: testing four explanations

The report contained four sentences of the form "a plausible reason, which we did not test". One script,
`experiments/run_explanations.py` (about 1 minute; data: the stored `x3_*.npz`, `x3_summary.json`, and a 400,000-day
reward simulation at the equilibrium), tests them. Predictions were written in its docstring first.

| | Claim | Test | Result |
|---|---|---|---|
| (a) | average types learn slowest because weak types' rewards are less noisy | reward sd of every decision cell divided by its payoff gap; (sd/gap)² is what a bandit must average away; rank-correlate with the final error rate over the 12 cells | **holds.** Reward sd 0.15 (weak) vs 0.30 / 0.31 (average / strong). (sd/gap)² = 1,474 / **2,934** / 1,730 under fixed seats; average is the largest in both rules. Spearman ρ = 0.80 (UCB1) and 0.70 (Thompson) under fixed seats |
| (b) | UCB1 has high regret because its bonus keeps it exploring | share of pulls on non-equilibrium rows | **holds.** UCB1 56 % / 48 % (seats / cut-off) vs Thompson 41 % / 23 % and ε-greedy 37 % / 23 %. The regret implied by the pull counts matches the stored regret within 6 % |
| (c) | ε-greedy's mistakes are on decisions where the wrong row costs little | mean payoff gap of the cells where it ends wrong ÷ mean gap over all cells; prediction < 1 | **REFUTED.** The ratio is 1.01 (seats) and 1.00 (cut-off). Its low regret is not because its mistakes are cheap but because it pulls wrong rows less often; agreement only looks at the row that ends up on top |
| (d) | Model-based UCB gains from pooling across rows that share a friend | relative regret reduction against UCB1 in 4-row cells (candidates with two later friends) vs 2-row cells | **holds, modest.** Larger in the 4-row cells in 4 of 4 settings (34 % vs 25 %, 86 % vs 81 %, 30 % vs 22 %, 80 % vs 77 %); the leak-aware learner shows no such pattern. The 4-row cells differ in other ways too |
| (e) | E0's wide CCE interval comes from redundant tells | — | **not done**; this is Phase 3.1 |

**A caveat I found while doing this.** The stored `final_counts` of the leak-aware learner (`model-based`) include
*credited* observations: a tell that did not land also increments the count of the arm without it. So they are not
pull counts. I excluded that learner from (b) and (c). I also checked whether this affects earlier claims that use
"the settled profile" of that learner: the profile from the credited counts equals the profile from the greedy means in
**100 % of runs** in all four E1/E2 settings, so the report's "97 % / 100 % exact equilibrium" and the Table 6 row are
unaffected. The caveat is written into the `learners.py` docstring and the README.

**Report changes.** Three sentences in §5.3 were replaced by what was measured: the regret paragraph (including that
our guess (c) was wrong), the "average types are slowest" sentence, and the Model-based UCB pooling sentence.

---

## 7. Item 1.6: bias of the counterfactual credit in E2

**Problem.** The report said the credit is "very slightly biased in E2; we did not measure by how much".

**What I did.** `experiments/run_e2_bias.py` (4 million days per configuration, 25 s). Candidate 1 chooses between
telling candidate 5 and silence, while candidate 3 tells candidate 5 (the worst case, because 5 then has a second
teller). The learner credits the silent arm with every day where the told question sets are disjoint. The bias is
E[u_silent | disjoint] − E[u_silent], per type and rule. A control with no second teller must show no bias.

**Result.** The largest |bias| with a second teller is **0.080 pp** (strong type, cut-off), about two standard errors
(≈ 0.04 pp), which is **22 % of the smallest payoff gap** (0.36 pp). The control scatters about as much (up to 0.055 pp).
So no bias is distinguishable from zero, but I cannot show it is zero either; the resolution is about a tenth of the
signal. The report now says exactly that.

**Changes.** `report/main.tex` §4.3 (the Leak-aware UCB bullet), `leakgame/learners.py` docstring, README.

---

## 8. Other changes made along the way

| Change | Where |
|---|---|
| New **Appendix A** with two tables (rounds to 90 %; baselines) | `report/main.tex` (after the references), `report/generated/tab_rounds90.tex`, `tab_baselines.tex` |
| `\label{sec:cost}` on the RQ4 subsection, so a cross-reference is not typed by hand | `report/main.tex` |
| Report text: "42 unit tests cover every module: …, the pivotality and baseline calculations" | `report/main.tex` §4 |
| README: four new rows in the reproduction table; new results paragraphs; **two stale statements fixed** (it still said the model-based learner "was not built" and that the E2 bias was "not measured") | `README.md` |
| `run_all.sh`: runs `run_h1.py`, `run_baselines.py`, `run_explanations.py`, `run_e2_bias.py` | `run_all.sh` |
| 2 new unit tests (42 in total) | `tests/test_leakgame.py` |
| macros: 709 → 780 (every number in the new text comes from a macro) | `report/generated/numbers.tex`, `results/numbers.json` |
| new result folders and logs | `results/h1/`, `results/baselines/`, `results/explanations/`, `results/e2_bias/`, `results/logs/*.log` |
| PDF rebuilt (now 11 pages) | `report/main.pdf` |

Figures were *not* changed: `make_figures.py` re-renders all of them, but their data did not change, so I restored the
committed versions rather than commit matplotlib rendering noise.

---

## 9. Problems I hit and fixed

- **Appendix inside the bibliography.** My first appendix was placed inside the two-column `multicols` group that holds
  the references. LaTeX silently drops floats there, so the tables vanished and their labels were undefined (`??` in the
  PDF, two overfull boxes). I found it in the build log and moved the appendix after the references.
- **A wrong prediction in 1.4** (above) and **a wrong guess in 1.5(c)** (above): both are reported as they came out.
- **Counts that are not pulls** (1.5): found because my pull-count cost accounting reproduced the stored regret within
  3–6 % for every learner except one, which was off by 56 % and 10×.
- **Small slips:** a wrong dictionary key in `run_h1.py` (`se_P_rank_K`), a report anchor that assumed a line break that
  was not there, and a macro-patching helper that lost backslashes. All were caught by running the code or checking
  the generated files before committing.

---

## 10. What I did not do

- Item **1.5(e)** (why E0's CCE interval is wide) is Phase 3.1 in the plan.
- I did not merge `phase-1` into `main` and did not push anything.
- The "Phase 1" experiments are not part of the clean-clone run of Phase 0 (that run was of `main`, before Phase 1).
  Their scripts are in `run_all.sh`, so the final reproduction run of Phase 6 will cover them.

---

## 11. How to check it yourself

```bash
python -m pytest -q                              # 42 pass
python experiments/run_h1.py                      # 1 s,   prints the 3.6 / 17.9 / 28.6 % table
python experiments/run_baselines.py               # 1 min, needs results/x3
python experiments/run_explanations.py            # 1 min, needs results/x3
python experiments/run_e2_bias.py                 # 30 s
python experiments/analyze_x3.py                  # 2 min, prints "36 significant; E1/E2: 24 of 24"
python experiments/make_figures.py                # 5 s,   regenerates tables and the 780 macros
cd report && ./tectonic.exe main.tex              # 11 pages
```

Where to read the results: `results/h1/pivotality.json`, `results/baselines/baselines.json`,
`results/explanations/explanations.json`, `results/e2_bias/e2_bias.json`, and the console outputs in
`results/logs/` (`h1.log`, `baselines.log`, `explanations.log`, `e2_bias.log`, `x3_analysis.log`). In the PDF: §5.1 (the
"Why" paragraph), §5.3 (four paragraphs), §5.4 (random sharing), Table 4 (H1 and H4 rows), Appendix A.
