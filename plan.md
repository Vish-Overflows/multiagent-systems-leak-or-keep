# Leak or Keep? Plan for the next phases

This plan says what to do next, in what order, and how we will know each step is done. It is written for the
four of us. You should be able to pick up any single item and work on it without reading the rest.

- **Where we are:** branch `fix/items-5-7`, 6 commits ahead of `main`, 40 unit tests passing. The review items
  1–8 are done; `fix.md` records exactly what was done.
- **What is open:** the stretch target `n = 8`, robustness of every headline number, E0 (where correlated play
  differs a lot from the pure equilibria), several explanations the report labels "plausible, not tested", and
  the final write-up.
- **What this plan is not:** a schedule with dates. We never fixed a deadline in this repository, so effort is
  given in rough working days and compute time in measured minutes. Put the dates in once we know the deadline
  (see [Decisions we need to make](#decisions-we-need-to-make)).

---

## Contents

1. [How to read this plan](#how-to-read-this-plan)
2. [Ground rules for every item](#ground-rules-for-every-item)
3. [Phase map at a glance](#phase-map-at-a-glance)
4. [Phase 0 — Close out the current work](#phase-0--close-out-the-current-work)
5. [Phase 1 — Quick wins on data we already have](#phase-1--quick-wins-on-data-we-already-have)
6. [Phase 2 — Robustness of the headline results](#phase-2--robustness-of-the-headline-results)
7. [Phase 3 — Understand E0](#phase-3--understand-e0)
8. [Phase 4 — Extend the scope (n = 8, bigger agents, more learning settings)](#phase-4--extend-the-scope)
9. [Phase 5 — Optional extensions](#phase-5--optional-extensions)
10. [Phase 6 — Final write-up and submission](#phase-6--final-write-up-and-submission)
11. [Decisions we need to make](#decisions-we-need-to-make)
12. [Risks and how we handle them](#risks-and-how-we-handle-them)
13. [Reference: measured run times and sizes](#reference-measured-run-times-and-sizes)

---

## How to read this plan

Every item has the same parts:

| Part | What it tells you |
|---|---|
| **Why** | the gap in the current work it closes |
| **What** | the exact change or experiment |
| **How** | files, functions and steps |
| **Prediction / refuted if** | for experiments: what we expect, written *before* running, and what result would prove us wrong |
| **Cost** | your working time, and compute time measured on the 16 GB, 24-thread laptop we used |
| **Output** | files it writes |
| **Report impact** | which section, table or figure changes |
| **Done when** | a checklist you can tick |
| **Owner** | a *suggestion*, following the split in the proposal (Jash: formulation, X2, X4, writing; Bhavay: simulator, tables, checks; Chaitanya: solvers, X1; Vishnu: learners, X3, statistics) |

Priorities: **Must** (the report is weaker or wrong without it), **Should** (clearly improves the report),
**Could** (only if time is left).

---

## Ground rules for every item

These are the habits that made the current repository trustworthy. Keep them.

1. **No number is typed by hand.** Every number in `report/main.tex` is a macro written by
   `experiments/make_figures.py` from a file in `results/`. New results get new macros.
2. **Write the prediction first.** For each new experiment, put the prediction and the "refuted if" condition in
   this file (or the commit message) *before* running it, as the proposal did for H1–H5. Report the outcome
   either way.
3. **One item, one commit.** Commit messages say what changed and what the result was. Never commit `main.pdf` /
   `proposal.pdf` from the repository root by accident; add files by name.
4. **Every new module gets unit tests,** including at least one test against a case whose answer is known
   (as `support_enum.py` is tested against matching pennies).
5. **Regression check before rerunning anything that already has results.** Rerun on the old code and the new
   code and diff the outputs (as was done for X2 and the learners). A change that should not move old results
   must not move them.
6. **Memory.** The laptop has 16 GB and usually 2–4 GB free. A two-groups CCE program needs about 4 GB; run those
   one at a time. Check free memory before starting two heavy jobs together.
7. **Environment.** Use the `uv` environment (`uv venv .venv`, `uv pip install -e ".[dev]"`). NumPy must be 2.0 or
   newer.

---

## Phase map at a glance

| Phase | Goal | Priority | Your time | Compute | Depends on |
|---|---|---|---|---|---|
| 0 | Merge, rebuild the PDF, pin the environment, prove a clean clone reproduces everything | Must | 1–2 days | ~4 h (full rerun) | — |
| 1 | Close small gaps using data already in `results/` (tests, baselines, "plausible reasons") | Must / Should | 2–3 days | < 30 min | 0 |
| 2 | Show the headline results survive other parameter values | Must / Should | 3–4 days | 2–4 h | 0 |
| 3 | Explain why E0 behaves differently, and settle the complete network under a cut-off | Should | 2–3 days | < 1 h | 0 |
| 4 | `n = 8`, mixed equilibria for bigger agents, learning in more settings | Should / Could | 4–7 days | 2–8 h | 0, ideally 2 |
| 5 | Optional: season game, a learner that best-responds, real-world grounding | Could | open | open | 1–4 |
| 6 | Final report, figures, presentation, release | Must | 2–3 days | ~4 h (final rerun) | everything chosen |

Phases 1, 2 and 3 are independent of each other and can run in parallel across people. Phase 4 should start
after Phase 2's sensitivity sweep, because that tells us whether `n = 8` is likely to change anything.

---

## Phase 0 — Close out the current work

### 0.1 Rebuild the PDF and fit the page limit — **Must**

- **Why.** `report/main.pdf` predates items 1–8. The source has grown (new findings, sixth learner, extra table
  column), and the old PDF was already 8 pages.
- **What / How.**
  1. `cd report && tectonic main.tex` (or `pdflatex main` twice, or Overleaf: upload `main.tex`, `generated/`,
     `figures/`).
  2. Check: no undefined references, no overfull boxes in Tables 3 and 4 (Table 3 now has 11 columns, Table 4 has
     5), Figure 4's six-entry legend fits.
  3. If over the page limit, trim in this order: shorten the regret paragraph in §5.3, merge the two
     paragraphs on mixed/correlated equilibria in §5.1, move the near-tie counts to a footnote. Do **not** drop
     the E0 caveat or the refutation verdicts.
- **Done when.** [ ] PDF builds with no warnings about undefined macros [ ] page count within the limit
  [ ] `report/main.pdf` committed.
- **Owner.** Jash. **Cost.** half a day.

### 0.2 Merge the branch — **Must**

- **What.** Review the 6 commits on `fix/items-5-7` (one per review item, plus one for the report and README),
  then merge into `main`.
  Decide whether `proposal.pdf` belongs in the repository (suggestion: yes, as `docs/proposal.pdf`) and delete
  the stale copy `main.pdf` at the repository root.
- **Done when.** [ ] reviewed by at least one person who did not write it [ ] merged [ ] root clean.
- **Owner.** anyone. **Cost.** 1–2 hours.

### 0.3 Pin the environment — **Should**

- **Why.** Results already differ at the 1e-14 level between machines (seen when re-running the X3 analysis).
  That is harmless today but a reviewer re-running everything should get the same files.
- **What.** Run `uv lock` to create `uv.lock`; commit it. Record Python, NumPy and SciPy versions in the README.
  Normalise line endings with a `.gitattributes` (`* text=auto`, `*.sh text eol=lf`): `run_all.sh` is stored
  with LF in git, but a Windows checkout (`core.autocrlf=true`) turns it into CRLF, which breaks it in Git Bash.
- **Done when.** [ ] `uv.lock` committed [ ] `run_all.sh` runs on Git Bash *and* on a Linux shell.
- **Owner.** Bhavay. **Cost.** 1–2 hours.

### 0.4 Prove a clean clone reproduces everything — **Must**

- **Why.** The README promises that `run_all.sh` regenerates every number. Nobody has run it end to end since
  the new scripts were added (`run_cce.py`, `run_support.py`, the sixth learner).
- **What.** Fresh clone into a new folder, `uv` environment, `./run_all.sh` with `CCE_WORKERS=1`, then
  `git diff --stat` against the committed results.
- **Expected.** Only timing lines in logs and binary figure noise differ. Any other diff is a bug to find
  before the report is final.
- **Cost.** about 4 hours of compute (X1 ~40 min, CCE ~1.8 h, X2 ~20 min, X3 ~10 min, X4 ~11 min, rest minutes);
  your time: 1 hour to start it and read the diff.
- **Done when.** [ ] full run finished without errors [ ] diff explained line by line.
- **Owner.** Bhavay.

---

## Phase 1 — Quick wins on data we already have

Everything here uses files already in `results/` or seconds of extra simulation. Do these first: each removes a
caveat from the report cheaply.

### 1.1 Fix the CCE-gap significance test — **Must**

- **Why.** `analyze_x3.py` tests "the CCE gap falls" by pooling 180 runs per learner across six different games.
  Runs that share a seed share random draws, so they are not independent, and the six games are not one
  population.
- **What.** Test each (leak model, rule) separately with the 30 seeds as the unit; Holm-correct over
  learners × settings. Keep the pooled test only as a summary.
- **How.** `experiments/analyze_x3.py`, block "H4: does the CCE gap fall?". Update the macros `GapFallSig` /
  `GapFallTests` in `make_figures.py` and the sentence in §5.3.
- **Prediction / refuted if.** The gap still falls significantly in every E1/E2 setting for every learner;
  refuted if any learner–setting pair is not significant after correction. Report the count either way.
- **Cost.** 1 hour; compute 1 minute.
- **Done when.** [ ] per-setting tests in `x3_summary.json` [ ] report sentence uses the new counts.
- **Owner.** Vishnu.

### 1.2 Report "rounds to 90 %" for all leak models — **Should**

- **Why.** Table 3 shows rounds-to-90 % only for E1; E0 and E2 are already computed (`rounds_to_90` in
  `x3_summary.json`).
- **What.** Either widen Table 3 or add a short appendix table. Suggest an appendix table to save space.
- **Owner.** Vishnu. **Cost.** 1 hour.

### 1.3 Make H1 non-circular — **Must**

- **Why.** H1 says "α\* is highest for the types most likely to be *pivotal*", but the report measures
  pivotality with ℓ, the own-loss from telling, and α\* = (ℓ + c)/G. With G roughly constant, the ordering of
  α\* follows ℓ by construction.
- **What.** Measure pivotality independently of sharing: for each type, the probability that the candidate's
  rank is K or K + 1 (i.e. she is at the hiring margin) when nobody shares. Then check whether that ordering
  matches the α\* ordering.
- **How.** New function in `leakgame/analysis.py`: simulate days with S = 0 (the table already has the
  needed draws; or 100,000 fresh days), compute P(rank ∈ {K, K+1} | type) under fixed seats. Add the numbers to
  Table 4's H1 row.
- **Prediction / refuted if.** Pivotality: weak < average ≈ strong, matching α\* = 0.10 < 0.32 < 0.38. Refuted
  if the pivotality order differs from the α\* order (then H1's verdict changes and we say so).
- **Cost.** half a day; compute seconds.
- **Owner.** Chaitanya.

### 1.4 Baselines the proposal promised — **Must**

- **Why.** The proposal listed five baselines: nobody shares, everybody shares, **random sharing**, the Level-1
  equilibrium, and a leak-free company. Random sharing never appears, and the learners are never compared with
  any baseline.
- **What.**
  1. For each game in X1 at α = 0.2 (and in X4): expected total utility and company value under nobody shares,
     everybody shares, random sharing (each possible tell independently with probability ½), and the
     equilibrium. All four come from the payoff tables, with no new simulation.
  2. For the learners: their average realised utility per round (`util_mean`, already stored in every
     `x3_*.npz`) against the same baselines.
- **How.** New helper `profile_value(game, distribution over S)` in `leakgame/analysis.py`; a small table in
  §5.3 or the appendix.
- **Prediction / refuted if.** Learners end above random sharing and close to the equilibrium value; refuted if
  any learner's final average utility is below random sharing.
- **Cost.** 1 day; compute minutes.
- **Owner.** Jash (company side), Vishnu (learner side).

### 1.5 Test the explanations the report calls "plausible, not tested" — **Should**

The report currently has five such sentences. Each can be tested cheaply. If a test fails, rewrite the sentence.

| # | Claim in the report | Test | Data needed |
|---|---|---|---|
| a | Average types learn slowest because weak types are almost never hired, so their rewards are less noisy | Compute the reward variance of each type from the simulator; check the order weak < strong < average or similar | 100k simulated days, seconds |
| b | UCB1 has high regret because its confidence bonus keeps it exploring | From `final_counts`, count pulls of non-equilibrium arms per learner; UCB1 should have many more than Thompson | already stored |
| c | ε-greedy's remaining mistakes are on decisions where the wrong row costs little | Per decision cell: is it wrong at the end? and what is its payoff gap (`payoff_gaps.per_cell`)? Correlate | already stored (`regret_cell`, `per_cell_correct`) |
| d | Model-based UCB helps because it pools information across arms that share a friend | Pooling only exists for candidates 1 and 2 (two later friends). Compare its gain over UCB1 for those cells vs. the one-friend cells | already stored (`regret_cell`) |
| e | E0's wide CCE interval comes from exactly redundant tells | See Phase 3.1 | — |

- **Done when.** [ ] each of a–d has a number in the report or the sentence is removed/rewritten.
- **Owner.** Vishnu (b–d), Bhavay (a). **Cost.** 1 day in total.

### 1.6 Measure the leak-aware learner's bias in E2 — **Could**

- **Why.** The report says the counterfactual credit is "very slightly biased" in E2 and that we did not measure
  it.
- **What.** For one teller–friend pair, simulate many days; compare the mean reward of the "silent" arm from
  (i) direct samples with (ii) the credited samples (days where the told question sets were disjoint). The
  difference is the bias.
- **Cost.** 2 hours. **Output.** one number for §4. **Owner.** Vishnu.

---

## Phase 2 — Robustness of the headline results

The report itself lists as a limit: "The bootstrap measures Monte Carlo error, not sensitivity to Δ, σ or K/n,
which we held fixed." Every headline number (the 0.10 / 0.32 / 0.38 thresholds, "cut-off shares more",
"8–12× more loss") comes from one parameter setting. This phase checks which conclusions are qualitative (survive
everywhere) and which are numbers specific to our setting.

### 2.1 One-at-a-time sensitivity sweep — **Must**

- **What.** Re-solve the two-groups game in E1 (both rules, the full α grid) changing one parameter at a time
  from the baseline (n = 6, K = 2, m = 3, Q = 15, Δ = σ = 1, c = 0.001):

  | Parameter | Values | Why these |
  |---|---|---|
  | Δ (marks per leaked question) | 0.5, 2 | half and double the value of a leak |
  | σ (marking noise) | 0.5, 2 | how much luck there is |
  | K (seats, out of 6) | 1, 3 | very selective vs. half hired |
  | c (cost of a tell) | 0.0001, 0.01 | is anything driven by the tiny message cost? |
  | hub position (star network) | 1, 6 | first vs. last interviewee as the popular candidate |

- **How.** New script `experiments/run_sensitivity.py` reusing `analysis.sweep_alpha` (pure equilibria,
  best-response check, a reduced bootstrap of 50 replicates). One payoff table per setting.
- **Predictions / refuted if** (write these into the commit before running):
  1. *Rule decides who talks*: under a cut-off everyone tells at every α > α\* ≈ c/G; under fixed seats the
     staircase weak → average → strong appears in every setting. Refuted if any setting has a strong type
     telling before a weak one under fixed seats.
  2. *H2*: cut-off shares at least as much as fixed seats at every α. Refuted if fixed seats share more in any
     E1 setting.
  3. The thresholds move: larger Δ lowers all α\*, larger σ lowers them too (more luck, less to lose).
     Refuted if the direction is the opposite.
- **Cost.** 1 day of work; compute about 40 s per table plus ~5 s per game: 9 settings × 42 games ≈ 35–45 min.
- **Output.** `results/sensitivity/sens_<param>_<value>.json`; one figure with the three thresholds against each
  parameter.
- **Report impact.** A short paragraph in §5.1 plus a figure (appendix if space is tight). Remove the limitation
  sentence or narrow it to what was not swept.
- **Owner.** Chaitanya.

### 2.2 The company-loss result at other friendliness levels — **Must**

- **Why.** The "8–12× more loss under a cut-off" holds at α = 0.2, where only weak candidates tell under fixed
  seats. The everyone-tells line already shows the rule alone gives 2.5–4.4×. Readers will ask what happens at
  other α.
- **What.** Re-run X4 for α ∈ {0, 0.1, 0.2, 0.3, 0.4, 0.5}. The tables for the 11 bank sizes are reused (utilities
  are linear in α), so only the equilibrium search repeats.
- **How.** Add an α loop to `experiments/run_x4.py`; plot loss ratio against α for 3 bank sizes (Q = 6, 15, 40).
- **Prediction / refuted if.** The ratio falls from about 10× toward the everyone-tells ratio as α passes the
  strong types' threshold (≈ 0.38); the cut-off loses more at every α > 0. Refuted if fixed seats ever lose more.
- **Cost.** half a day; compute ~15 min.
- **Owner.** Jash.

### 2.3 How much does the company-value definition matter? — **Should**

- **Why.** V = Σ_{j hired} (s_j − τ) rewards *not* hiring weak candidates, which only a cut-off can do. "The
  cut-off is the better rule from Q ≥ 9" depends on this choice.
- **What.** Use V_κ = Σ_{j hired} (s_j − κ) for κ ∈ {0, τ/2, τ}. This needs no new simulation:
  V_κ = V_τ + (τ − κ)·E[number hired], and E[number hired] = Σ_j P(x_j) is already in the tables.
- **Prediction / refuted if.** For κ = 0 (every hire is worth its skill, no per-hire cost) fixed seats look
  better at every Q. Report the κ at which the ranking flips.
- **Cost.** half a day. **Owner.** Jash.

### 2.4 Full factorial sweep — **Could**

- Only if 2.1 shows an interaction worth studying: Δ × σ × K at 3 levels each, 27 settings, ≈ 2–3 h compute.

---

## Phase 3 — Understand E0

In E0 (a told question always comes up), the CCE interval is wider than the pure equilibria in 48 of 106 games.
In one example, all four pure equilibria share 67% of tells while correlated play can share 66% to 99%. The report
offers a reason it has not tested. This phase tests it and makes E0's story complete.

### 3.1 Test the redundancy explanation — **Should**

- **Claim to test.** In E0 a second tell to the same friend changes nothing and costs only c, so candidates are
  almost indifferent about who tells, and a CCE can add redundant tells almost for free.
- **Experiments.**
  1. **Vary c.** For two groups, E0, cut-off, α = 0.2, solve the CCE programs at c ∈ {0.0005, 0.001, 0.005, 0.01, 0.02}.
  2. **Inspect the extreme CCEs.** `cce_lp` already returns the distribution z. For the max-sharing CCE, compute
     how often some friend is told by two people. If the explanation is right, almost all extra sharing is
     redundant tells.
- **Prediction / refuted if.** The CCE width shrinks roughly in proportion to c, and ≥ 90% of the extra sharing in
  the max-sharing CCE is redundant tells. Refuted if the width does not shrink with c, or the extra sharing is
  mostly first tells.
- **Cost.** 1 day; compute ~10 min (each LP pair ≈ 100 s).
- **Output.** `results/x1/e0_mechanism.json`. **Report impact.** replace "plausible reason, not tested" in §5.1
  with the result.
- **Owner.** Chaitanya.

### 3.2 Where do the learners end up in E0? — **Should**

- **Why.** Agreement in E0 under a cut-off looks poor (51–74%). The report explains that learners tell redundantly
  instead of coordinating. Now that we know the CCE interval, we can check this directly.
- **What.** From `final_counts` compute each learner's average sharing over the whole run in E0. Check that it lies
  inside the CCE interval, and where (near the pure 67% or near the 99% end?).
- **Prediction / refuted if.** Learners sit inside the interval, toward the high end (redundant telling). Refuted
  if any learner is outside the interval by more than its CCE gap.
- **Cost.** half a day; data already stored. **Owner.** Vishnu.

### 3.3 Settle the complete network under a cut-off — **Should**

- **Why.** For the complete network at n = 6 we only showed that "everybody tells" *is* an equilibrium under a
  cut-off, not that it is the only one. Full enumeration is impossible (2^15 sharing matrices, a 211,000 GB value
  table).
- **What.** Prove it instead. Candidate proposition: *under a cut-off with leak model E1 or E2 and friendly
  utilities, telling a friend is strictly dominant once α·g_min > c, where g_min is the smallest gain in the
  friend's hiring probability from one extra teller.* Lemma 1 already gives g ≥ 0 and Proposition 2 removes
  the own-loss. Strictness needs g_min > 0, which holds because one more teller raises the leak count with
  positive probability.
- **How.** Write the proof (one paragraph, in the style of §3). Check it numerically on the n = 4 complete network,
  where the full table exists: compute g_min and the implied α threshold, and confirm the enumerated equilibria
  switch there.
- **Prediction / refuted if.** The numeric threshold matches the first α at which "everybody tells" becomes the
  unique equilibrium at n = 4. Refuted if they differ by more than one grid step.
- **Cost.** 1 day. **Output.** Proposition 4 in §3, a check in `leakgame/checks.py` and Table 1.
- **Owner.** Jash (proof), Bhavay (check).

---

## Phase 4 — Extend the scope

### 4.1 n = 8 candidates (the proposal's stretch target) — **Should**

What scales and what does not, measured from the code (2,000 simulated days per cell):

| Network at n = 8 | Sharing matrices | Strategy profiles | Simulated outcomes | Largest value table | Feasible as today? |
|---|---|---|---|---|---|
| pairs | 16 | 4,096 | 2.1×10⁸ (2× today's biggest) | < 1 MB | **yes**, everything |
| star | 128 | 2.1×10⁶ | 1.7×10⁹ (18×) | 13 MB | **yes**, slower (~15 min per table) |
| two groups | 4,096 | 6.9×10¹⁰ | 5.4×10¹⁰ (576×) | **412 GB** | **no** |
| complete | 2.7×10⁸ | — | 3.5×10¹⁵ | — | no |

Plan:

1. **Decide K first.** K/n = 1/3 at n = 6. At n = 8 the choices are K = 2 (0.25) or K = 3 (0.375). Suggest
   running both for pairs (cheap) and picking one for the rest. *Decision needed.*
2. **Pairs and star: the full pipeline** (pure equilibria, dominance, support enumeration for pairs, CCE programs,
   bootstrap), E0–E2, both rules, the α grid. Mostly config changes; `GameConfig(n=8, K=...)`. Watch memory:
   the pairwise block table for star at n = 8 is about 6561 × 128 × 2 × 28 × 20 × 4 bytes ≈ 3.8 GB, too big.
   Build star tables without `Pxx_blocks` (needed only for fairness utilities) by adding a flag to
   `build_table`.
3. **Two groups at n = 8: a different method, not a bigger run.**
   - Simulate only the matrices that local search visits (as `complete_local_table` does for the complete
     network), not all 4,096.
   - Find equilibria by best-response dynamics from many starts plus iterated dominance on the visited part,
     and verify each candidate against *every* single-candidate deviation.
   - State the result honestly: "equilibria found and verified", not "all equilibria".
- **Predictions / refuted if.** The staircase and the cut-off-shares-more pattern hold at n = 8; the thresholds
  move (more competitors per seat when K/n falls). Refuted if a strong type tells before a weak one under fixed
  seats, or fixed seats share more than a cut-off in E1/E2.
- **Cost.** pairs + star: 2 days, compute 1–2 h. Two groups: 3–4 more days, compute several hours. Do two groups
  only if the team decides it is worth it (*decision needed*).
- **Owner.** Chaitanya (solvers), Bhavay (tables).

### 4.2 Mixed equilibria for agents with more than two actions — **Could**

- **Why.** Support enumeration currently runs only on the pairs network. Dominance leaves 34 two-groups games and
  30 star games open, all but one in E0.
- **What.**
  1. The one open game outside E0 (two groups, E1, fixed seats, α = 0.1) has only 36 profiles left after dominance.
     Enumerate its supports exactly: extend `support_enum.py` from two actions to any number, applied to the
     residual game only.
  2. For E0 games with large residuals (up to 16,384 profiles), keep the CCE bound as the statement. Phase 3.1
     explains why E0 has many near-equivalent equilibria.
- **Cost.** 2 days. **Owner.** Chaitanya.

### 4.3 Learning in more settings — **Should**

- **Why.** All learning results come from one game: two groups, α = 0.2. A reader cannot tell whether "learners
  find the equilibrium" depends on that choice.
- **What.** Run all six learners (30 seeds, E0–E2, both rules, 200,000 rounds) at:
  - α = 0.1 and α = 0.3 (two groups): either side of the thresholds, where payoff gaps are smallest. These are
    the hardest cases.
  - the pairs network at α = 0.2: smallest game, should be easiest.
- **How.** Add `network` and `alpha` arguments to `experiments/run_x3.py` and `analyze_x3.py`; write results to
  `results/x3/<setting>/`.
- **Prediction / refuted if.** Agreement ordering of learners is the same in every setting; agreement is lowest at
  α = 0.3 (smallest gaps). Model-based UCB is fastest everywhere. Refuted if UCB1 or Thompson beats Model-based UCB
  on rounds-to-90 % in any E1/E2 setting.
- **Cost.** 1 day of work; compute ≈ 9 min per learner per setting, 6 learners × 3 settings ≈ 3 h. Run at most 3
  learners at a time (memory).
- **Owner.** Vishnu.

---

## Phase 5 — Optional extensions

Only start these if Phases 0–4 (the parts we chose) are done. Each is a project in itself.

### 5.1 A season game: placed candidates leave — **Could**

- The proposal listed this as "beyond this proposal (not committed)": over a placement season, hired candidates
  leave the pool, so telling a friend now affects who is left later. This turns the stage game into a stochastic
  game.
- First step if we do it: two rounds only, n = 6, with the round-2 pool determined by round-1 hires; solve by
  backward induction on the existing tables.

### 5.2 A learner that best-responds in an estimated game — **Could**

- The proposal's original "model-based" learner. Model-based UCB estimates a reduced-form value of a leaked
  question and plans alone. A best-responding learner would also estimate the others' strategies (from observed
  tells it receives) and the structural Δ, then compute a best response in its estimated game.
- Worth it only if 4.3 shows Model-based UCB struggling somewhere.

### 5.3 Real-world grounding — **Could**

- Every claim is comparative because there is no real leak data. A short, anonymous survey of students about
  whether they share questions, with whom, and under which hiring rule, would let the discussion connect the model
  to behaviour. Needs ethics approval from the institute *before* any data collection. Treat as a discussion item,
  not a deliverable.

---

## Phase 6 — Final write-up and submission

### 6.1 Integrate results into the report — **Must**

For each finished item above, in this order:

1. Macros in `make_figures.py` (never typed numbers).
2. Text in the right section (see each item's "Report impact").
3. Update the deviations list and the Limits paragraph in §5.5. Remove limits we have closed; add new ones honestly.
4. Update Table 4 if any verdict changes (H1 after 1.3; H2 after 2.1 / 4.1).
5. Update the README's Results and Known limits sections to match.

### 6.2 Figure and table checklist — **Must**

| Item | Check |
|---|---|
| Figure 1–3 | unchanged unless 2.1 / 4.1 adds panels |
| Figure 4 | six learners, legend fits at print size |
| Figure 5 | consider adding the α sweep of 2.2 as panel (c) |
| Table 1 | add the Proposition 4 check from 3.3 |
| Table 3 | 11 columns: check it fits; rounds-to-90 for E0/E2 in the appendix (1.2) |
| Table 4 | 5 columns with "Refuted if"; verdicts match the text |
| Appendix | sensitivity (2.1), X4 at other α (2.2), baselines (1.4), learning in more settings (4.3) |

### 6.3 Presentation — **Must if there is one** (*decision needed*)

- Suggested 10-slide outline: question and example (Aarav) → model in one picture → who tells (Figure 1) → why
  (Figure 2a,b) → robustness (2.1) → learning (Figure 4) → company loss (Figure 5 + 2.2) → what E0 teaches us
  (3.1) → limits → conclusion.
- Every number on a slide must also be in the report.

### 6.4 Final reproducibility run and release — **Must**

1. Fresh clone, `./run_all.sh`, diff against committed results (as in 0.4).
2. Rebuild the PDF from the regenerated `report/generated/`.
3. Tag the commit (`v2.0`) and, if the course asks for it, attach a zip of the repository to the submission.

---

## Decisions we need to make

| # | Decision | Options | Suggestion | Blocks |
|---|---|---|---|---|
| D1 | Deadline and page limit of the final report | — | find out first | 0.1, all of Phase 6 |
| D2 | Is there a presentation? | yes / no | — | 6.3 |
| D3 | Which phases are in scope | 0–3 only; 0–4; all | 0–3 plus 4.1 (pairs, star) and 4.3 | everything after Phase 1 |
| D4 | Seats at n = 8 | K = 2 or K = 3 | run both for pairs, then choose | 4.1 |
| D5 | Attempt two groups at n = 8 | yes (new method, several days) / no | only if 2.1 shows thresholds move a lot with n-related parameters | 4.1 step 3 |
| D6 | Keep `proposal.pdf` in the repository | `docs/` / leave out | `docs/proposal.pdf` | 0.2 |
| D7 | Who owns which item | — | owners suggested per item above | — |

---

## Risks and how we handle them

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Report over the page limit after adding results | high | medium | appendix for robustness tables; trim order in 0.1 |
| A robustness sweep contradicts a headline (e.g. H2 fails at some Δ) | medium | high, but honest | that is a result: report it, narrow the claim; predictions were written first |
| Memory: two heavy jobs crash the laptop | medium | lost hours | one two-groups CCE job at a time; resumable scripts (as `run_cce.py`) for anything > 15 min |
| n = 8 two groups takes far longer than planned | high | medium | it is gated by decision D5; pairs and star alone already meet the stretch target partially |
| Results drift between machines | low | low | `uv.lock` (0.3); regression diffs before every rerun |
| Someone edits numbers by hand in `main.tex` | low | high | review rule: grep the diff of `main.tex` for digits outside macros before merging |
| A "plausible reason" test (1.5) fails | medium | low | rewrite the sentence; that is the point of the test |

---

## Reference: measured run times and sizes

Measured on our 24-thread, 16 GB laptop unless noted.

| Job | Time | Notes |
|---|---|---|
| Unit tests (40) | 10–15 s | |
| Payoff table, two groups, n = 6 | 15–40 s | pairs ~8 s; tables cached in `results/tables/` |
| X1 (`run_x1.py`) | ~40 min | original machine; nine cells in parallel |
| CCE programs E0 + E2 (`run_cce.py`) | ~1.8 h with one worker | two-groups LP pair ≈ 60–110 s and ~4 GB |
| Dominance + support enumeration (`run_support.py`) | ~5 min | |
| X2 (`run_x2.py`) | ~18 min | fairness bootstraps resample the pairwise table |
| X3, one learner (180 runs × 200,000 rounds) | ~9 min | `run_x3.py <learner>` |
| X3 analysis (`analyze_x3.py`) | ~2 min | |
| X4 (`run_x4.py`) | ~11 min | original machine |
| Figures, tables, macros (`make_figures.py`) | ~5 s | |
| Full `run_all.sh` | ~4 h | dominated by the CCE programs |

Current headline facts the plan builds on (all from `results/`):

- Dominance solves 311 of 378 X1 games exactly; all but one of the 67 open games are in E0.
- CCE sharing is a single point in all 212 E1/E2 games, but in only 55 of 106 E0 games.
- Support enumeration finds 3 mixed equilibria (pairs, E0, fixed seats).
- Model-based UCB reaches 90% agreement in 12,000 / 4,500 rounds (E1, fixed seats / cut-off), against
  56,000 / 18,000 for UCB1.
- At α = 0.2 a cut-off company loses 8–12× more hiring value than a fixed-seats company; 2.5–4.4× if everyone
  told under both rules.
