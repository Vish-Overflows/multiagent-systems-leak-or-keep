# Phase 0 report: closing out the current work

What was asked: do Phase 0 of `plan.md` (rebuild the PDF, merge the branch, pin the environment, prove that a clean
clone reproduces everything). This file says what was done, what changed in the repository, what went wrong on
the way, and how you can check each claim yourself.

**Result in one line:** all four items are done. `main` now contains the review items 1–8 and Phase 0. A fresh clone,
with its committed results deleted, ran `run_all.sh` start to finish (exit code 0, 2 h 43 min), and every JSON, NPZ
and generated LaTeX file came out identical to the committed one.

---

## 1. Summary table

| Item | Status | One-line outcome |
|---|---|---|
| 0.1 Rebuild the PDF | done | `report/main.pdf` rebuilt with your `tectonic.exe`: 10 pages at the end of Phase 0 (11 after Phase 1 added an appendix), no undefined references, no overfull boxes. **The page limit is still unknown** |
| 0.2 Merge the branch | done (local only) | `fix/items-5-7` merged into `main` with a merge commit; `proposal.pdf` moved to `docs/`; stale root `main.pdf` removed; nothing pushed |
| 0.3 Pin the environment | done | `uv.lock`, README environment section, `.gitattributes`, `X1_WORKERS` / `X3_WORKERS` settings |
| 0.4 Clean-clone reproduction | done | exit code 0; 38 files identical, 5 logs differ for harmless reasons, 14 figure binaries not compared |

Commits (in order): `0e39b8a` Phase 0 changes, `5186232` merge into `main`, and (on branch `phase-1`, because it
finished after the merge) `4c70827` for 0.4 and the comparison tool.

---

## 2. Item 0.1: rebuild the PDF

**What I found.** You had already built `report/main.pdf` (309.9 KB, 10 pages) with `report/tectonic.exe`. I read it
page by page. Every macro and reference had resolved, and Tables 3 and 4 and the six-learner legend of Figure 4 fit.

**What I changed.**
- One real defect: in §5.1 the bold heading "What about mixed and correlated equilibria?" ran on inside the previous
  paragraph instead of starting its own. This came from my own earlier edit (it had no blank line before it). I added the
  blank line in `report/main.tex`.
- I rebuilt the PDF with the same `tectonic.exe`: `cd report && ./tectonic.exe main.tex`.

**What I checked** (with `--keep-logs`, then read `main.log`):
- 0 overfull boxes, 0 undefined references or citations, no `??` in the extracted text.
- The only warnings are `algorithm2e.sty` "Invalid UTF-8 byte" (inside that package, not in our source) and
  "Font shape `T1/lmr/m/scit' undefined" (small-caps italic falls back to another shape in the title style). Both
  are harmless and were there before.
- Page count 10 (counted with `pypdf` in a throwaway `uv run --with pypdf`, which does not touch the project's
  dependencies).

**Not resolved.** The page limit. The report grew from 8 to 10 pages with the review items, and to 11 pages in Phase 1.
If the limit is 8–10, the cheapest cuts are listed in `plan.md` item 0.1 (shorten the regret paragraph, merge the two
mixed/correlated paragraphs in §5.1, move the near-tie counts to a footnote). I did not cut anything because I do not
know the limit.

`tectonic.exe` (51 MB) is now in `.gitignore`; it is not committed.

---

## 3. Item 0.2: merge the branch

**What I did.**
- Checked that the 6 commits on `fix/items-5-7` were each one review item (plus one for the report/README), then ran
  `git checkout main && git merge --no-ff fix/items-5-7`. History now shows the merge commit `5186232`.
- `proposal.pdf` was untracked at the repository root. I moved it to `docs/proposal.pdf` and committed it (this was
  the plan's suggestion, decision D6; easy to undo).
- The untracked `main.pdf` at the root was a stale copy. Before deleting it I compared it byte for byte with the old
  committed `report/main.pdf` (from the first commit): **identical**, so nothing was lost.
- Nothing was pushed. The remote is `https://github.com/Vish-Overflows/multiagent-systems-leak-or-keep.git`; pushing is
  yours to do.

**Limit of this step.** The plan said "reviewed by someone who did not write it". I did the merge on your request;
the review by a teammate has not happened.

---

## 4. Item 0.3: pin the environment

| Change | File | Why |
|---|---|---|
| Lock file with exact versions | `uv.lock` (32 packages, 1,573 lines) | a reviewer re-running everything gets the same versions |
| Environment section rewritten | `README.md` | `uv sync --frozen --extra dev`; records that the committed results come from Python 3.10.19, NumPy 2.2.6, SciPy 1.15.3, matplotlib 3.10.9; says other versions agree to about 1e-14; warns that a two-group CCE program needs about 4 GB |
| Line-ending and binary rules | `.gitattributes` | `* text=auto`, `*.sh text eol=lf`, and `*.pdf *.png *.npz binary` |
| Worker settings | `experiments/run_x1.py`, `experiments/run_x3.py` | `X1_WORKERS` and `X3_WORKERS` environment variables; defaults unchanged (9 and one per learner), so nothing changes for anyone who does not set them |
| Ignore the local LaTeX binary | `.gitignore` | `report/tectonic.exe` |

**A correction to my own plan.** `plan.md` said `run_all.sh` has CRLF endings in git and breaks on Linux. That was wrong:
`git ls-files --eol` shows `i/lf`, so it is stored with LF. Only a Windows working copy (`core.autocrlf=true`) turns it
into CRLF, which breaks it in Git Bash. The `.gitattributes` rule fixes exactly that, and I verified it: a fresh clone
on this Windows machine has an LF `run_all.sh`. I corrected the sentence in `plan.md`.

The environment itself was created with `uv venv .venv` and `uv sync`; it is Python 3.10.19 with NumPy 2.2.6.

---

## 5. Item 0.4: clean-clone reproduction

### The procedure
1. `git clone FAI ../FAI_clean` (a clone of `main` after the merge) and `uv sync --frozen --extra dev`. It installed
   the same versions as the main working copy (3.10.19 / 2.2.6 / 1.15.3).
2. **Delete every committed result** in the clone: `rm -rf results report/generated report/figures`.
3. `X1_WORKERS=2 X3_WORKERS=2 CCE_WORKERS=1 bash ./run_all.sh > ../clean_run.log 2>&1`.
4. Compare with `experiments/compare_with_committed.py` (added in this phase, see below).

### The mistake I made first, and why step 2 exists
My first launch skipped step 2. I noticed from the file listing that `results/x1/` already held all its files:
**a clone contains the committed results, and `run_x1.py` and `run_cce.py` skip any cell whose result file already
exists.** The run would have "reproduced" nothing for X1 and the CCE programs. I killed it (it had only reached the
checks stage), checked that no stray Python process remained, deleted the committed results in the clone, and
relaunched. This is now written into the README and into the docstring of the comparison tool, and it changes how
Phase 6's final run must be done.

### What it cost
(Started 17:26, finished 20:09; 2 workers for X1, 2 for X3, 1 for CCE; the laptop had only 1.5–3 GB of free RAM because
of other programs, so I kept the worker counts low.)

| Stage | Time |
|---|---|
| unit tests (40) and verification checks | 4 min |
| X1 (nine cells, with the CCE programs for E1 inside) | 47 min |
| CCE programs for E0 and E2 (`run_cce.py`) | 69 min |
| dominance and support enumeration | 46 s |
| X2 | 14 min |
| X3 (six learners) + analysis | 15 min |
| X4 | 12 min |
| figures, tables, numbers, console screenshots | seconds |
| **Total** | **2 h 43 min** |

The PDF step was skipped by `run_all.sh` itself (it looks for `tectonic` on the PATH or `.tools/tectonic`).

### The comparison result
`python experiments/compare_with_committed.py ../FAI_clean`:

```
38 identical   5 differ   2 missing   14 binary (listed, not compared)
```

- **Identical (38):** all X1 result files (the nine cells, the six CCE files, the complete-network file, the support
  file, the summary), `x2.json`, all six X3 `.npz` files and `x3_summary.json`, `x4.json`, `checks.json`,
  `numbers.json`, and the generated LaTeX tables. Numeric comparison with relative tolerance 1e-9, timing fields
  ignored. This is the substantive result: the pipeline reproduces.
- **Differ (5), all logs, all explained:**

  | Log | Why it differs |
  |---|---|
  | `checks.log` | last-digit float noise (for example `...94941` vs `...94942`, about 1e-16 relative) and one timing line that contains the original author's Mac path |
  | `x1.log`, `x1_cce.log` | the committed logs come from a *resumed* run and contain "skip … (already done)" lines; the README already says so |
  | `x3.log` | progress lines of parallel workers are interleaved differently (the underlying NPZ data is identical) |
  | `figures.log` | the committed one was stale (524 macros, written before the review fixes) |
- **Missing (2):** `x3_model_based_rerun.log` and `x3_model_ucb.log`. They document separate reruns and are not
  produced by `run_all.sh`.
- **Not compared (14):** PNG and PDF files. Matplotlib output is not byte-stable across machines. They are drawn
  from the identical data and tables, so they were not checked pixel by pixel.

### Files left outside the repository
On your Desktop: the clone `FAI_clean`, `clean_run.log` and `clean_run_started.txt`. Delete them when you no longer
need them (the clone is a few hundred MB).

---

## 6. Complete list of changes (commit `0e39b8a` and `4c70827`)

Phase 0 commit `0e39b8a` (10 files):

| File | Change |
|---|---|
| `.gitattributes` | new |
| `.gitignore` | + `report/tectonic.exe` |
| `README.md` | environment section, worker notes |
| `docs/proposal.pdf` | new (moved from the root) |
| `experiments/run_x1.py`, `experiments/run_x3.py` | `X1_WORKERS`, `X3_WORKERS` |
| `plan.md` | new (the plan itself), with the CRLF note corrected |
| `report/main.pdf` | rebuilt (302,889 → 317,282 bytes) |
| `report/main.tex` | one blank line (run-in heading) |
| `uv.lock` | new |

Then the merge commit `5186232`, and on `phase-1` the commit `4c70827` for 0.4: `experiments/compare_with_committed.py`
(new), `README.md` (how to reproduce), `plan.md` (outcome), `results/logs/figures.log` (refreshed).

No result file was changed in Phase 0.

---

## 7. What to check yourself

```bash
git log --oneline --graph | head          # the merge commit and the Phase 0 commit
python -m pytest -q                        # 42 pass on the Phase 1 branch (40 on main)
cd report && ./tectonic.exe main.tex       # the PDF builds
python experiments/compare_with_committed.py "../FAI_clean"   # the 38 / 5 / 2 / 14 result
```

To redo the clean-clone run later (Phase 6), follow the numbered procedure at the top of
`experiments/compare_with_committed.py`; do not forget the deletion step.

---

## 8. Open points

- **Page limit** (decision D1 in `plan.md`): unknown; the PDF is 11 pages now.
- **Review by a teammate** of the merge into `main`.
- **Pushing** `main` (and later `phase-1`) to GitHub.
- The CCE programs dominate the run time (69 of 163 minutes) and need about 4 GB each, so a full rerun needs a quiet
  machine.
