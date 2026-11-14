#!/usr/bin/env bash
# Regenerate every result, figure, table and number in the report from scratch.
# About 75 minutes on a 10-core laptop for everything except the CCE programs of run_cce.py, which take about
# 2 hours with one worker (set CCE_WORKERS=3 if you have the memory: a two-groups program needs ~4 GB).
# X1 dominates the rest: its nine cells run in parallel, the slowest takes ~40 min.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
if [ ! -x "$PY" ] && [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; fi   # Windows
export MPLCONFIGDIR="$PWD/.mplconfig"
mkdir -p results/logs "$MPLCONFIGDIR"

$PY -m pytest -q                                   | tee results/logs/tests.log     # 42 unit tests, ~15 s
$PY experiments/run_checks.py                      | tee results/logs/checks.log    # Table 1
$PY experiments/run_x1.py                          | tee results/logs/x1.log        # Figures 1-3   (skips finished cells)
$PY experiments/run_cce.py ${CCE_WORKERS:-1}        | tee results/logs/x1_cce.log    # CCE programs for E0 and E2 (E1 is in run_x1)
$PY experiments/run_support.py                     | tee results/logs/x1_support.log # dominance + support enumeration
$PY experiments/run_x2.py                          | tee results/logs/x2.log        # Table 2, Figure 3c
$PY experiments/run_x3.py                          | tee results/logs/x3.log        # 1,080 learning runs
$PY experiments/analyze_x3.py                      | tee results/logs/x3_analysis.log   # Table 3, Figure 4
$PY experiments/run_h1.py                          | tee results/logs/h1.log        # H1 pivotality
$PY experiments/run_baselines.py                   | tee results/logs/baselines.log # needs X3
$PY experiments/run_explanations.py                | tee results/logs/explanations.log  # needs X3
$PY experiments/run_e2_bias.py                     | tee results/logs/e2_bias.log
$PY experiments/run_x4.py                          | tee results/logs/x4.log        # Figure 5
$PY experiments/make_figures.py                    | tee results/logs/figures.log   # figures, tables, numbers.tex
$PY experiments/make_console_shots.py                                               # README screenshots
# PDF (needs tectonic, or any LaTeX: cd report && pdflatex main && pdflatex main)
if command -v tectonic >/dev/null 2>&1; then (cd report && tectonic main.tex); elif [ -x .tools/tectonic ]; then (cd report && ../.tools/tectonic main.tex); fi
