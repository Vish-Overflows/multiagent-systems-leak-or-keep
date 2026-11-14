"""Render the console output of the main scripts as images for the README.

These are not hand-made: each image is the text of a log file in results/logs/ (written by the
scripts themselves), drawn in a monospace font.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(ROOT, ".mplconfig"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOGS = os.path.join(ROOT, "results", "logs")
OUT = os.path.join(ROOT, "results", "figures")


def render(lines, name, title):
    lines = [ln.rstrip("\n")[:200] for ln in lines]
    width = max(len(ln) for ln in lines + [title]) * 0.0685 + 0.4
    height = (len(lines) + 2) * 0.165 + 0.3
    fig = plt.figure(figsize=(width, height), facecolor="#1e1e1e")
    fig.text(0.01, 1 - 0.16 / height, "$ " + title, family="monospace", fontsize=8, color="#9cdcfe", va="top")
    for k, ln in enumerate(lines):
        fig.text(0.01, 1 - (0.16 + 0.165 * (k + 1.3)) / height, ln, family="monospace", fontsize=8, color="#d4d4d4", va="top")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  wrote", name)


def read(name):
    p = os.path.join(LOGS, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return [ln for ln in f.readlines() if "Warning" not in ln and "nice(5)" not in ln and ln.strip()]


def main():
    x3 = read("x3_analysis.log")
    if x3:
        render(x3[:37], "console_x3_learning", "python experiments/analyze_x3.py      # final metrics, 30 seeds per row")
    x4 = read("x4.log")
    if x4:
        render(x4, "console_x4_company", "python experiments/run_x4.py      # company loss vs question-bank size")
    x2 = read("x2.log")
    if x2:
        render(x2, "console_x2_utilities", "python experiments/run_x2.py      # <kind> <rule> <param> <share of tells> <#equilibria>")
    t = read("tests.log")
    if t:
        render(t[-3:], "console_tests", "python -m pytest -q")


if __name__ == "__main__":
    main()
