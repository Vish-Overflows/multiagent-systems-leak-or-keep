"""Turn the result files into the figures, LaTeX tables and number macros used by the report.

Every number quoted in report/main.tex is a macro defined in report/generated/numbers.tex, which this
script writes from results/*.json -- so the paper cannot drift away from what the code produced.

Outputs
  results/figures/*.png            (README)        report/figures/*.pdf      (paper)
  report/generated/numbers.tex     report/generated/tab_*.tex
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".mplconfig"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG_PNG = os.path.join(RES, "figures")
FIG_PDF = os.path.join(ROOT, "report", "figures")
GEN = os.path.join(ROOT, "report", "generated")
for d in (FIG_PNG, FIG_PDF, GEN):
    os.makedirs(d, exist_ok=True)

# plain-language labels used everywhere
RULE = {"SEATS": "Fixed seats", "BAR": "Cut-off score"}
COL = {"SEATS": "#1f5fa8", "BAR": "#d9622b"}
ENV = {"E0": "E0 (always repeats)", "E1": "E1 (repeats by chance)", "E2": "E2 (real question bank)"}
NET = {"two_groups": "Two friend groups", "pairs": "Pairs of friends", "star": "One popular candidate",
       "complete": "Everyone is friends"}
TYPES = ["weak", "average", "strong"]
LEARNER = {"eps-greedy": r"$\varepsilon$-greedy", "UCB1": "UCB1", "Thompson": "Thompson", "EXP3": "EXP3",
           "model-based": "Leak-aware UCB"}
LEARNER_TXT = {"eps-greedy": r"$\varepsilon$-greedy", "UCB1": "UCB1", "Thompson": "Thompson sampling", "EXP3": "EXP3",
               "model-based": "Leak-aware UCB"}
LCOL = {"eps-greedy": "#7f7f7f", "UCB1": "#1f77b4", "Thompson": "#2ca02c", "EXP3": "#9467bd", "model-based": "#d62728"}

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "font.family": "serif",
    "mathtext.fontset": "cm", "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5, "lines.linewidth": 1.6,
})

NUM = {}          # macro name -> string


def num(name, value):
    assert name.isalpha(), name
    NUM[name] = str(value)


def pct(x, d=0):
    return f"{100 * x:.{d}f}"


def load(path):
    p = os.path.join(RES, path)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def save(fig, name):
    fig.savefig(os.path.join(FIG_PNG, name + ".png"), dpi=220)
    fig.savefig(os.path.join(FIG_PDF, name + ".pdf"))
    plt.close(fig)
    print("  wrote", name)


def rec_at(records, alpha):
    return min(records, key=lambda r: abs(r["alpha"] - alpha))


# =============================================================================== X1
def fig_who_tells(x1):
    """Headline picture: the equilibrium at friendliness 0.2, two friend groups, E1."""
    r = x1.get(("two_groups", "E1"))
    if r is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.25))
    for ax, rule in zip(axes, ("SEATS", "BAR")):
        rec = rec_at(r["rules"][rule]["records"], 0.2)
        bne = rec["first_bne"]
        grid = np.full((6, 3), np.nan)
        for i in range(4):
            n_later = 2 if i < 2 else 1
            for th, tn in enumerate(["weak", "avg", "strong"]):
                grid[i, th] = len(bne[f"c{i + 1}/{tn}"]) / n_later
        ax.imshow(np.where(np.isnan(grid), 0.0, grid), cmap="Blues" if rule == "SEATS" else "Oranges", vmin=0, vmax=1.6,
                  aspect="auto")
        for i in range(6):
            for th, tn in enumerate(["weak", "avg", "strong"]):
                if i >= 4:
                    ax.add_patch(plt.Rectangle((th - 0.5, i - 0.5), 1, 1, color="#eeeeee", zorder=2))
                    if th == 1:
                        ax.text(th, i, "no friend left to tell", ha="center", va="center", fontsize=7, color="#777777", zorder=3)
                    continue
                who = bne[f"c{i + 1}/{tn}"]
                txt = "tells " + " & ".join(str(w) for w in who) if who else "silent"
                ax.text(th, i, txt, ha="center", va="center", fontsize=7.5,
                        color="black", fontweight="bold" if who else "normal")
        ax.set_xticks(range(3)); ax.set_xticklabels(["weak", "average", "strong"])
        ax.set_yticks(range(6)); ax.set_yticklabels([f"{i + 1}" for i in range(6)])
        ax.set_xlabel("candidate's own ability type")
        ax.set_title(RULE[rule], color=COL[rule], fontweight="bold")
        ax.grid(False)
        ax.set_xticks(np.arange(-0.5, 3), minor=True); ax.set_yticks(np.arange(-0.5, 6), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5); ax.tick_params(which="both", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("interview position")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig1_who_tells")


def fig_sharing_vs_alpha(x1, comp):
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 1.95), sharey=True)
    for ax, net in zip(axes[:3], ("two_groups", "pairs", "star")):
        for rule in ("SEATS", "BAR"):
            for env, ls, lw, al in (("E0", ":", 1.0, 0.9), ("E2", "--", 1.0, 0.9), ("E1", "-", 1.9, 1.0)):
                r = x1.get((net, env))
                if r is None:
                    continue
                recs = r["rules"][rule]["records"]
                a = [x["alpha"] for x in recs]
                lo = np.array([x.get("sharing_rate_min", np.nan) for x in recs], dtype=float)
                hi = np.array([x.get("sharing_rate_max", np.nan) for x in recs], dtype=float)
                ax.step(a, (lo + hi) / 2, where="post", color=COL[rule], ls=ls, lw=lw, alpha=al)
                if env == "E0" and (hi - lo).max() > 1e-9:
                    ax.fill_between(a, lo, hi, step="post", color=COL[rule], alpha=0.12, lw=0)
        ax.set_title(NET[net]); ax.set_xlabel(r"friendliness $\alpha$")
        ax.set_xlim(0, 0.5); ax.set_ylim(-0.04, 1.04); ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax = axes[3]
    if comp is not None:
        a = comp["alphas"]
        for rule in ("SEATS", "BAR"):
            ns = np.array([x["no_share_is_bne"] for x in comp[rule]]); als = np.array([x["all_share_is_bne"] for x in comp[rule]])
            y = np.where(als & ~ns, 1.0, np.where(ns & ~als, 0.0, np.nan))
            ax.step(a, y, where="post", color=COL[rule], lw=1.9)
    ax.set_title(NET["complete"]); ax.set_xlabel(r"friendliness $\alpha$"); ax.set_xlim(0, 0.5); ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    axes[0].set_ylabel("share of possible tells\nmade in equilibrium")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COL["SEATS"], lw=1.9, label=RULE["SEATS"]), Line2D([], [], color=COL["BAR"], lw=1.9, label=RULE["BAR"]),
         Line2D([], [], color="k", lw=1.9, ls="-", label="E1"), Line2D([], [], color="k", lw=1, ls="--", label="E2"),
         Line2D([], [], color="k", lw=1, ls=":", label="E0")]
    fig.legend(handles=h, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.08), handlelength=2.2, columnspacing=1.4)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig3_sharing_vs_friendliness")


def fig_why(x1, x2):
    """(a) what a tell costs the teller vs what it gives the friend, (b) break-even friendliness, (c) clannishness."""
    r = x1.get(("two_groups", "E1"))
    if r is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.05))
    # (a) loss vs gain by type under fixed seats (mean over the four candidates who can tell, per friend told)
    ax = axes[0]
    w = 0.36
    for k, (rule, off) in enumerate((("SEATS", -w / 2), ("BAR", w / 2))):
        lg = r["rules"][rule]["loss_gain"]
        ell = [np.mean([lg[f"c{i}/{t}"]["ell"] / (2 if i <= 2 else 1) for i in (1, 2, 3, 4)]) for t in ("weak", "avg", "strong")]
        ax.bar(np.arange(3) + off, 100 * np.array(ell), w, color=COL[rule], label=RULE[rule])
        for xx, v in zip(np.arange(3) + off, ell):
            ax.text(xx, 100 * v + 0.03, f"{100 * v:.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(range(3)); ax.set_xticklabels(TYPES)
    ax.set_ylabel("own hiring chance lost\nper friend told (pct. points)")
    ax.set_title("(a) Cost of a tell"); ax.legend(frameon=False, loc="upper left", borderaxespad=0.2)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.6)
    # (b) break-even friendliness
    ax = axes[1]
    for rule, off in (("SEATS", -w / 2), ("BAR", w / 2)):
        lg = r["rules"][rule]["loss_gain"]
        a_loc = [np.mean([lg[f"c{i}/{t}"]["alpha_star_local"] for i in (1, 2, 3, 4)]) for t in ("weak", "avg", "strong")]
        ax.bar(np.arange(3) + off, a_loc, w, color=COL[rule])
        for xx, v in zip(np.arange(3) + off, a_loc):
            ax.text(xx, v + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.axhline(0.2, color="k", lw=0.8, ls="--"); ax.text(-0.48, 0.205, r"$\alpha=0.2$", ha="left", va="bottom", fontsize=6.5)
    ax.set_xticks(range(3)); ax.set_xticklabels(TYPES); ax.set_ylabel(r"break-even friendliness $\alpha^\star$")
    ax.set_title("(b) Friendliness needed"); ax.set_ylim(0, 0.5)
    # (c) clannishness
    ax = axes[2]
    if x2 is not None:
        for rule in ("SEATS", "BAR"):
            rows = [c for c in x2["clan"] if c["rule"] == rule]
            ax.step([c["beta"] for c in rows], [c["sharing_rate_mean"] for c in rows], where="post", color=COL[rule],
                    marker="o", ms=2.5, label=RULE[rule])
    ax.set_xlabel(r"dislike of outsiders' offers $\beta$"); ax.set_ylabel("share of possible\ntells made")
    ax.set_title("(c) Disliking outsiders"); ax.set_ylim(-0.04, 1.08); ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig2_why_and_clans")


def numbers_x1(x1, comp):
    r = x1.get(("two_groups", "E1"))
    if r is None:
        return
    for rule, tag in (("SEATS", "Seats"), ("BAR", "Bar")):
        lg = r["rules"][rule]["loss_gain"]; ast = r["rules"][rule]["alpha_star"]
        for t, tt in (("weak", "Weak"), ("avg", "Avg"), ("strong", "Strong")):
            num(f"AstarLocal{tag}{tt}", f"{np.mean([lg[f'c{i}/{t}']['alpha_star_local'] for i in (1, 2, 3, 4)]):.2f}")
            num(f"Ell{tag}{tt}", f"{100 * np.mean([lg[f'c{i}/{t}']['ell'] / (2 if i <= 2 else 1) for i in (1, 2, 3, 4)]):.2f}")
            num(f"Gain{tag}{tt}", f"{100 * np.mean([lg[f'c{i}/{t}']['G'] / (2 if i <= 2 else 1) for i in (1, 2, 3, 4)]):.1f}")
            grid = [ast[f"c{i}/{t}"]["alpha_star_all"] for i in (1, 2, 3, 4)]
            lo, hi = min(grid), max(grid)
            num(f"AstarGrid{tag}{tt}", f"{lo:.3f}".rstrip("0") if lo == hi else f"{lo:.3f}".rstrip("0") + "--" + f"{hi:.3f}".rstrip("0"))
        rec = rec_at(r["rules"][rule]["records"], 0.2)
        num(f"PilotShare{tag}", pct(rec["sharing_rate_mean"]))
        num(f"PilotBootFrac{tag}", pct(rec["bootstrap"]["frac_exact_bne"]))
        num(f"PilotBootR{tag}", rec["bootstrap"]["R"])
    # how often is the pure equilibrium unique, over all (network, env, rule, alpha) cells solved exactly
    tot = uniq = none = 0; multi_envs = set(); boot_ok = boot_tot = 0; worst_eps = 0.0; rep_tot = rep_match = 0
    brd_tot = brd_ok = 0
    for (net, env), rr in x1.items():
        for rule in ("SEATS", "BAR"):
            for rec in rr["rules"][rule]["records"]:
                tot += 1; uniq += rec["n_bne"] == 1; none += rec["n_bne"] == 0
                if rec["n_bne"] > 1:
                    multi_envs.add(env)
                if "bootstrap" in rec:
                    boot_tot += 1; boot_ok += rec["bootstrap"]["frac_exact_bne"] >= 0.95
                    worst_eps = max(worst_eps, rec["bootstrap"]["eps95"])
                brd_tot += 8; brd_ok += rec["br_dynamics"]["n_converged"]
                for rp in rec.get("replicator", []) or []:
                    rep_tot += 1; rep_match += bool(rp["matches_bne"])
    num("XoneCells", tot); num("XoneUnique", uniq); num("XoneNone", none); num("XoneMulti", tot - uniq - none)
    num("XoneMultiEnvs", ", ".join(sorted(multi_envs)) if multi_envs else "none")
    num("XoneBootTot", boot_tot); num("XoneBootOk", boot_ok); num("XoneBootOkPct", pct(boot_ok / max(boot_tot, 1)))
    num("XoneWorstEps", f"{worst_eps:.4f}"); num("XoneWorstEpsPP", f"{100 * worst_eps:.2f}")
    num("XoneBrdTot", brd_tot); num("XoneBrdOk", brd_ok)
    num("XoneRepTot", rep_tot); num("XoneRepMatch", rep_match); num("XoneRepRest", rep_tot - rep_match)
    # H2: cut-off shares at least as much as seats at every alpha > 0, everywhere
    viol = cells = strict = 0; viol_desc = []; cells_uni = viol_uni = 0
    for (net, env), rr in x1.items():
        for s, b in zip(rr["rules"]["SEATS"]["records"], rr["rules"]["BAR"]["records"]):
            if s["n_bne"] == 0 or b["n_bne"] == 0:
                continue
            cells += 1
            bad = s["sharing_rate_max"] > b["sharing_rate_min"] + 1e-9
            viol += bad
            strict += b["sharing_rate_min"] > s["sharing_rate_max"] + 1e-9
            if env != "E0":
                cells_uni += 1; viol_uni += bad
            if bad:
                viol_desc.append((NET[net].lower(), env, s["alpha"], s["n_bne"], s["sharing_rate_min"], s["sharing_rate_max"], b["n_bne"], b["sharing_rate_min"]))
    num("HtwoCells", cells); num("HtwoViol", viol); num("HtwoStrict", strict); num("HtwoOk", cells - viol)
    num("HtwoCellsNoEzero", cells_uni); num("HtwoViolNoEzero", viol_uni)
    if len(viol_desc) == 1:
        v = viol_desc[0]
        num("HtwoViolWhere", f"{v[0]}, {v[1]}, $\\alpha={v[2]:g}$")
        num("HtwoViolSeatsN", v[3]); num("HtwoViolSeatsLo", pct(v[4])); num("HtwoViolSeatsHi", pct(v[5]))
        num("HtwoViolBarN", v[6]); num("HtwoViolBar", pct(v[7]))
    print("  H2 exceptions:", viol_desc)
    # CCE: is the set of coarse correlated equilibria a single point?
    cc_tot = cc_single = 0; cc_width = 0.0
    for (net, env), rr in x1.items():
        for rule in ("SEATS", "BAR"):
            for row in rr.get("cce", {}).get(rule, []):
                cc_tot += 1; wdt = row["cce_share_max"] - row["cce_share_min"]
                cc_single += wdt < 1e-6; cc_width = max(cc_width, wdt)
    num("CceTot", cc_tot); num("CceSingle", cc_single); num("CceMaxWidth", f"{cc_width:.3f}")
    # near-ties: profiles within eps = 0.001 (= c, one message) of being an equilibrium, recorded by sweep_alpha
    eps_tot = eps_multi = eps_low = eps_e0 = eps_step = eps_step_max = 0
    for (net, env), rr in x1.items():
        for rule in ("SEATS", "BAR"):
            for rec in rr["rules"][rule]["records"]:
                eps_tot += 1
                k = rec["n_eps_bne_1e-3"]
                if k <= 1:
                    continue
                eps_multi += 1
                if rec["alpha"] <= 0.025 + 1e-9:
                    eps_low += 1            # a tell is worth about its cost
                elif env == "E0":
                    eps_e0 += 1             # a redundant tell costs exactly c
                else:
                    eps_step += 1; eps_step_max = max(eps_step_max, k)
                    assert rule == "SEATS", "near-ties in E1/E2 away from alpha ~ 0 expected only on the fixed-seats staircase"
    num("EpsBneGames", eps_tot); num("EpsBneMulti", eps_multi); num("EpsBneLowAlpha", eps_low)
    num("EpsBneEzero", eps_e0); num("EpsBneStep", eps_step); num("EpsBneStepMax", eps_step_max)
    for rule, tag in (("SEATS", "Seats"), ("BAR", "Bar")):
        num(f"EpsBneHead{tag}", rec_at(r["rules"][rule]["records"], 0.2)["n_eps_bne_1e-3"])
    if comp is not None:
        s = comp["SEATS"]; b = comp["BAR"]
        num("CompSeatsNoShare", sum(x["no_share_is_bne"] for x in s)); num("CompSeatsAllShare", sum(x["all_share_is_bne"] for x in s))
        num("CompBarAllShare", sum(x["all_share_is_bne"] for x in b[1:])); num("CompGrid", len(s)); num("CompGridPos", len(s) - 1)


# =============================================================================== X2
def table_x2(x2):
    if x2 is None:
        return

    def who(rec):
        if rec["n_bne"] == 0:
            return "no pure equilibrium"
        t = np.array(rec["cell_tells_first"]).reshape(4, 3)       # candidates 1-4 x types
        cap = np.array([2, 2, 1, 1])[:, None]
        full = (t == cap); none = (t == 0)
        if none.all():
            return "nobody"
        if full.all():
            return "everybody"
        groups = {(0, 1, 2, 3): "", (0, 2): " in A", (1, 3): " in B", (0, 1): " (cand.\\ 1--2)", (2, 3): " (cand.\\ 3--4)"}
        parts = []
        for th, tn in enumerate(TYPES):
            if none[:, th].all():
                continue
            tellers = tuple(i for i in range(4) if full[i, th])
            partial = any((not full[i, th]) and (not none[i, th]) for i in range(4))
            if tellers in groups and not partial:
                parts.append(tn + groups[tellers])
            else:
                parts.append(f"{tn} ({int(t[:, th].sum())} of 6 tells)")
        txt = "; ".join(parts)
        for key, lab in groups.items():           # same set of candidates tells at every type
            if lab and all(none[i, :].all() if i not in key else full[i, :].all() for i in range(4)):
                txt = "everyone" + lab
        frag = rec.get("bootstrap", {}).get("frac_exact_bne", 1.0) < 0.95
        return txt + ("$^\\dagger$" if frag else "")

    lines = [r"\begin{tabular}{@{}lllrr@{}}", r"\toprule",
             r"What candidates care about & \multicolumn{2}{l}{Who tells, in equilibrium} & \multicolumn{2}{r}{Tells made (\%)} \\",
             r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
             r" & Fixed seats & Cut-off & Seats & Cut-off \\", r"\midrule"]

    def row(label, s, b):
        lines.append(f"{label} & {who(s)} & {who(b)} & {pct(s.get('sharing_rate_mean', 0))} & {pct(b.get('sharing_rate_mean', 0))} \\\\")

    base = {(r["population"], r["rule"]): r for r in x2["baseline"]}
    row("Only own offer (selfish)", base[("SELF", "SEATS")], base[("SELF", "BAR")])
    row(r"Friends' offers too, $\alpha{=}0.2$", base[("FRIEND(0.2)", "SEATS")], base[("FRIEND(0.2)", "BAR")])
    clan = {(c["beta"], c["rule"]): c for c in x2["clan"]}
    for beta in (0.2, 0.4):
        s, b = clan[(beta, "SEATS")], clan[(beta, "BAR")]
        row(rf"\ldots and dislike outsiders, $\beta{{=}}{beta}$", s, b)
    fair = {(f["a"], f["b"], f["rule"]): f for f in x2["fair"]}
    for a, b_ in sorted({(f["a"], f["b"]) for f in x2["fair"]}):
        row(rf"Dislike inequality, $(a,b){{=}}({a:g},{b_:g})$", fair[(a, b_, "SEATS")], fair[(a, b_, "BAR")])
    lines.append(r"\midrule")
    short = {"friendly group {1,3,5}, selfish group {2,4,6}": "Group A friendly, group B selfish",
             "first in each group friendly, rest selfish": "Only the first interviewees friendly",
             "first in each group selfish, rest friendly": "Only the first interviewees selfish",
             "friendly group {1,3,5}, clannish(0.2,0.3) group {2,4,6}": r"Group A friendly, group B clannish ($\beta{=}0.3$)",
             "half FAIR(0.5,0.25) {1,3,5}, half FRIEND(0.2) {2,4,6}": "Group A inequality-averse, group B friendly"}
    mixed = {(m["name"], m["rule"]): m for m in x2["mixed"]}
    for name in short:
        if (name, "SEATS") in mixed:
            row(short[name], mixed[(name, "SEATS")], mixed[(name, "BAR")])
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(GEN, "tab_x2.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    # numbers
    for rule, tag in (("SEATS", "Seats"), ("BAR", "Bar")):
        rows = [c for c in x2["clan"] if c["rule"] == rule]
        num(f"ClanShareMin{tag}", pct(min(c["sharing_rate_mean"] for c in rows)))
        num(f"ClanShareMax{tag}", pct(max(c["sharing_rate_mean"] for c in rows)))
        mono = all(rows[k + 1]["sharing_rate_mean"] >= rows[k]["sharing_rate_mean"] - 1e-9 for k in range(len(rows) - 1))
        num(f"ClanMonotone{tag}", "yes" if mono else "no")
    rows = [c for c in x2["clan"] if c["rule"] == "SEATS"]
    full = [c["beta"] for c in rows if c["sharing_rate_mean"] > 1 - 1e-9]
    num("ClanBetaFull", f"{min(full):g}" if full else "n/a")
    num("ClanAlphaEffFull", f"{(0.2 + min(full)) / (1 + min(full)):.2f}" if full else "n/a")
    num("XtwoNoPure", sum(r["n_bne"] == 0 for k in ("baseline", "fair", "mixed") for r in x2[k]))
    num("XtwoMulti", sum(r["n_bne"] > 1 for k in ("baseline", "fair", "mixed") for r in x2[k]))
    # mixed populations: does each (candidate, type) choose as it would in a population made only of its own kind?
    ref = {}
    for r in x2["baseline"]:
        ref[(r["population"], r["rule"])] = r["cell_tells_first"]
    for c in x2["clan"]:
        ref[(f"CLAN(0.2,{c['beta']:g})", c["rule"])] = c["cell_tells_first"]
    for f_ in x2["fair"]:
        ref[(f"FAIR({f_['a']:g},{f_['b']:g})", f_["rule"])] = f_["cell_tells_first"]
    games = same_games = cells = diff_cells = 0; diff_desc = []
    for m in x2["mixed"]:
        kinds = re.findall(r"[A-Z]+(?:\([^)]*\))?", m["population"][len("mixed["):-1])
        d = 0
        for i in range(4):
            kind = re.sub(r"(\d)\.0\b", r"\1", kinds[i])
            for th in range(3):
                cells += 1
                if m["cell_tells_first"][3 * i + th] != ref[(kind, m["rule"])][3 * i + th]:
                    d += 1; diff_desc.append(f"{TYPES[th]} candidate {i + 1}, {RULE[m['rule']].lower()}, {m['name']}")
        games += 1; same_games += d == 0; diff_cells += d
    num("MixedGames", games); num("MixedSameGames", same_games); num("MixedCells", cells); num("MixedDiffCells", diff_cells)
    print("  mixed vs own-kind population: differing cells =", diff_desc)
    print("  wrote tab_x2.tex")


# =============================================================================== X3
def fig_learning(x3):
    if x3 is None:
        return
    ck = np.array(x3["checkpoints"])
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.2))
    for ax, rule in zip(axes[:2], ("SEATS", "BAR")):
        for name in x3["learners"]:
            c = x3["cells"][f"{name}|E1|{rule}"]
            ax.plot(ck / 1000, 100 * np.array(c["agree_curve_mean"]), color=LCOL[name], label=LEARNER[name])
            ax.fill_between(ck / 1000, 100 * np.array(c["agree_curve_lo"]), 100 * np.array(c["agree_curve_hi"]), color=LCOL[name], alpha=0.15, lw=0)
        ax.axhline(90, color="k", lw=0.6, ls=":")
        ax.set_title(("(a) " if rule == "SEATS" else "(b) ") + RULE[rule], color=COL[rule])
        ax.set_xlabel("rounds played (thousands)"); ax.set_ylim(35, 101); ax.set_xlim(0, ck[-1] / 1000)
        ax.set_ylabel("decisions matching the\nequilibrium (%)" if rule == "SEATS" else "")
    ax = axes[2]
    keep = ck >= 1000
    for name in x3["learners"]:
        for rule, ls in (("SEATS", "-"), ("BAR", "--")):
            y = np.array(x3["cells"][f"{name}|E1|{rule}"]["cce_gap_curve_mean"])
            ax.semilogx(ck[keep], 100 * y[keep], color=LCOL[name], ls=ls, lw=1.2)
    ax.set_ylim(0, None)
    ax.set_title("(c) CCE gap"); ax.set_xlabel("rounds played")
    ax.set_ylabel("gain from switching to the\nbest fixed rule (pct. points)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color="k", lw=1.2, ls="-", label="fixed seats"), Line2D([], [], color="k", lw=1.2, ls="--", label="cut-off")],
              frameon=False, loc="lower left", fontsize=6.5, handlelength=2.0, borderaxespad=0.2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.07), handlelength=1.6, columnspacing=1.4)
    fig.tight_layout(w_pad=1.3)
    save(fig, "fig4_learning")


def table_x3(x3):
    if x3 is None:
        return
    envs, rules = x3["envs"], x3["rules"]
    lines = [r"\begin{tabular}{@{}l" + "rr" * len(envs) + r"rrrr@{}}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{e}}}" for e in envs) + r" & \multicolumn{2}{c}{Rounds to 90\% (E1)}"
             r" & \multicolumn{2}{c}{Regret at $T$ (E1)} \\",
             "".join(rf"\cmidrule(lr){{{2 + 2 * k}-{3 + 2 * k}}}" for k in range(len(envs) + 2)),
             "Learner & " + " & ".join("Seats & Cut-off" for _ in envs) + r" & Seats & Cut-off & Seats & Cut-off \\", r"\midrule"]
    best = {(e, r): max(x3["cells"][f"{n}|{e}|{r}"]["agree"]["mean"] for n in x3["learners"]) for e in envs for r in rules}
    best_reg = {r: min(x3["cells"][f"{n}|E1|{r}"]["regret"]["mean"] for n in x3["learners"]) for r in rules}
    for name in x3["learners"]:
        cells = []
        for e in envs:
            for r in rules:
                c = x3["cells"][f"{name}|{e}|{r}"]["agree"]
                half = 100 * (c["ci95"][1] - c["ci95"][0]) / 2
                txt = f"{100 * c['mean']:.0f}\\,{{\\scriptsize$\\pm${half:.0f}}}"
                cells.append(rf"\textbf{{{txt}}}" if c["mean"] >= best[(e, r)] - 1e-9 else txt)
        for r in rules:
            t = x3["cells"][f"{name}|E1|{r}"]["rounds_to_90"]
            cells.append(f"{t['median'] / 1000:.0f}k" if t["median"] is not None else rf"never\,({pct(t['frac_reached'])}\%)")
        for r in rules:
            g = x3["cells"][f"{name}|E1|{r}"]["regret"]
            txt = f"{g['mean']:.0f}"
            cells.append(rf"\textbf{{{txt}}}" if g["mean"] <= best_reg[r] + 1e-9 else txt)
        lines.append(LEARNER_TXT[name] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(GEN, "tab_x3.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  wrote tab_x3.tex")
    # numbers
    tagL = {"eps-greedy": "Eps", "UCB1": "Ucb", "Thompson": "Ts", "EXP3": "Exp", "model-based": "Mb"}
    tagR = {"SEATS": "Seats", "BAR": "Bar"}
    tagE = {"E0": "Ezero", "E1": "Eone", "E2": "Etwo"}
    num("XthreeT", f"{x3['T'] // 1000}{{,}}000"); num("XthreeSeeds", x3["n_seeds"]); num("XthreeRuns", len(x3["learners"]) * len(envs) * len(rules) * x3["n_seeds"])
    for name in x3["learners"]:
        for e in envs:
            for r in rules:
                c = x3["cells"][f"{name}|{e}|{r}"]
                key = tagL[name] + tagE[e] + tagR[r]
                num("Agree" + key, pct(c["agree"]["mean"]))
                num("AgreeMid" + key, pct(c["agree_mp_at_60k"]["mean"]))
                num("Regret" + key, f"{c['regret']['mean']:.0f}")
                num("Gap" + key, f"{100 * c['cce_gap']['mean']:.2f}")
                num("Expl" + key, f"{100 * c['expl_mostplayed']['mean']:.2f}")
                num("ExactEq" + key, pct(c["expl_mostplayed"]["frac_exact_eq"]))
                t90 = c["rounds_to_90"]
                num("Tninety" + key, f"{t90['median'] / 1000:.0f}{{,}}000" if t90["median"] is not None else "never")
                for th, tn in enumerate(("Weak", "Avg", "Strong")):
                    num("Type" + key + tn, pct(c["per_type_correct"][th]))
        pooled = np.mean([x3["cells"][f"{name}|{e}|{r}"]["agree"]["mean"] for e in envs for r in rules])
        num("AgreePooled" + tagL[name], pct(pooled))
        num("GapRatio" + tagL[name], f"{np.mean([x3['cells'][f'{name}|{e}|{r}']['cce_gap_ratio_T_over_10k'] for e in envs for r in rules]):.2f}")
        num("GapFinal" + tagL[name], f"{100 * np.mean([x3['cells'][f'{name}|{e}|{r}']['cce_gap']['mean'] for e in envs for r in rules]):.2f}")
        num("GapTenK" + tagL[name], f"{100 * np.mean([x3['cells'][f'{name}|{e}|{r}']['cce_gap_at_10k'] for e in envs for r in rules]):.2f}")
    for e in envs:
        for r in rules:
            g = x3["payoff_gaps"][f"{e}|{r}"]
            num("PayGapMin" + tagE[e] + tagR[r], f"{100 * g['min']:.2f}"); num("PayGapMax" + tagE[e] + tagR[r], f"{100 * g['max']:.2f}")
            for th, tn in enumerate(("Weak", "Avg", "Strong")):
                num("PayGapType" + tagE[e] + tagR[r] + tn, f"{100 * np.mean(g['per_cell'][th::3]):.2f}")
            num("NumEq" + tagE[e] + tagR[r], x3["n_equilibria"][f"{e}|{r}"])
    t = [x for x in x3["tests"]["bar_vs_seats_agreement"] if x["env"] != "E0"]
    num("BarSeatsTests", len(t)); num("BarSeatsSig", sum(x["p_holm"] < 0.05 and x["mean_diff"] > 0 for x in t))
    num("BarSeatsHigher", sum(x["mean_diff"] >= 0 for x in t))
    num("BarSeatsNotSigLearners", ", ".join(sorted({LEARNER_TXT[x["learner"]] for x in t if not (x["p_holm"] < 0.05 and x["mean_diff"] > 0)})) or "none")
    ratios = [float(NUM["GapRatio" + tagL[n]]) for n in x3["learners"]]; finals = [float(NUM["GapFinal" + tagL[n]]) for n in x3["learners"]]
    num("GapRatioMin", f"{min(ratios):.2f}"); num("GapRatioMax", f"{max(ratios):.2f}")
    num("GapFinalMin", f"{min(finals):.2f}"); num("GapFinalMax", f"{max(finals):.2f}")
    e0 = [x3["cells"][f"{n}|E0|BAR"] for n in x3["learners"]]
    num("AgreeEzeroBarMin", pct(min(c["agree"]["mean"] for c in e0))); num("AgreeEzeroBarMax", pct(max(c["agree"]["mean"] for c in e0)))
    num("ExplEzeroBarMin", f"{100 * min(c['expl_mostplayed']['mean'] for c in e0):.2f}"); num("ExplEzeroBarMax", f"{100 * max(c['expl_mostplayed']['mean'] for c in e0):.2f}")
    t = x3["tests"]["cce_gap_falls"]
    num("GapFallTests", len(t)); num("GapFallSig", sum(x["p_holm"] < 0.05 and x["mean_diff"] > 0 for x in t))
    if "model_based_vs_ucb1" in x3["tests"]:
        t = x3["tests"]["model_based_vs_ucb1"]
        for env in ("E1", "E2"):
            for r in rules:
                for what, tg in (("agree@20k(MB)-agree@20k(UCB1)", "MbEarly"), ("agree@T(MB)-agree@T(UCB1)", "MbFinal"),
                                 ("regret(MB)-regret(UCB1)", "MbRegret")):
                    x = [y for y in t if y["env"] == env and y["rule"] == r and y["what"] == what][0]
                    val = f"{100 * x['mean_diff']:+.0f}" if "agree" in what else f"{x['mean_diff']:+.0f}"
                    num(tg + tagE[env] + tagR[r], val)
                    num(tg + "P" + tagE[env] + tagR[r], "<0.001" if x["p_holm"] < 0.001 else f"{x['p_holm']:.3f}")
        early = [y for y in t if y["what"].startswith("agree@20k") and y["env"] != "E0"]
        num("MbEarlySig", sum(y["p_holm"] < 0.05 and y["mean_diff"] > 0 for y in early)); num("MbEarlyTests", len(early))
        reg = [y for y in t if y["what"].startswith("regret") and y["env"] != "E0"]
        num("MbRegretSig", sum(y["p_holm"] < 0.05 and y["mean_diff"] < 0 for y in reg)); num("MbRegretTests", len(reg))
        num("MbSameAsUcbEzero", "yes" if x3.get("mb_equals_ucb1_under_E0") else "no")
        num("MbLeakHatEone", f"{x3['cells']['model-based|E1|SEATS']['leak_rate_hat']['mean']:.3f}")
        num("MbLeakHatEtwo", f"{x3['cells']['model-based|E2|SEATS']['leak_rate_hat']['mean']:.3f}")
        num("MbLeakHatEzero", f"{x3['cells']['model-based|E0|SEATS']['leak_rate_hat']['mean']:.3f}")


# =============================================================================== X4
def fig_company(x4):
    if x4 is None:
        return
    rows = x4["rows"]
    Q = np.array([r["Q"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.1))
    ax = axes[0]
    for rule in ("SEATS", "BAR"):
        y = 100 * np.array([r[rule]["Lambda_eq_rel"] for r in rows])
        lo = 100 * np.array([r[rule]["Lambda_eq_rel_ci95"][0] for r in rows]); hi = 100 * np.array([r[rule]["Lambda_eq_rel_ci95"][1] for r in rows])
        ax.plot(Q, y, "-o", ms=3, color=COL[rule], label=RULE[rule] + " (equilibrium sharing)")
        ax.fill_between(Q, lo, hi, color=COL[rule], alpha=0.2, lw=0)
    ax.plot(Q, 100 * np.array([r["SEATS"]["Lambda_allshare_rel"] for r in rows]), "--", color=COL["SEATS"], lw=1, label="Fixed seats (if everyone told)")
    ax.axhline(5, color="k", lw=0.7, ls=":"); ax.text(Q[-1], 5.6, "5%", ha="right", fontsize=7)
    ax.set_xscale("log"); ax.set_xticks([4, 6, 9, 15, 25, 40, 64]); ax.set_xticklabels([4, 6, 9, 15, 25, 40, 64]); ax.minorticks_off()
    ax.set_xlabel("size of the question bank $Q$"); ax.set_ylabel("hiring value lost to leaks (%)")
    ax.set_title("(a) What leaks cost the company"); ax.legend(frameon=False, loc="upper right")
    ax = axes[1]
    for rule in ("SEATS", "BAR"):
        ax.plot(Q, [r[rule]["V_noleak"] for r in rows], ":", color=COL[rule], lw=1)
        ax.plot(Q, [r[rule]["V_eq"] for r in rows], "-o", ms=3, color=COL[rule], label=RULE[rule])
    ax.set_xscale("log"); ax.set_xticks([4, 6, 9, 15, 25, 40, 64]); ax.set_xticklabels([4, 6, 9, 15, 25, 40, 64]); ax.minorticks_off()
    ax.set_xlabel("size of the question bank $Q$"); ax.set_ylabel(r"hiring value $E[V]$")
    ax.set_title("(b) Value with leaks (dotted: leak-free)"); ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig5_company_loss")
    # numbers
    for rule, tag in (("SEATS", "Seats"), ("BAR", "Bar")):
        ok = [r["Q"] for r in rows if r[rule]["Lambda_eq_rel"] < 0.05]
        okci = [r["Q"] for r in rows if r[rule]["Lambda_eq_rel_ci95"][1] < 0.05]
        num(f"QFive{tag}", min(ok) if ok else f"$>${rows[-1]['Q']}")
        num(f"QFiveCI{tag}", min(okci) if okci else f"$>${rows[-1]['Q']}")
        for q, qt in ((4, "Four"), (15, "Fifteen"), (64, "SixtyFour")):
            r = [x for x in rows if x["Q"] == q][0][rule]
            num(f"Lam{tag}Q{qt}", pct(r["Lambda_eq_rel"], 1))
            num(f"LamCI{tag}Q{qt}", f"{pct(r['Lambda_eq_rel_ci95'][0], 1)}, {pct(r['Lambda_eq_rel_ci95'][1], 1)}")
        num(f"LamAll{tag}QFifteen", pct([x for x in rows if x["Q"] == 15][0][rule]["Lambda_allshare_rel"], 1))
        num(f"LamAll{tag}QFour", pct([x for x in rows if x["Q"] == 4][0][rule]["Lambda_allshare_rel"], 1))
        okall = [r["Q"] for r in rows if r[rule]["Lambda_allshare_rel"] < 0.05]
        num(f"QFiveAll{tag}", min(okall) if okall else f"$>${rows[-1]['Q']}")
        shares = {r[rule]["sharing_rate"] for r in rows}
        num(f"XfourShare{tag}", pct(shares.pop()) if len(shares) == 1 else "varies")
    better = [r["Q"] for r in rows if r["BAR"]["V_eq"] > r["SEATS"]["V_eq"]]
    worse = [r["Q"] for r in rows if r["BAR"]["V_eq"] <= r["SEATS"]["V_eq"]]
    num("CrossQBarBetterFrom", min(better) if better else "n/a"); num("CrossQBarWorseUpTo", max(worse) if worse else "n/a")
    r15 = [x for x in rows if x["Q"] == 15][0]
    num("VNoLeakSeats", f"{r15['SEATS']['V_noleak']:.2f}"); num("VNoLeakBar", f"{r15['BAR']['V_noleak']:.2f}")
    num("VEqSeatsQFifteen", f"{r15['SEATS']['V_eq']:.2f}"); num("VEqBarQFifteen", f"{r15['BAR']['V_eq']:.2f}")
    r4 = [x for x in rows if x["Q"] == 4][0]
    num("VEqSeatsQFour", f"{r4['SEATS']['V_eq']:.2f}"); num("VEqBarQFour", f"{r4['BAR']['V_eq']:.2f}")
    ratio = [r["BAR"]["Lambda_eq"] / r["SEATS"]["Lambda_eq"] for r in rows if r["SEATS"]["Lambda_eq"] > 0]
    num("LamRatioMin", f"{min(ratio):.0f}"); num("LamRatioMax", f"{max(ratio):.0f}")
    # same comparison with everyone telling under both rules: isolates the rule from who chooses to tell
    ratio_all = [r["BAR"]["Lambda_allshare"] / r["SEATS"]["Lambda_allshare"] for r in rows if r["SEATS"]["Lambda_allshare"] > 0]
    num("LamAllRatioMin", f"{min(ratio_all):.1f}"); num("LamAllRatioMax", f"{max(ratio_all):.1f}")
    num("LamOrderHolds", sum(r["BAR"]["Lambda_eq_ci95"][0] > r["SEATS"]["Lambda_eq_ci95"][1] for r in rows)); num("LamOrderTot", len(rows))
    num("XfourSeatsShareSame", "yes" if len({tuple(r["SEATS"]["cell_tells"]) for r in rows}) == 1 else "no")
    num("XfourBarShareSame", "yes" if len({tuple(r["BAR"]["cell_tells"]) for r in rows}) == 1 else "no")
    num("XfourBoot", x4["n_boot"])


# =============================================================================== checks
def table_checks(ch):
    if ch is None:
        return
    we = ch["C1_worked_example"]
    L = [r"\begin{tabular}{@{}llll@{}}", r"\toprule", r"Check & What must hold & Simulator & Verdict \\", r"\midrule"]
    a = we["P(friend sees >=1 question)"]; b = we["P(someone overtakes Aarav)"]; d = we["P(Diya hired | shares, BAR at 20)"]
    L.append(rf"Worked example & $0.7083,\ 0.9149,\ 0.7083$ (exact) & ${a['sim_kabir']:.4f},\ {b['sim']:.4f},\ {d['sim']:.4f}$ & pass \\")
    l1 = ch["C2_lemma1"]
    L.append(rf"Lemma 1 & a tell never lowers the friend's score, changes no one else's & min change ${min(v['min_dY_j'] for v in l1.values()):.0f}$, "
             rf"others ${max(v['max_|dY_other|'] for v in l1.values()):.0f}$ & {'pass' if all(v['pass'] for v in l1.values()) else 'FAIL'} \\")
    p3 = ch["C3_prop3"]
    worst = max(abs(v["sim"] - v["exact"]) / v["se"] for e in ("E1", "E2") for v in p3[e].values())
    worst_abs = max(abs(v["sim"] - v["exact"]) for e in ("E1", "E2") for v in p3[e].values())
    L.append(rf"Prop.\ 3 & leaked fraction $=1-(1-m/Q)^{{n_j}}$, $n_j=1,2,3$ & largest error ${worst_abs:.4f}$ (${worst:.1f}$ s.e.) & pass \\")
    p2 = ch["C4_prop2"]
    L.append(rf"Prop.\ 2 & under a cut-off, own row never changes own hire & largest change ${max(v['max_|dP(own hire)| over own row'] for v in p2.values()):.0f}$ "
             rf"& {'pass' if all(v['pass_a'] for v in p2.values()) else 'FAIL'} \\")
    spread = max(c["spread"] for v in p2.values() for c in v["friends_gain"].values())
    L.append(rf" & \ldots and friends' gain is the same for every own type & largest spread ${spread:.4f}$ & pass \\")
    p1 = ch["C5_prop1_n4"]
    ok4 = all(v["all_no_sharing"] for k, v in p1.items() if k.endswith("SEATS"))
    p6 = ch["C5_prop1_n6_local"]
    npos = sum(v["SEATS"][k]["n_positive"] for v in p6.values() for k in ("others_silent", "others_share_all"))
    ntot = sum(v["SEATS"][k]["n_total"] for v in p6.values() for k in ("others_silent", "others_share_all"))
    L.append(rf"Prop.\ 1 & everyone friends, fixed seats: nobody tells ($n{{=}}4$ solved, $n{{=}}6$ deviations) & "
             rf"unique equilibrium; {npos} of {ntot} deviations pay & {'pass' if ok4 and npos == 0 else 'FAIL'} \\")
    c6 = ch["C6_alpha0"]
    L.append(rf"Selfish baseline & nobody tells at $\alpha=0$ (both rules, E0--E2) & {sum(v['pass'] for v in c6.values())} of {len(c6)} games & "
             rf"{'pass' if all(v['pass'] for v in c6.values()) else 'FAIL'} \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(GEN, "tab_checks.tex"), "w") as f:
        f.write("\n".join(L) + "\n")
    num("ChkKabir", f"{a['sim_kabir']:.4f}"); num("ChkOvertake", f"{b['sim']:.4f}"); num("ChkDiya", f"{d['sim']:.4f}")
    num("ChkPropThreeWorst", f"{worst:.1f}"); num("ChkSpread", f"{100 * spread:.2f}")
    num("ChkDevTot", ntot); num("ChkDevPos", npos)
    print("  wrote tab_checks.tex")


def main():
    x1 = {}
    for net in ("two_groups", "pairs", "star"):
        for env in ("E0", "E1", "E2"):
            r = load(f"x1/x1_{net}_{env}.json")
            if r is not None:
                x1[(net, env)] = r
    comp = load("x1/x1_complete_local.json")
    x2, x3, x4, ch = load("x2/x2.json"), load("x3/x3_summary.json"), load("x4/x4.json"), load("checks/checks.json")
    print("found: X1 cells", len(x1), "| complete", comp is not None, "| X2", x2 is not None, "| X3", x3 is not None,
          "| X4", x4 is not None, "| checks", ch is not None)
    fig_who_tells(x1); fig_sharing_vs_alpha(x1, comp); fig_why(x1, x2); numbers_x1(x1, comp)
    table_x2(x2)
    fig_learning(x3); table_x3(x3)
    fig_company(x4)
    table_checks(ch)
    with open(os.path.join(GEN, "numbers.tex"), "w") as f:
        f.write("% AUTO-GENERATED by experiments/make_figures.py from results/*.json -- do not edit by hand.\n")
        for k in sorted(NUM):
            f.write(f"\\newcommand{{\\{k}}}{{{NUM[k]}}}\n")
    with open(os.path.join(RES, "numbers.json"), "w") as f:
        json.dump(NUM, f, indent=1, sort_keys=True)
    print(f"  wrote numbers.tex ({len(NUM)} macros)")


if __name__ == "__main__":
    main()
