#!/usr/bin/env python3
"""Data figures of the article, drawn in one style from the deliverable data and the generated tables.

Usage: python3 scripts/make_figs_v4.py [phenomenon intervention scores counterfactual density check ladder ar_time public]
       (all nine if none is named; run after the table scripts, which write the tables read here)

Inputs (numbers are read, never typed in):
  data33_17/ED_MECH.json, ED_HIST.json, Y2_RESULTS.json, TABLES_P1.md, derived_v3.json   Figures 1, 4, 6
  data33_22/derived_s22.json (ladder statistics written by make_s22.py)                  Figure 3
  tables/tab_counterfactual.tex, tab_check.tex, tab_ladder.tex, main_mean_time.csv,
  tables/supp_s23_public.tex                                                              Figures 5, 7, 9, 10, 11
Outputs: figures/fig_phenomenon.pdf, fig_intervention.pdf, fig_scores_both.pdf, fig_counterfactual.pdf,
         fig_density.pdf, fig_check.pdf, fig_ladder.pdf, fig_ar_time.pdf, fig_public.pdf

Style: one meaning per colour across all figures (vermillion: per-call or reversed FoundationPose scores;
blue: joint scoring; green: MegaPose; grey: baselines and context), charcoal text, direct labels instead of
legends where they fit, no gridlines or frames that carry no data. The hues are Okabe-Ito and pass the CVD
and contrast checks of the dataviz validator.
"""
import json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory as btf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = os.path.join(ROOT, "tables")
F = os.path.join(ROOT, "figures")
W2 = 7.1    # text width of the 5p two-column layout (inches)
W1 = 3.45   # column width

C = dict(fp="#D55E00", joint="#0072B2", mega="#009E73", base="#8C8C8C", ink="#3C3C3C", muted="#666666",
         light="#D4D4D4", grid="#EBEBEB", tint="#FBEDE3")
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.2, "axes.titlesize": 7.4, "axes.labelsize": 7.2,
    "xtick.labelsize": 6.7, "ytick.labelsize": 6.7, "legend.fontsize": 6.6, "legend.frameon": False,
    "text.color": C["ink"], "axes.labelcolor": C["ink"], "xtick.color": C["ink"], "ytick.color": C["ink"],
    "axes.edgecolor": "#7A7A7A", "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.2, "pdf.fonttype": 42, "axes.titlepad": 3,
})
DS = [("lmo", "LM-O"), ("tless", "T-LESS"), ("tudl", "TUD-L"), ("icbin", "IC-BIN"), ("ycbv", "YCB-V")]
BINS = ["1", "2–3", "4–7", "8–15", "≥16"]


def load(rel):
    return json.load(open(os.path.join(ROOT, rel)))


def num(s):
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s).replace("$-$", "-").replace("$", "").strip()
    return None if s in ("--", "") else float(s)


def rows(path):
    out = []
    for line in open(path, encoding="utf8"):
        line = line.strip()
        if line.startswith("%") or "&" not in line or line.startswith("\\multicolumn") or "Repair (gain" in line:
            continue
        out.append([c.strip() for c in line.rstrip("\\").rstrip().rstrip("\\").split("&")])
    return out


def md_table(rel, title):
    out, on = [], False
    for line in open(os.path.join(ROOT, rel), encoding="utf8"):
        if line.startswith("## "):
            on = line[3:].strip() == title
            continue
        if on and line.startswith("|") and not line.startswith("| ---"):
            out.append([c.strip() for c in line.strip().strip("|").split("|")])
    return out[1:]


def save(fig, name):
    fig.savefig(os.path.join(F, name), pad_inches=0.0)
    plt.close(fig)


def dot(ax, x, y, col, ms=4.4, hollow=False, z=4, mk="o"):
    ax.plot([x], [y], mk, ms=ms, mfc="white" if hollow else col, mec=col if hollow else "white",
            mew=0.9 if hollow else 0.5, zorder=z, clip_on=False)


# ================================================================== Figure 1: the inversion and its cost
def fig_phenomenon():
    ed = load("data33_17/ED_MECH.json")
    ar = {r[0]: [float(x) for x in r[1:]] for r in md_table("data33_17/TABLES_P1.md", "E-A 与 X2 的 AR（百分点）")}
    w = ed["within_call_by_dataset"]
    across = {d: 100 * ed["cross_mask_by_dataset"][d]["raw"]["accuracy"] for d, _ in DS}
    npairs = {d: ed["cross_mask_by_dataset"][d]["raw"]["opposite_label_pairs"] for d, _ in DS}
    within = {d: 100 * w[d]["accuracy"] for d, _ in DS}
    across["all"] = 100 * ed["cross_mask_combined"]["raw"]["accuracy"]
    npairs["all"] = ed["cross_mask_combined"]["raw"]["opposite_label_pairs"]
    within["all"] = 100 * sum(v["correct_first"] for v in w.values()) / sum(v["opposite_label_pairs"] for v in w.values())
    keys = [d for d, _ in DS] + ["all"]
    names = [n for _, n in DS] + ["All five"]
    ys = [6.2 - i for i in range(5)] + [0.0]          # a small gap before the summary row

    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 2.5), gridspec_kw=dict(width_ratios=[1.0, 1.0], wspace=0.62))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.86, bottom=0.19)
    # (a) the same scorer, two kinds of pairs
    a.axvspan(0, 50, color=C["tint"], lw=0, zorder=0)
    a.axvline(50, color=C["muted"], lw=0.7, ls=(0, (4, 3)), zorder=1)
    for k, y in zip(keys, ys):
        a.plot([across[k], within[k]], [y, y], color=C["light"], lw=1.6, zorder=2, solid_capstyle="butt")
        dot(a, within[k], y, C["ink"], ms=4.6)
        dot(a, across[k], y, C["fp"], ms=4.6)
        a.text(across[k] + 3.0, y, "%.1f" % across[k], va="center", ha="left", fontsize=6.4, color=C["fp"])
        a.text(1.04, y, "{:,}".format(npairs[k]), transform=btf(a.transAxes, a.transData), va="center", ha="left",
               fontsize=6.2, color=C["muted"])
    a.text(1.04, ys[0] + 0.85, "pairs", transform=btf(a.transAxes, a.transData), va="center", ha="left", fontsize=6.1, color=C["muted"])
    a.text(across["lmo"], ys[0] + 0.85, "across calls", ha="left", va="center", fontsize=6.4, color=C["fp"])
    a.text(within["lmo"], ys[0] + 0.85, "within one call", ha="right", va="center", fontsize=6.4, color=C["ink"])
    a.text(25, -1.05, "inverted", ha="center", va="center", fontsize=6.3, color=C["fp"])
    a.set_xlim(0, 100); a.set_xticks([0, 25, 50, 75, 100]); a.set_ylim(-1.5, ys[0] + 2.3)
    a.set_yticks(ys); a.set_yticklabels(names); a.tick_params(axis="y", length=0)
    a.get_yticklabels()[-1].set_fontweight("bold")
    a.spines["left"].set_visible(False)
    a.set_xlabel("Correct pose scored higher (% of correct–incorrect pairs)")
    # (b) average recall per dataset
    spec = [("A0", C["base"], True, 4.4, "CNOS masks only"), ("A1", C["fp"], False, 4.6, "+ NOCTIS, per-call scores"),
            ("A4", C["joint"], False, 5.0, "+ NOCTIS, joint scoring"), ("A5", C["mega"], False, 3.0, "+ NOCTIS, MegaPose")]
    for i, (k, y) in enumerate(zip(keys, ys)):
        vals = [ar[r][i if k != "all" else -1] for r, *_ in spec]
        b.plot([min(vals), max(vals)], [y, y], color=C["light"], lw=1.6, zorder=2, solid_capstyle="butt")
        for (r, col, hol, ms, _), v in zip(spec, vals):
            dot(b, v, y, col, ms=ms, hollow=hol, z={"A0": 3, "A1": 5, "A4": 4, "A5": 6}[r])
        if vals[1] > 12:
            b.text(vals[1] - 3.0, y, "%.1f" % vals[1], va="center", ha="right", fontsize=6.4, color=C["fp"])
        else:
            b.text(vals[1] + 3.0, y, "%.1f" % vals[1], va="center", ha="left", fontsize=6.4, color=C["fp"])
    b.set_xlim(0, 100); b.set_xticks([0, 25, 50, 75, 100]); b.set_ylim(-1.5, ys[0] + 2.3)
    b.set_yticks(ys); b.set_yticklabels([n if n != "All five" else "Mean" for n in names]); b.tick_params(axis="y", length=0)
    b.get_yticklabels()[-1].set_fontweight("bold")
    b.spines["left"].set_visible(False)
    b.set_xlabel("Average recall of the best poses (AR, points)")
    handles = [Line2D([], [], marker="o", ls="none", mfc="white" if hol else col, mec=col if hol else "white",
                      mew=0.9 if hol else 0.5, ms=4.6, label=lab) for _, col, hol, ms, lab in spec]
    b.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, 1.0), ncol=2, handletextpad=0.1,
             columnspacing=0.7, borderaxespad=0, fontsize=6.2)
    for ax, letter, txt in ((a, "(a)", "FoundationPose orders the poses of one call, not of different calls"),
                            (b, "(b)", "Ranked by per-call scores, pooled masks lose recall")):
        ax.text(-0.215, 1.10, letter, transform=ax.transAxes, fontsize=7.6, fontweight="bold", va="bottom")
        ax.text(-0.14, 1.10, txt, transform=ax.transAxes, fontsize=7.0, va="bottom")
    save(fig, "fig_phenomenon.pdf")


# ================================================================== Figure 3: displacement ladders
def fig_intervention():
    lad = load("data33_22/derived_s22.json")["E3"]["ladders"]
    LAD = [("T_xy", "Lateral shift ($D_o$)"), ("R", "Rotation (degrees)"), ("T_z", "Shift along the ray ($D_o$)")]
    fig, axes = plt.subplots(1, 3, figsize=(W2, 1.75), sharey=True, gridspec_kw=dict(wspace=0.16))
    fig.subplots_adjust(left=0.1, right=0.985, top=0.93, bottom=0.24)
    for i, (ax, (key, xlabel)) in enumerate(zip(axes, LAD)):
        for scorer, col in (("fp_single", C["fp"]), ("megapose_single", C["mega"])):
            d = lad[key][scorer]
            xs = sorted(float(v) for v in d)
            st = [d[next(s for s in d if float(s) == x)] for x in xs]
            ys = [s["share_higher_pct"] for s in st]
            parts = [list(range(len(xs)))] if key != "T_z" else [[j for j, x in enumerate(xs) if x < 0], [j for j, x in enumerate(xs) if x > 0]]
            for p in parts:
                ax.plot([xs[j] for j in p], [ys[j] for j in p], color=col, lw=1.2, zorder=3)
                for j in p:
                    full = st[j]["n"] == max(s["n"] for s in st)
                    ax.plot([xs[j]], [ys[j]], "o", ms=3.4, mfc=col if full else "white", mec=col, mew=0.8, zorder=4)
            if key == "T_xy" and scorer == "megapose_single":
                for j, x in enumerate(xs):
                    if st[j]["n"] < 300:
                        ax.text(x, ys[j] + 8, "n=%d" % st[j]["n"], fontsize=5.9, color=C["muted"], ha="center")
        if key == "T_xy":
            ax.set_xscale("log"); ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1, 2, 4]); ax.set_xticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2", "4"])
        elif key == "R":
            ax.set_xscale("log"); ax.set_xticks([5, 10, 20, 45, 90, 180]); ax.set_xticklabels(["5", "10", "20", "45", "90", "180"])
        else:
            ax.set_xticks([-0.5, -0.2, 0, 0.2, 0.5]); ax.set_xticklabels(["−0.5", "−0.2", "0", "0.2", "0.5"])
            ax.axvline(0, color=C["light"], lw=0.6, zorder=1)
        ax.minorticks_off()
        ax.axhline(50, color=C["muted"], lw=0.6, ls=(0, (4, 3)), zorder=1)
        ax.set_ylim(-4, 106); ax.set_yticks([0, 50, 100]); ax.set_xlabel(xlabel)
        ax.text(0.0, 1.0, "(%s)" % "abc"[i], transform=ax.transAxes, fontsize=7.6, fontweight="bold", va="bottom", ha="left")
    axes[0].set_ylabel("Scored above the true\npose (% of instances)")
    a = axes[1]
    a.text(7.0, 84, "FoundationPose", color=C["fp"], fontsize=6.6, ha="left", va="center")
    a.text(11.5, 34, "MegaPose", color=C["mega"], fontsize=6.6, ha="left", va="center")
    save(fig, "fig_intervention.pdf")


# ================================================================== Figure 4: score distributions
def fig_scores():
    h = load("data33_17/ED_HIST.json")["scopes"]
    H = 2.15
    fig, axes = plt.subplots(2, 5, figsize=(W2, H), gridspec_kw=dict(wspace=0.16, hspace=1.45))
    fig.subplots_adjust(left=0.02, right=0.995, top=1 - 0.37 / H, bottom=0.30 / H)
    rowsp = (("s_raw", "(a)", "Per-call scores: incorrect poses score higher in every dataset"),
             ("s_joint", "(b)", "Joint scores: the two classes overlap once pooled over targets"))
    for r, (key, letter, text) in enumerate(rowsp):
        for ax, (d, name) in zip(axes[r], DS):
            e = np.array(h[d][key]["bin_edges"]); mid = 0.5 * (e[1:] + e[:-1]); wdt = e[1] - e[0]
            for cls, col, fill, alpha in (("correct", C["ink"], "#BDBDBD", 0.55), ("incorrect", C["fp"], C["fp"], 0.22)):
                c = np.array(h[d][key][cls]["counts"], float); c = c / (c.sum() * wdt)
                ax.fill_between(mid, c, step="mid", color=fill, alpha=alpha, lw=0)
                ax.step(mid, c, where="mid", color=col, lw=0.9)
                ax.axvline(h[d][key][cls]["median"], color=col, lw=0.7, ls=(0, (3, 2)))
            ax.set_title(name, fontsize=6.9, pad=4, color=C["ink"])
            ax.set_yticks([]); ax.spines["left"].set_visible(False)
            ax.tick_params(axis="x", labelsize=6.3)
        top = axes[r][0].get_position().y1
        fig.text(0.012, top + 0.19 / H, letter, fontsize=7.6, fontweight="bold", va="bottom")   # just above the dataset names
        fig.text(0.045, top + 0.19 / H, text, fontsize=7.0, va="bottom")
        axes[r][2].set_xlabel("FoundationPose score, %s" % ("per call" if key == "s_raw" else "joint call"), labelpad=1)
    a = axes[0][0]
    a.text(h["lmo"]["s_raw"]["correct"]["median"] - 4, a.get_ylim()[1] * 0.52, "correct", color=C["ink"], fontsize=6.4, ha="right")
    a.text(h["lmo"]["s_raw"]["incorrect"]["median"] - 6, a.get_ylim()[1] * 0.80, "incorrect", color=C["fp"], fontsize=6.4, ha="right")
    save(fig, "fig_scores_both.pdf")


# ================================================================== Figure 5: counterfactual repairs
def fig_counterfactual():
    R = {}
    for c in rows(os.path.join(T, "tab_counterfactual.tex")):
        if c[0] == "" or len(c) < 9:
            continue
        R[c[0]] = [num(x) for x in c[1:9]]
    order = [("Mask quality: true masks where covered", "Mask quality"),
             ("Pose accuracy: true poses for near misses", "Pose accuracy"),
             ("Coverage: true masks where uncovered", "Coverage"),
             ("Identity: drop masks of other objects", "Identity"),
             ("Selection: oracle among best poses", "Selection"),
             ("Count: true identities and counts", "Count"),
             ("All repairs together", "All together")]
    base = R["Baseline AR or AP"]
    ys = [7.3 - i for i in range(6)] + [0.0]
    fig, axes = plt.subplots(1, 4, figsize=(W2, 1.9), sharey=True,
                             gridspec_kw=dict(width_ratios=[1.25, 1.25, 1.25, 0.95], wspace=0.10))
    fig.subplots_adjust(left=0.105, right=0.99, top=0.80, bottom=0.17)
    for k, (ax, ttl) in enumerate(zip(axes[:3], ("LM-O", "T-LESS (300 images)", "IC-BIN"))):
        for (key, _), y in zip(order, ys):
            pc, jt = R[key][2 * k], R[key][2 * k + 1]
            if pc is None:
                continue
            ax.plot([pc, jt], [y, y], color=C["light"], lw=1.4, zorder=2)
            dot(ax, pc, y, C["fp"], ms=5.6)
            dot(ax, jt, y, C["joint"], ms=4.0, z=5)
            if key.startswith(("Selection", "All")) and pc > 20:
                if pc >= jt:
                    ax.text(pc + 2.5, y, "+%.1f" % pc, fontsize=6.3, color=C["fp"], ha="left", va="center")
                else:
                    ax.text(pc - 2.5, y, "+%.1f" % pc, fontsize=6.3, color=C["fp"], ha="right", va="center")
        ax.axvline(0, color="#9A9A9A", lw=0.6)
        ax.set_xlim(-8, 84); ax.set_xticks([0, 20, 40, 60, 80])
        ax.set_title("%s\nbaseline %.1f / %.1f AR" % (ttl, base[2 * k], base[2 * k + 1]), fontsize=6.9, loc="left")
    a = axes[0]
    a.text(R[order[4][0]][0], ys[4] + 0.75, "per-call ranking", fontsize=6.3, color=C["fp"], ha="center", va="center")
    a.text(R[order[1][0]][1] + 3, ys[1] + 0.62, "joint scoring", fontsize=6.3, color=C["joint"], ha="left", va="center")
    ax = axes[3]
    for (key, _), y in zip(order, ys):
        for j, (mk, hol) in enumerate((("D", False), ("s", True))):
            v = R[key][6 + j]
            if v is not None:
                ax.plot([v], [y + (0.16 if j == 0 else -0.16)], mk, ms=4.0, mfc="white" if hol else C["joint"],
                        mec=C["joint"], mew=0.9, zorder=3)
    ax.text(R[order[6][0]][6], ys[6] + 0.8, "IPD", fontsize=6.3, color=C["joint"], ha="center", va="center")
    ax.text(R[order[6][0]][7] - 3.0, ys[6] - 0.16, "XYZ-IBD", fontsize=6.3, color=C["joint"], ha="right", va="center")
    ax.axvline(0, color="#9A9A9A", lw=0.6)
    ax.set_xlim(-3, 48); ax.set_xticks([0, 20, 40])
    ax.set_title("Bins, joint only\nbaseline %.1f / %.1f AP" % (base[6], base[7]), fontsize=6.9, loc="left")
    for a in axes:
        a.axhline(0.65, color=C["light"], lw=0.5)
        a.tick_params(axis="y", length=0); a.spines["left"].set_visible(False)
    axes[0].set_yticks(ys); axes[0].set_yticklabels([o[1] for o in order])
    axes[0].get_yticklabels()[-1].set_fontweight("bold")
    fig.text(0.545, 0.02, "Gain from the repair (points)", ha="center", fontsize=7.2)
    save(fig, "fig_counterfactual.pdf")


# ================================================================== Figure 6: density
def fig_density():
    y2 = load("data33_17/Y2_RESULTS.json")["tasks"]
    out = load("data33_17/derived_v3.json")["density_A4_basis_posthoc"]
    fig, (a, c, e) = plt.subplots(1, 3, figsize=(W2, 1.8), gridspec_kw=dict(wspace=0.62, width_ratios=[1.12, 1.12, 0.9]))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.30)
    x = np.arange(5)

    def frames_row(ax, counts, ypos, label=None):
        tf = btf(ax.transData, ax.transAxes)
        for xi, n in zip(x, counts):
            if n is not None:
                ax.text(xi, ypos, str(n), transform=tf, ha="center", va="top", fontsize=5.9, color=C["muted"])
        if label:
            ax.text(-0.55, ypos, label, transform=tf, ha="right", va="top", fontsize=5.9, color=C["muted"])

    for task, col, mk, lab in (("localization", C["ink"], "o", "localization"), ("detection", C["base"], "s", "detection")):
        bins = y2[task]["bins"]
        v = np.array([100 * q["missing_share"]["estimate"] for q in bins])
        lo = np.array([100 * q["missing_share"]["ci95"][0] for q in bins]); hi = np.array([100 * q["missing_share"]["ci95"][1] for q in bins])
        n = [q["frames"] for q in bins]
        sel = np.array([k for k in range(5) if n[k] >= 10])
        a.errorbar(x[sel], v[sel], yerr=[v[sel] - lo[sel], hi[sel] - v[sel]], color=col, marker=mk, ms=3.3, capsize=0, lw=1.1, elinewidth=0.8)
        a.text(4.18, v[-1], lab, color=col, fontsize=6.3, va="center", ha="left")
    a.set_ylim(0, 62); a.set_yticks([0, 20, 40, 60]); a.set_ylabel("Instances without a\ncandidate mask (%)")
    frames_row(a, [q["frames"] for q in y2["localization"]["bins"]], -0.16, "loc.")
    frames_row(a, [q["frames"] if q["frames"] >= 10 else None for q in y2["detection"]["bins"]], -0.25, "det.")
    for key, ck, col, mk, lab in (("A1_mean", "A1_ci", C["fp"], "o", "per-call scores"), ("A4_mean", "A4_ci", C["joint"], "o", "joint scoring")):
        v = np.array(out[key]); ci = np.array(out[ck])
        c.errorbar(x, v, yerr=[v - ci[:, 0], ci[:, 1] - v], color=col, marker=mk, ms=3.3, capsize=0, lw=1.1, elinewidth=0.8)
        c.text(4.18, v[-1] + (1.2 if key == "A4_mean" else -1.2), lab, color=col, fontsize=6.3, va="center", ha="left")
    c.axhline(0, color="#9A9A9A", lw=0.6)
    c.set_ylim(-9, 19); c.set_ylabel("AR gain from the true masks\nof missed instances (points)")
    frames_row(c, out["frames"], -0.16, "frames")
    bins = y2["localization"]["bins"]
    v = np.array([q["object_gain"]["estimate"] for q in bins]); lo = np.array([q["object_gain"]["ci95"][0] for q in bins])
    hi = np.array([q["object_gain"]["ci95"][1] for q in bins])
    e.errorbar(x, v, yerr=[v - lo, hi - v], color=C["joint"], marker="o", ms=3.3, capsize=0, lw=1.1, elinewidth=0.8)
    e.set_ylim(0, 70); e.set_yticks([0, 20, 40, 60]); e.set_ylabel("AR gain of joint over\nper-call scores (points)")
    frames_row(e, [q["frames"] for q in bins], -0.16, "frames")
    for i, ax in enumerate((a, c, e)):
        ax.set_xticks(x); ax.set_xticklabels(BINS); ax.set_xlim(-0.35, 4.35 if i == 2 else 5.75)
        ax.spines["bottom"].set_bounds(-0.35, 4.35)
        ax.text(-0.02, 1.02, "(%s)" % "abc"[i], transform=ax.transAxes, fontsize=7.6, fontweight="bold", va="bottom", ha="right")
    fig.text(0.53, 0.03, "Largest number of same-object instances in the frame", ha="center", fontsize=7.2)
    save(fig, "fig_density.pdf")


# ================================================================== Figure 7: the check across scores
def fig_check():
    """Forest plot: one row per score, a stem from 50% (no information) to the cross-call pairwise accuracy;
    colour marks only the scores that fall below 50%."""
    blocks, cur, pairs = {}, None, {}
    for line in open(os.path.join(T, "tab_check.tex"), encoding="utf8"):
        m = re.search(r"Best poses of (FoundationPose|WAPR).*?([\d,]+) pairs", line)
        if m:
            pairs[m.group(1)] = m.group(2)
        if "Best poses of FoundationPose" in line: cur = "fp"
        elif "Best poses of WAPR" in line: cur = "wapr"
        elif cur and "&" in line and not line.lstrip().startswith("Score"):
            c = [x.strip() for x in line.strip().rstrip("\\").split("&")]
            m = re.search(r"([\d.]+) \[([\d.]+), ([\d.]+)\]", c[2])
            blocks.setdefault(cur, []).append(dict(name=c[0], ss=c[1], acc=float(m.group(1)), lo=float(m.group(2)), hi=float(m.group(3))))
    rws = [("head", "(a) Best poses of FoundationPose (%s pairs)" % pairs["FoundationPose"])] + [("row", r) for r in blocks["fp"]] + \
          [("head", "(b) Best poses of WAPR (%s pairs)" % pairs["WAPR"])] + [("row", r) for r in blocks["wapr"]]
    n = len(rws); h = 0.165 * n + 0.48
    fig, ax = plt.subplots(figsize=(W1, h))
    fig.subplots_adjust(left=0.355, right=0.765, top=0.91, bottom=0.17)
    tf = btf(fig.transFigure, ax.transData)
    ax.axvspan(0, 50, color=C["tint"], zorder=0, lw=0)
    ax.axvline(50, color=C["muted"], lw=0.7, ls=(0, (4, 3)), zorder=1)
    yt, yl = [], []
    for i, (kind, r) in enumerate(rws):
        y = n - 1 - i
        if kind == "head":
            ax.text(0.005, y, r, transform=tf, fontsize=6.7, fontweight="bold", ha="left", va="center")
            continue
        col = C["fp"] if r["acc"] < 50 else C["ink"]
        ax.plot([50, r["acc"]], [y, y], color=col if r["acc"] < 50 else "#B5B5B5", lw=1.2, alpha=0.6 if r["acc"] < 50 else 1, zorder=2)
        ax.plot([r["lo"], r["hi"]], [y, y], color=col, lw=2.4, solid_capstyle="butt", zorder=3)   # 95% interval
        dot(ax, r["acc"], y, col, ms=4.4)
        yt.append(y); yl.append(r["name"].replace("per-call", "per call"))
        ax.text(0.885, y, "%.1f" % r["acc"], transform=tf, fontsize=6.6, color=col, ha="right", va="center")
        ss = r["ss"].replace("no (0)", "0").replace("--", "–")
        ax.text(0.995, y, ss, transform=tf, fontsize=6.6, color=C["muted"] if ss == "–" else C["ink"], ha="right", va="center")
    ax.text(0.885, n - 0.45, "Acc.\n(%)", transform=tf, fontsize=6.1, color=C["muted"], ha="right", va="bottom", linespacing=0.95)
    ax.text(0.995, n - 0.45, "Set\nsens.", transform=tf, fontsize=6.1, color=C["muted"], ha="right", va="bottom", linespacing=0.95)
    ax.set_yticks(yt); ax.set_yticklabels(yl); ax.tick_params(axis="y", length=0, pad=3)
    for lab, (_, r) in zip(ax.get_yticklabels(), [x for x in rws if x[0] == "row"]):
        if r["acc"] < 50:
            lab.set_color(C["fp"])
    ax.text(25, -1.0, "inverted", fontsize=6.3, color=C["fp"], ha="center", va="center")
    ax.text(75, -1.0, "comparable", fontsize=6.3, color=C["muted"], ha="center", va="center")
    ax.set_xlim(0, 100); ax.set_ylim(-1.5, n - 0.45)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Cross-call pairwise accuracy (%)", labelpad=2)
    save(fig, "fig_check.pdf")


# ================================================================== Figure 9: building the configuration
def fig_ladder():
    names = {"CNOS only": "CNOS masks only", "Raw-score pooling": "+ NOCTIS, per-call scores",
             "+ Joint scoring": "+ joint scoring", "+ SAM-6D": "+ SAM-6D masks",
             "+ Low-confidence NOCTIS": "+ low-confidence NOCTIS", "NMS radius $0.15D$": "NMS radius 0.15D",
             "+ Identity demotion": "+ identity demotion", "+ Ray refinement (\\basedev{})": "+ ray refinement"}
    steps = []
    for c in rows(os.path.join(T, "tab_ladder.tex")):
        if c[0] in names:
            d = c[7].replace("$-$", "-")
            m = re.match(r"([+\-]?[\d.]+)", d)
            steps.append(dict(name=names[c[0]], ds=[float(x) for x in c[1:6]], mean=float(c[6]),
                              delta=None if not m else m.group(1)))
    n = len(steps); h = 0.215 * n + 0.62
    fig, ax = plt.subplots(figsize=(W1, h))
    fig.subplots_adjust(left=0.395, right=0.83, top=1 - 0.30 / h, bottom=0.36 / h)
    tf = btf(fig.transFigure, ax.transData)
    ys = [n - 1 - i for i in range(n)]
    ax.plot([s["mean"] for s in steps], ys, color=C["ink"], lw=0.9, zorder=3)
    for s, y in zip(steps, ys):
        ax.plot([min(s["ds"]), max(s["ds"])], [y, y], color=C["light"], lw=1.4, zorder=1, solid_capstyle="butt")
        for v in s["ds"]:
            ax.plot([v], [y], "|", ms=5.5, mew=0.9, color="#A0A0A0", zorder=2)
        col = C["fp"] if s["name"].startswith("+ NOCTIS, per") else C["joint"] if s["name"] == "+ joint scoring" else C["ink"]
        dot(ax, s["mean"], y, col, ms=4.8, z=4)
        if s["delta"] is not None:
            dv = s["delta"].replace("-", "−")
            ax.text(0.995, y, dv if dv[0] in "+−" else "+" + dv, transform=tf, fontsize=6.5, ha="right", va="center",
                    color=col if col != C["ink"] else C["ink"])
    ax.text(0.995, ys[0] + 0.65, "Δ mean", transform=tf, fontsize=6.1, color=C["muted"], ha="right", va="center")
    ax.text(steps[0]["mean"], ys[0] + 0.6, "mean", fontsize=6.1, color=C["ink"], ha="center", va="center")
    ax.text(max(steps[1]["ds"]) + 1.5, ys[1] - 0.05, "| one dataset", fontsize=6.1, color="#8A8A8A", ha="left", va="center")
    ax.set_yticks(ys); ax.set_yticklabels([s["name"] for s in steps]); ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_ylim(-0.6, n - 0.25)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("AR after each step (points)", labelpad=2)
    save(fig, "fig_ladder.pdf")


# ================================================================== Figure 10: accuracy and time
def fig_ar_time():
    """Mean AR of the seven datasets against seconds per image, read from tables/main_mean_time.csv (the Mean and
    s/img columns moved out of Table 7); every point carries its two values, so the figure replaces those columns."""
    import csv
    names = {"WAPR.v2~\\cite{wang2026wapr}": "WAPR.v2", "FRTPose-WAPR.v2$^{a}$~\\cite{wang2026wapr}": "FRTPose-WAPR.v2*",
             "FRTPose-WAPR$^{a}$": "FRTPose-WAPR*", "FreeZeV2.2~\\cite{caraffa2025accurate}": "FreeZeV2.2",
             "\\baseuniform{}": "Base-Uniform", "\\ours{} \\full{}": "Full", "\\ours{} \\fast{}": "Fast"}
    pts = []
    with open(os.path.join(T, "main_mean_time.csv"), encoding="utf8") as f:
        for r in csv.DictReader(line for line in f if not line.startswith("#")):
            pts.append(dict(name=names[r["row"]], ar=r["mean_ar"], t=r["s_per_img"], ours=r["row"].startswith("\\")))
    assert len(pts) == len(names)
    fig, ax = plt.subplots(figsize=(2.80, 1.5957))
    fig.subplots_adjust(left=0.205, right=0.975, top=0.99, bottom=0.215)
    # label placement: (dx, dy) in points, alignment
    off = {"WAPR.v2": (0, 5, "center", "bottom"), "FRTPose-WAPR.v2*": (0, 5, "center", "bottom"),
           "FRTPose-WAPR*": (0, -5, "center", "top"), "FreeZeV2.2": (8, 5, "center", "bottom"),
           "Base-Uniform": (5, -1, "left", "top"), "Full": (4, -3, "left", "top"), "Fast": (5, 0, "left", "center")}
    for p in pts:
        col = C["joint"] if p["ours"] else C["base"]
        x, y = float(p["t"]), float(p["ar"])
        dot(ax, x, y, col, ms=5.0 if p["ours"] else 4.4)
        dx, dy, ha, va = off[p["name"]]
        ax.annotate("%s\n%s, %s s" % (p["name"], p["ar"], p["t"]), (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va=va, fontsize=5.9, linespacing=1.05, color=col if p["ours"] else C["muted"])
    ax.set_xscale("log"); ax.set_xlim(14, 420); ax.set_xticks([20, 50, 100, 200]); ax.set_xticklabels(["20", "50", "100", "200"])
    ax.minorticks_off()
    ax.set_ylim(0.8285, 0.8508); ax.set_yticks([0.830, 0.835, 0.840, 0.845]); ax.set_yticklabels(["0.830", "0.835", "0.840", "0.845"])
    ax.yaxis.grid(True, color=C["grid"], lw=0.5); ax.set_axisbelow(True)
    ax.set_xlabel("Seconds per image (log scale)", labelpad=2); ax.set_ylabel("Mean AR", labelpad=2)
    save(fig, "fig_ar_time.pdf")


# ================================================================== Figure 11: public submission files
SHORT = [("(Default Detections)", "(Default)"), ("(Default Detection)", "(Default)"), ("(Multi 2D Detections)", "(Multi 2D)"),
         ("(SAM6D-FastSAM)", "(FastSAM)"), ("(SAM6D, Coarse-to-Fine)", ""), ("(CNOS, Coarse)", "")]
CONTROL = ("Co-op (CNOS, Coarse)", "FreeZeV2 (SAM6D, Coarse-to-Fine)")   # entries whose pages do not mention FoundationPose


def fig_public():
    """One row per entry: mean AP (AR) of its files with the submitted confidence, depth consistency, a random order and
    the reversed confidence, read from the supplementary table; entries without correct--incorrect pairs are left out."""
    rws = []
    for line in open(os.path.join(T, "supp_s23_public.tex"), encoding="utf8"):
        if "&" not in line or line.lstrip().startswith("Entry"): continue
        c = [x.strip() for x in line.strip().rstrip("\\").rstrip().rstrip("\\").split("&")]
        if c[3] == "const.": continue
        name = re.sub(r"\$\^\{[a-z]\}\$", "", c[0]).strip()
        rws.append(dict(name=name, metric=c[1], rng=c[3].replace("--", "–").replace("100.00", "100"),
                        sub=float(c[4]), dep=float(c[5]), rev=float(c[6]), rnd=float(c[7]), control=name in CONTROL))
    assert sum(r["control"] for r in rws) == len(CONTROL)

    def short(nm):
        for x, y in SHORT: nm = nm.replace(x, y)
        return nm.strip()
    ment = sorted([r for r in rws if not r["control"]], key=lambda r: -r["sub"])
    ctrl = sorted([r for r in rws if r["control"]], key=lambda r: -r["sub"])
    order = [("head", "Entries whose pages mention FoundationPose")] + [("row", r) for r in ment] + \
            [("head", "Control entries")] + [("row", r) for r in ctrl]
    n = len(order); h = 0.155 * n + 0.62
    fig, ax = plt.subplots(figsize=(W1, h))
    fig.subplots_adjust(left=0.405, right=0.83, top=1 - 0.36 / h, bottom=0.30 / h)
    tf = btf(fig.transFigure, ax.transData)
    yt, yl = [], []
    for i, (kind, r) in enumerate(order):
        y = n - 1 - i
        if kind == "head":
            ax.text(0.005, y, r, transform=tf, fontsize=6.6, fontweight="bold", ha="left", va="center"); continue
        vals = [r["sub"], r["dep"], r["rev"], r["rnd"]]
        ax.plot([min(vals), max(vals)], [y, y], color=C["light"], lw=1.2, zorder=1)
        dot(ax, r["rnd"], y, C["base"], ms=4.0, hollow=True, z=2)
        ax.plot([r["dep"]], [y], "D", ms=3.4, mfc=C["base"], mec="white", mew=0.4, zorder=3)
        dot(ax, r["rev"], y, C["fp"], ms=4.0, z=4)
        dot(ax, r["sub"], y, C["ink"], ms=4.0, z=5)
        yt.append(y); yl.append(short(r["name"]) + (" (AR)" if r["metric"] == "AR" else ""))
        ax.text(0.995, y, r["rng"], transform=tf, fontsize=6.0, ha="right", va="center")
    ax.text(0.995, n - 0.4, "Pairwise\naccuracy (%)", transform=tf, fontsize=5.9, color=C["muted"], ha="right", va="bottom", linespacing=0.95)
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=6.2); ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xlim(0, 90); ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_ylim(-0.6, n - 0.4)
    ax.spines["left"].set_visible(False); ax.xaxis.grid(True, color=C["grid"], lw=0.5); ax.set_axisbelow(True)
    ax.set_xlabel("AP or AR (%), mean over the files of an entry", labelpad=2)
    handles = [Line2D([], [], marker="o", ls="none", mfc=C["ink"], mec="white", ms=4.2, label="submitted"),
               Line2D([], [], marker="D", ls="none", mfc=C["base"], mec="white", ms=3.6, label="depth consistency"),
               Line2D([], [], marker="o", ls="none", mfc="white", mec=C["base"], mew=0.9, ms=4.0, label="random order"),
               Line2D([], [], marker="o", ls="none", mfc=C["fp"], mec="white", ms=4.2, label="reversed")]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0), handletextpad=0.2, columnspacing=0.9, fontsize=6.1)
    save(fig, "fig_public.pdf")


if __name__ == "__main__":
    import sys
    todo = {"phenomenon": fig_phenomenon, "intervention": fig_intervention, "scores": fig_scores,
            "counterfactual": fig_counterfactual, "density": fig_density, "check": fig_check, "ladder": fig_ladder,
            "ar_time": fig_ar_time, "public": fig_public}
    names = sys.argv[1:] or list(todo)
    for nm in names:
        todo[nm]()
    print("wrote " + ", ".join(names))
