#!/usr/bin/env python3
"""Figures of the analysis-first manuscript, drawn only from the 33-17 deliverable JSON files.

Usage: python3 scripts/make_figures_v3.py            (run from the manuscript root)
Inputs : data33_17/*.json  (extracted from P1_deliver.tgz and P2_deliver.tgz, unchanged)
Outputs: figures/fig_phenomenon.pdf, fig_density.pdf, fig_counts.pdf, fig_slots.pdf
         data33_17/derived_v3.json  (numbers computed here and quoted in the text)
"""
import json, os, hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data33_17")
F = os.path.join(ROOT, "figures")
W = float(os.environ.get("FIGW", 5.45))  # text width in inches: 5.45 for the review format, 7.1 for the 5p two-column format

# Okabe-Ito colours, distinguishable in grey scale by lightness and by marker/hatch
C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00",
         purple="#CC79A7", sky="#56B4E9", grey="#7F7F7F", black="#222222")
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.6, "lines.linewidth": 1.3,
    "pdf.fonttype": 42, "font.family": "DejaVu Sans", "legend.frameon": False,
})
DS = [("lmo", "LM-O"), ("tless", "T-LESS"), ("tudl", "TUD-L"), ("icbin", "IC-BIN"), ("ycbv", "YCB-V")]
BINS = ["1", "2–3", "4–7", "8–15", "≥16"]
derived = {"inputs": {}}


def load(name):
    p = os.path.join(D, name)
    derived["inputs"][name] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return json.load(open(p))


def md_table(fname, title):
    """Rows of one table of an auto-generated TABLES_*.md file."""
    rows, on = [], False
    for line in open(os.path.join(D, fname), encoding="utf8"):
        if line.startswith("## "):
            on = line[3:].strip() == title
            continue
        if on and line.startswith("|") and not line.startswith("| ---"):
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
    return rows[1:]


def panel(ax, s):
    ax.text(-0.02, 1.06, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")


# ---------------------------------------------------------------- Figure 1
def fig_phenomenon():
    ed = load("ED_MECH.json")
    ar = {r[0]: [float(x) for x in r[1:]] for r in md_table("TABLES_P1.md", "E-A 与 X2 的 AR（百分点）")}
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.25), gridspec_kw=dict(width_ratios=[1.9, 1], wspace=0.78))
    keys = [k for k, _ in DS] + ["pooled"]
    labels = [n for _, n in DS] + ["Pooled"]
    series = [("raw", "FoundationPose, per-call scores", C["red"], "//"),
              ("joint", "FoundationPose, joint scoring", C["blue"], ""),
              ("megapose", "MegaPose", C["green"], "..")]
    x = np.arange(len(keys)); w = 0.26
    for i, (k, lab, col, hatch) in enumerate(series):
        v = [100 * (ed["cross_mask_by_dataset"][d][k]["accuracy"] if d != "pooled" else ed["cross_mask_combined"][k]["accuracy"]) for d in keys]
        a.bar(x + (i - 1) * w, v, w, color=col, label=lab, hatch=hatch, edgecolor="white", linewidth=0.3)
        if k == "raw":
            for xi, vi in zip(x, v):
                a.text(xi - w, vi + 1.5, f"{vi:.1f}", ha="center", va="bottom", fontsize=6.3, color=col)
    within = [100 * ed["within_call_by_dataset"][d]["accuracy"] for d, _ in DS]
    a.plot(x[:-1], within, ls="none", marker="_", ms=13, mew=1.4, color=C["black"], label="FoundationPose, within one call")
    a.axhline(50, color=C["grey"], lw=0.7, ls=(0, (3, 2)))
    a.set_xlim(-0.55, len(keys) - 0.45)
    a.text(len(keys) - 0.4, 50, "chance", fontsize=6.3, color=C["grey"], ha="left", va="center", clip_on=False)
    a.set_xticks(x); a.set_xticklabels(labels, fontsize=6.6); a.set_ylim(0, 108); a.set_yticks([0, 25, 50, 75, 100])
    a.set_ylabel("Correct pose ranked first (%)")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, 1.36), ncol=2, columnspacing=1.0, handlelength=1.4)
    panel(a, "(a)")
    rows = [("CNOS masks only", ar["A0"][-1], C["grey"]),
            ("+ NOCTIS masks,\nper-call scores", ar["A1"][-1], C["red"]),
            ("+ NOCTIS masks,\njoint scoring", ar["A4"][-1], C["blue"]),
            ("+ NOCTIS masks,\nMegaPose scores", ar["A5"][-1], C["green"])]
    y = np.arange(len(rows))[::-1]
    b.barh(y, [r[1] for r in rows], 0.62, color=[r[2] for r in rows])
    for yi, r in zip(y, rows):
        b.text(r[1] + 1.5, yi, f"{r[1]:.1f}", va="center", fontsize=7)
    b.set_yticks(y); b.set_yticklabels([r[0] for r in rows]); b.set_xlim(0, 100)
    b.set_xlabel("AR on five datasets (points)")
    panel(b, "(b)")
    fig.savefig(os.path.join(F, "fig_phenomenon.pdf"), bbox_inches="tight", pad_inches=0.02); plt.close(fig)
    derived["pairwise_pooled"] = {k: ed["cross_mask_combined"][k] for k in ("raw", "joint", "detection", "c2", "megapose")}
    derived["within_call"] = {d: ed["within_call_by_dataset"][d]["accuracy"] for d, _ in DS}


# ---------------------------------------------------------------- Figure 4 (density)
def density_bin(n):
    return 0 if n <= 1 else 1 if n <= 3 else 2 if n <= 7 else 3 if n <= 15 else 4


def spearman(x, y):
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def fig_density():
    y2 = load("Y2_RESULTS.json")["tasks"]
    x1 = load("X1_PER_FRAME.json")["datasets"]
    # A4-basis coverage repair per frame (post hoc; the pre-registered reading Y2-b uses the A1 basis)
    recs = []
    for ds in ("lmo", "tless", "icbin"):
        for r in x1[ds]["records"]:
            recs.append((r["density"], r["bases"]["A1"]["delta_pp"]["Fcov"], r["bases"]["A4"]["delta_pp"]["Fcov"],
                         r["bases"]["A4"]["failure_counts"]["eligible"], ds))
    dens = np.array([r[0] for r in recs]); b = np.array([density_bin(n) for n in dens])
    g1 = np.array([r[1] for r in recs]); g4 = np.array([r[2] for r in recs]); elig = np.array([r[3] for r in recs])
    rng = np.random.default_rng(20261026)
    idx = rng.integers(0, len(recs), size=(10000, len(recs)))
    out = {"bins": BINS, "frames": [], "A1_mean": [], "A4_mean": [], "A4_ci": [], "A1_ci": [], "A4_instance_weighted": []}
    for k in range(5):
        m = b == k
        out["frames"].append(int(m.sum())); out["A1_mean"].append(float(g1[m].mean())); out["A4_mean"].append(float(g4[m].mean()))
        out["A4_instance_weighted"].append(float((g4[m] * elig[m]).sum() / elig[m].sum()))
        for key, g in (("A4_ci", g4), ("A1_ci", g1)):
            boot = []
            for row in idx[:, :]:
                mm = b[row] == k
                boot.append(g[row][mm].mean() if mm.any() else np.nan)
            out[key].append([float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))])
    rho = spearman(dens, g4)
    rb = [spearman(dens[row], g4[row]) for row in idx[:2000]]
    out["A4_spearman"] = {"rho": rho, "ci95": [float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))], "draws": 2000}
    out["A1_spearman_check"] = spearman(dens, g1)
    out["by_dataset_A4_mean"] = {ds: float(np.mean([r[2] for r in recs if r[4] == ds])) for ds in ("lmo", "tless", "icbin")}
    derived["density_A4_basis_posthoc"] = out

    fig, (a, c, e) = plt.subplots(1, 3, figsize=(W, 1.9), gridspec_kw=dict(wspace=0.62))
    x = np.arange(5)
    for task, col, mk, lab in (("localization", C["blue"], "o", "Localization"), ("detection", C["orange"], "s", "Detection")):
        bins = y2[task]["bins"]
        v = np.array([100 * q["missing_share"]["estimate"] for q in bins])
        lo = np.array([100 * q["missing_share"]["ci95"][0] for q in bins]); hi = np.array([100 * q["missing_share"]["ci95"][1] for q in bins])
        n = [q["frames"] for q in bins]
        sel = np.array([k for k in range(5) if n[k] >= 10])
        a.errorbar(x[sel], v[sel], yerr=[v[sel] - lo[sel], hi[sel] - v[sel]], color=col, marker=mk, ms=3.5, capsize=2, lw=1.2, label=lab)
    a.set_xticks(x); a.set_xticklabels(BINS, fontsize=6.4); a.set_ylabel("Instances without\na candidate mask (%)"); a.set_ylim(0, 62)
    a.legend(loc="upper left", handlelength=1.6); panel(a, "(a)")
    for key, ck, col, mk, lab in (("A1_mean", "A1_ci", C["red"], "v", "per-call scores"), ("A4_mean", "A4_ci", C["blue"], "o", "joint scoring")):
        v = np.array(out[key]); ci = np.array(out[ck])
        c.errorbar(x, v, yerr=[v - ci[:, 0], ci[:, 1] - v], color=col, marker=mk, ms=3.5, capsize=2, lw=1.2, label=lab)
    c.axhline(0, color=C["grey"], lw=0.6)
    c.set_xticks(x); c.set_xticklabels(BINS, fontsize=6.4); c.set_ylabel("AR gain from true masks\nof missed instances (points)")
    c.set_xlabel("Largest number of same-object instances in the frame"); c.legend(loc="upper right", handlelength=1.6); c.set_ylim(-9, 19); panel(c, "(b)")
    bins = y2["localization"]["bins"]
    v = np.array([q["object_gain"]["estimate"] for q in bins]); lo = np.array([q["object_gain"]["ci95"][0] for q in bins]); hi = np.array([q["object_gain"]["ci95"][1] for q in bins])
    e.errorbar(x, v, yerr=[v - lo, hi - v], color=C["black"], marker="D", ms=3.2, capsize=2, lw=1.2)
    e.set_xticks(x); e.set_xticklabels(BINS, fontsize=6.4); e.set_ylabel("AR gain of joint over\nper-call scores (points)"); e.set_ylim(0, 70)
    panel(e, "(c)")
    fig.savefig(os.path.join(F, "fig_density.pdf"), bbox_inches="tight", pad_inches=0.02); plt.close(fig)


# ---------------------------------------------------------------- Figure 5 (counts)
def fig_counts():
    y7 = {"xyzibd": load("Y7_XYZIBD.json"), "ipd": load("Y7_IPD.json")}
    names = {"xyzibd": "XYZ-IBD", "ipd": "IPD"}; cols = {"xyzibd": C["blue"], "ipd": C["orange"]}
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.15), gridspec_kw=dict(width_ratios=[1, 1.25], wspace=0.36))
    edges = np.logspace(np.log10(0.08), np.log10(16), 25)
    summ = {}
    for k in ("xyzibd", "ipd"):
        r = np.array([f["ratios"]["detector_all"] for f in y7[k]["per_frame"]])
        truth = sum(f["truth_count"] for f in y7[k]["per_frame"])
        a.hist(np.clip(r, edges[0], edges[-1]), bins=edges, histtype="stepfilled", alpha=0.35, color=cols[k], weights=np.full(len(r), 100 / len(r)))
        a.hist(np.clip(r, edges[0], edges[-1]), bins=edges, histtype="step", color=cols[k], lw=1.1, weights=np.full(len(r), 100 / len(r)), label=f"{names[k]} (median {np.median(r):.2f})")
        summ[k] = {"frames": len(r), "truth_instances": int(truth), "median_ratio": float(np.median(r)), "share_below_1": float((r < 1).mean())}
    a.axvline(1, color=C["black"], lw=0.8, ls=(0, (3, 2)))
    a.set_xscale("log"); a.set_xticks([0.08, 0.25, 1, 4, 16]); a.set_xticklabels(["≤0.08", "0.25", "1", "4", "≥16"]); a.minorticks_off()
    a.set_xlabel("Detections per annotated instance, per frame"); a.set_ylabel("Frames (%)"); a.set_ylim(0, 30)
    a.text(0.8, 29.5, "under-reports", ha="right", va="top", fontsize=6.5, color=C["grey"]); a.text(1.25, 29.5, "over-reports", ha="left", va="top", fontsize=6.5, color=C["grey"])
    a.legend(loc="upper center", bbox_to_anchor=(0.5, 1.3), ncol=1); panel(a, "(a)")
    cls = [("TP", "correct", C["green"], ""), ("identity", "wrong object", C["purple"], "xx"), ("pose", "wrong pose", C["sky"], ".."),
           ("duplicate", "duplicate", C["grey"], ""), ("background", "background", C["red"], "//")]
    pos, ticks = [], []
    x0 = 0
    for k in ("xyzibd", "ipd"):
        truth = summ[k]["truth_instances"]
        for j, row in enumerate(("B0", "B1", "B3", "B4")):
            bottom = 0
            for name, lab, col, hatch in cls:
                n = y7[k]["rows"][row]["classification"][name]["n"] / truth
                b.bar(x0 + j, n, 0.72, bottom=bottom, color=col, hatch=hatch, edgecolor="white", linewidth=0.3, label=lab if (k == "xyzibd" and j == 0) else None)
                bottom += n
            pos.append(x0 + j); ticks.append(row)
            summ[k][row] = {c0: y7[k]["rows"][row]["classification"][c0]["n"] for c0, *_ in cls}
            summ[k][row]["outputs"] = y7[k]["rows"][row]["outputs"]
        b.text(x0 + 1.5, -0.95, names[k], ha="center", va="top", fontsize=8)
        x0 += 5
    b.axhline(1, color=C["black"], lw=0.8, ls=(0, (3, 2)))
    b.set_xticks(pos); b.set_xticklabels(ticks); b.set_ylabel("Outputs per annotated instance"); b.set_ylim(0, 5.8)
    b.legend(loc="upper left", ncol=1, handlelength=1.2, labelspacing=0.25); panel(b, "(b)")
    fig.savefig(os.path.join(F, "fig_counts.pdf"), bbox_inches="tight", pad_inches=0.02); plt.close(fig)
    for k in ("xyzibd", "ipd"):
        b4 = y7[k]["rows"]["B4"]["by_source"]
        summ[k]["B4_by_source"] = {s: {c0: b4[s][c0]["n"] for c0, *_ in cls} for s in ("2D", "S", "G")}
    derived["counts"] = summ


# ---------------------------------------------------------------- Figure 6 (slots)
def fig_slots():
    y3 = load("Y3_RESULTS.json")["datasets"]
    spec = [("icbin", "IC-BIN (localization)", "AR (points)", "ar"), ("xyzibd", "XYZ-IBD (detection)", "AP (points)", "ap"), ("ipd", "IPD (detection)", "AP (points)", "ap")]
    src = [("Low", "lower-ranked 2D", C["grey"], "o"), ("S", "SAM2 masks", C["orange"], "s"), ("G", "depth–normal segments", C["green"], "^"), ("Mix", "2D + segments", C["purple"], "D")]
    fig, axes = plt.subplots(1, 3, figsize=(W, 1.85), gridspec_kw=dict(wspace=0.48))
    eff = {}
    for ax, (ds, title, yl, key), tag in zip(axes, spec, "abc"):
        rows = y3[ds]["rows"]; base = rows["Y3Base"]
        eff[ds] = {}
        for s, lab, col, mk in src:
            xs = [0.0] + [rows[f"Y3{s}{k}"]["mean_new_registrations"] for k in (1, 2, 4, 8)]
            ys = [base["estimate_pp"][key]] + [rows[f"Y3{s}{k}"]["estimate_pp"][key] for k in (1, 2, 4, 8)]
            ax.plot(xs, ys, color=col, marker=mk, ms=3.2, label=lab)
            r8 = rows[f"Y3{s}8"]
            eff[ds][s] = {"gain_k8": r8["estimate_pp"][key] - base["estimate_pp"][key],
                          "recovered_per_100_registrations": 100 * (r8["correct"] - base["correct"]) / r8["new_registrations"] if r8.get("new_registrations") else None,
                          "recall_k8": r8["recall"], "recall_base": base["recall"]}
        ax.set_title(f"({tag}) {title}", loc="left", fontsize=8); ax.set_ylabel(yl)
        if tag == "b": ax.set_xlabel("Added registrations per frame")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.9, 1.45), ncol=4, columnspacing=1.2, handlelength=1.6)
    fig.savefig(os.path.join(F, "fig_slots.pdf"), bbox_inches="tight", pad_inches=0.02); plt.close(fig)
    derived["slots"] = eff


# ---------------------------------------------------------------- Figure 3 (score distributions) and supplementary joint-score figure
def fig_scores():
    h = load("ED_HIST.json")["scopes"]
    for key, fname, xlab in (("s_raw", "fig_scores.pdf", "Per-call FoundationPose score"), ("s_joint", "fig_scores_joint.pdf", "Joint FoundationPose score")):
        fig, axes = plt.subplots(1, 5, figsize=(W, 1.55), sharey=False, gridspec_kw=dict(wspace=0.32))
        for ax, (d, name) in zip(axes, DS):
            e = np.array(h[d][key]["bin_edges"]); mid = 0.5 * (e[1:] + e[:-1]); wdt = e[1] - e[0]
            for cls, col, lab in (("correct", C["blue"], "correct pose"), ("incorrect", C["red"], "incorrect pose")):
                c = np.array(h[d][key][cls]["counts"], float); c = c / (c.sum() * wdt)
                ax.fill_between(mid, c, step="mid", color=col, alpha=0.25, lw=0)
                ax.step(mid, c, where="mid", color=col, lw=1.0, label=lab)
                ax.axvline(h[d][key][cls]["median"], color=col, lw=0.8, ls=(0, (3, 2)))
            ax.set_title(name); ax.set_yticks([]); ax.spines["left"].set_visible(False)
            ax.tick_params(axis="x", labelsize=6.5)
        axes[2].set_xlabel(xlab)
        axes[0].legend(loc="upper center", bbox_to_anchor=(2.9, 1.42), ncol=2, handlelength=1.6)
        fig.savefig(os.path.join(F, fname), bbox_inches="tight", pad_inches=0.02); plt.close(fig)
    # combined figure of the article: (a) per-call and (b) joint scores, one row each
    fig, axes = plt.subplots(2, 5, figsize=(W, 2.75), gridspec_kw=dict(wspace=0.22, hspace=0.95))
    fig.subplots_adjust(left=0.035, right=0.995, top=0.86, bottom=0.12)
    for r, (key, xlab, tag) in enumerate((("s_raw", "Per-call FoundationPose score", "(a)"), ("s_joint", "Joint FoundationPose score", "(b)"))):
        for ax, (d, name) in zip(axes[r], DS):
            e = np.array(h[d][key]["bin_edges"]); mid = 0.5 * (e[1:] + e[:-1]); wdt = e[1] - e[0]
            for cls, col, lab in (("correct", C["blue"], "correct pose"), ("incorrect", C["red"], "incorrect pose")):
                c = np.array(h[d][key][cls]["counts"], float); c = c / (c.sum() * wdt)
                ax.fill_between(mid, c, step="mid", color=col, alpha=0.25, lw=0)
                ax.step(mid, c, where="mid", color=col, lw=1.0, label=lab)
                ax.axvline(h[d][key][cls]["median"], color=col, lw=0.8, ls=(0, (3, 2)))
            ax.set_title(name, fontsize=7.5, pad=2); ax.set_yticks([]); ax.spines["left"].set_visible(False)
            ax.tick_params(axis="x", labelsize=6.5)
        axes[r][2].set_xlabel(xlab, labelpad=1)
        axes[r][0].text(-0.16, 1.08, tag, transform=axes[r][0].transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")
    axes[0][0].legend(loc="lower center", bbox_to_anchor=(0.5, 0.955), bbox_transform=fig.transFigure, ncol=2, handlelength=1.6)
    fig.savefig(os.path.join(F, "fig_scores_both.pdf"), bbox_inches="tight", pad_inches=0.02); plt.close(fig)
    derived["score_medians"] = {d: {k: {c: h[d][k][c]["median"] for c in ("correct", "incorrect")} for k in ("s_raw", "s_joint", "megapose_logit")} for d in h}
    derived["setsens"] = {sc: json.load(open(os.path.join(D, "SETSENS.json")))["scopes"]["pooled"][sc] for sc in ("FP", "MegaPose")}
    load("SETSENS.json")


if __name__ == "__main__":
    os.makedirs(F, exist_ok=True)
    fig_phenomenon(); fig_scores(); fig_density(); fig_counts(); fig_slots()
    json.dump(derived, open(os.path.join(D, "derived_v3.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in derived.items() if k != "inputs"}, indent=1, ensure_ascii=False)[:6000])
