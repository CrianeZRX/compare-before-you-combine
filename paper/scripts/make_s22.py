#!/usr/bin/env python3
"""Figures, tables and derived numbers from the 33-22 deliverable (data33_22/).

Usage:  python scripts/make_s22.py          (run from the manuscript root)
Inputs: data33_22/S22_E1.json, S22_E2.json, S22_E3.json, S22_E4.json (unchanged)
        data33_17/ED_MECH.json, data33_17/TABLES_P1.md, data33_19/S19_E1.json (for Table 2 and Table 3)
Outputs:
  (figures/fig_intervention.pdf is drawn by scripts/make_figs_v4.py)
  figures/fig2/uses_data.tex        numbers drawn in Figure 2
  tables/tab_uses.tex               Table: four uses of a pose score, FoundationPose poses
  tables/tab_check.tex              Table: the check across scores, two pose sets
  tables/supp_s22_e3.tex            Supplementary table: intervention by dataset
  data33_22/derived_s22.json        numbers quoted in the text
"""
import json, os, hashlib, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data33_22"); F = os.path.join(ROOT, "figures"); T = os.path.join(ROOT, "tables")
os.makedirs(os.path.join(F, "fig2"), exist_ok=True); os.makedirs(T, exist_ok=True)
W = float(os.environ.get("FIGW", 7.1))
C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00", grey="#7F7F7F", black="#222222")
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.6, "lines.linewidth": 1.3,
    "pdf.fonttype": 42, "font.family": "DejaVu Sans", "legend.frameon": False,
})
DS = ["lmo", "tless", "tudl", "icbin", "ycbv"]
NAME = dict(lmo="LM-O", tless="T-LESS", tudl="TUD-L", icbin="IC-BIN", ycbv="YCB-V")
inputs = {}


def load(path):
    p = os.path.join(ROOT, path)
    inputs[path] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return json.load(open(p))


e1, e2, e3, e4 = (load("data33_22/S22_%s.json" % k) for k in ("E1", "E2", "E3", "E4"))
ed = load("data33_17/ED_MECH.json")
s19 = load("data33_19/S19_E1.json")


def md_table(fname, title):
    rows, on = [], False
    for line in open(os.path.join(ROOT, fname), encoding="utf8"):
        if line.startswith("## "):
            on = line[3:].strip() == title
            continue
        if on and line.startswith("|") and not line.startswith("| ---"):
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
    return rows[1:]


ar_p1 = {r[0]: float(r[-1]) for r in md_table("data33_17/TABLES_P1.md", "E-A 与 X2 的 AR（百分点）")}


def ap(ds, pool, row):
    r = e1["results"][ds][pool]["AP"].get(row)
    return None if not r or not r.get("ap") else r["ap"]["point_pp"]


def top1(ds, pool, row):
    m = e2["results"][ds][pool]["metrics"].get(row)
    return None if not m or not m.get("top1") else m["top1"]["point_pp"]


# ------------------------------------------------------------------ intervention (E3)
LAD = [("T_xy", "Lateral shift ($D_o$)"), ("R", "Rotation (degrees)"), ("T_z", "Shift along the ray ($D_o$)")]


def ladder_stats():
    out = {}
    for lad, _ in LAD:
        out[lad] = {}
        for scorer in ("fp_single", "megapose_single"):
            up, diff = {}, {}
            for ds in DS:
                for inst in e3["per_instance"][ds]:
                    z = [x for x in inst["ladders"][lad] if x["value"] == 0]
                    if not z or z[0].get(scorer) is None:
                        continue
                    z0 = z[0][scorer]
                    for x in inst["ladders"][lad]:
                        if x["value"] == 0 or x["out_of_frame"] or x.get(scorer) is None:
                            continue
                        up.setdefault(x["value"], []).append(x[scorer] > z0)
                        diff.setdefault(x["value"], []).append(x[scorer] - z0)
            out[lad][scorer] = {str(v): {"share_higher_pct": 100 * sum(up[v]) / len(up[v]), "median_change": st.median(diff[v]), "n": len(up[v])}
                                for v in sorted(up)}
    return out


lad = ladder_stats()
# post hoc: in the joint call, is the top rung of each rotation ladder a correct pose?
r10r_fail = r10r_fail_correct = 0
for ds in DS:
    for inst in e3["per_instance"][ds]:
        R = [x for x in inst["ladders"]["R"] if not x["out_of_frame"] and x.get("fp_joint") is not None]
        if not R:
            continue
        top = max(R, key=lambda x: x["fp_joint"])
        if top["value"] != 0:
            r10r_fail += 1
            r10r_fail_correct += top["mssd_over_diameter"] < 0.1

fig, axes = plt.subplots(1, 3, figsize=(W, 1.85), gridspec_kw=dict(wspace=0.32))
for i, (ax, (key, xlabel)) in enumerate(zip(axes, LAD)):
    for scorer, lab, col, mk in (("fp_single", "FoundationPose", C["red"], "o"), ("megapose_single", "MegaPose", C["blue"], "s")):
        d = lad[key][scorer]
        xs = sorted(float(v) for v in d)
        ys = [d[str(v) if str(v) in d else str(int(v))]["share_higher_pct"] for v in xs]
        if key == "T_z":
            neg = [(x, y) for x, y in zip(xs, ys) if x < 0]; pos = [(x, y) for x, y in zip(xs, ys) if x > 0]
            for part in (neg, pos):
                ax.plot([p[0] for p in part], [p[1] for p in part], marker=mk, ms=3, color=col, label=lab if part is pos else None)
        else:
            ax.plot(xs, ys, marker=mk, ms=3, color=col, label=lab)
    if key == "T_xy":
        ax.set_xscale("log"); ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1, 2, 4]); ax.set_xticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2", "4"])
    elif key == "R":
        ax.set_xscale("log"); ax.set_xticks([5, 10, 20, 45, 90, 180]); ax.set_xticklabels(["5", "10", "20", "45", "90", "180"])
    else:
        ax.set_xticks([-0.5, -0.2, 0, 0.2, 0.5]); ax.axvline(0, color=C["grey"], lw=0.5, ls=":")
    ax.minorticks_off()
    ax.axhline(50, color=C["grey"], lw=0.6, ls=(0, (3, 2)))
    ax.set_ylim(-3, 103); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_xlabel(xlabel)
    if i == 0:
        ax.set_ylabel("Scored above the true\npose alone (% of instances)")
    ax.text(-0.02, 1.04, "(%s)" % "abc"[i], transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")
axes[0].legend(loc="center right", bbox_to_anchor=(1.0, 0.62), handlelength=1.6)
plt.close(fig)  # the article figure is drawn by scripts/make_figs_v4.py; this draft is not saved

# ------------------------------------------------------------------ Table 2: four uses, FoundationPose poses
cm = "classic_mean"
uses = {
    "AR": {"percall": ar_p1["A1"], "joint": ar_p1["A4"], "neg": None, "megapose": ar_p1["A5"], "detdepth": None, "ref": ar_p1["A0"]},
    "AP": {"percall": ap(cm, "P1", "C1"), "joint": ap(cm, "P1", "C4"), "neg": ap(cm, "P1", "C2"), "megapose": ap(cm, "P1", "C5"), "detdepth": ap(cm, "P1", "C7"), "ref": ap(cm, "P1", "C0")},
    "top1": {"percall": top1(cm, "P1", "C1"), "joint": top1(cm, "P1", "C4"), "neg": top1(cm, "P1", "C2"), "megapose": top1(cm, "P1", "C5"), "detdepth": top1(cm, "P1", "C7"), "ref": top1(cm, "P1", "C0"),
             "random": e2["results"][cm]["P1"]["random_top1"]["point_pp"]},
}
s20 = load("data33_20/derived_s20.json")
uses["AR"]["neg"] = round(s20["AR_full_mean"]["A1neg"], 2)
uses["AR"]["detdepth"] = ar_p1["A10"]
uses["AR"]["softmax"] = ar_p1["A8"]


def f(v, nd=2):
    return "--" if v is None else ("%.*f" % (nd, v))


bins = {"xyzibd": ("XYZ-IBD", "P1"), "ipd": ("IPD", "P2")}
lines = ["% generated by scripts/make_s22.py from data33_17, data33_20 and data33_22 -- do not edit by hand\n",
         "\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{@{}l l rrrrr r@{}}\n\\toprule\n",
         " & & \\multicolumn{3}{c}{FoundationPose} & \\multicolumn{2}{c}{Comparable scores} & \\\\\n\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\n",
         "Use & Measure & Per-call & Joint & Reversed & MegaPose & Det.\\,$\\times$\\,depth & Reference \\\\\n\\midrule\n",
         "\\multicolumn{8}{@{}l}{\\emph{Five classic datasets}}\\\\\n",
         "Choose among masks of two detectors & AR & %s & %s & %s & %s & %s & %s$^a$ \\\\\n" % tuple(f(uses["AR"][k]) for k in ("percall", "joint", "neg", "megapose", "detdepth", "ref")),
         "Output confidence & AP & %s & %s & %s & %s & \\textbf{%s} & %s$^b$ \\\\\n" % tuple(f(uses["AP"][k]) for k in ("percall", "joint", "neg", "megapose", "detdepth", "ref")),
         "Which pose first & Top-1 (\\%%) & %s & %s & \\textbf{%s} & %s & %s & %s$^b$ / %s$^c$ \\\\\n" % tuple(f(uses["top1"][k]) for k in ("percall", "joint", "neg", "megapose", "detdepth", "ref", "random")),
         "\\midrule\n\\multicolumn{8}{@{}l}{\\emph{Bin picking}}\\\\\n"]
binrows = {}
for ds, (nm, pool) in bins.items():
    binrows[ds] = {"pool": pool, "AP": {r: ap(ds, pool, r) for r in ("C0", "C1", "C2", "C4", "C6", "C7")},
                   "top1": {r: top1(ds, pool, r) for r in ("C0", "C1", "C2", "C4", "C6", "C7")},
                   "random": e2["results"][ds][pool]["random_top1"]["point_pp"]}
    b = binrows[ds]
    dd = b["AP"]["C7"] if b["AP"]["C7"] is not None else None
    lines.append("%s, output confidence & AP & %s & %s & %s & -- & %s & %s \\\\\n" % (nm, f(b["AP"]["C1"]), f(b["AP"]["C4"]), f(b["AP"]["C2"]), f(dd), f(b["AP"]["C0"]) + "$^b$" if b["AP"]["C0"] is not None else "--"))
    lines.append("%s, which pose first & Top-1 (\\%%) & %s & %s & %s & -- & %s & %s \\\\\n" % (nm, f(b["top1"]["C1"]), f(b["top1"]["C4"]), f(b["top1"]["C2"]), f(b["top1"]["C7"]),
                 (f(b["top1"]["C0"]) + "$^b$ / " if b["top1"]["C0"] is not None else "") + f(b["random"]) + "$^c$"))
lines.append("\\bottomrule\n\\end{tabular}\n")
open(os.path.join(T, "tab_uses.tex"), "w").write("".join(lines))

# ------------------------------------------------------------------ Table 3: the check across scores
E19 = load("data33_19/derived_s19.json")["E1"]
s19rows = {}
for key, name in (("FPraw", "FoundationPose, per-call"), ("FPsingle", "FoundationPose, alone"), ("FPjoint", "FoundationPose, joint"), ("MPsingle", "MegaPose, alone"),
                  ("MPjoint", "MegaPose, joint"), ("Depth", "Depth consistency"), ("MaskIoU", "Rendered-mask IoU"), ("Chamfer", "Truncated chamfer")):
    r = E19[key]
    s19rows[name] = ("%.2f [%.2f, %.2f]" % (r["pooled"], r["ci"][0], r["ci"][1]), "%.2f" % r["ar"])
setsens = {"FoundationPose": "%.3f" % E19["set_sensitivity"]["FoundationPose"]["p90"], "MegaPose": "%.3f" % E19["set_sensitivity"]["MegaPose"]["p90"]}


def pw(key):
    v = e4["M3"]["pooled"][key]["ties_half"]
    return "%.2f [%.2f, %.2f]" % (v["point_pp"], *v["ci95_image_pp"])


def arw(row):
    return f(e4["M4"]["classic_mean"]["metrics"][row]["ar"]["point_pp"])


m2 = e4["M2"]["all"]
lines = ["% generated by scripts/make_s22.py from data33_19 and data33_22 -- do not edit by hand\n",
         "\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{@{}l c l r@{}}\n\\toprule\n",
         "Score & Set sensitivity (P90) & Cross-call pairwise accuracy [95\\% CI] & AR \\\\\n\\midrule\n",
         "\\multicolumn{4}{@{}l}{\\emph{Best poses of FoundationPose (CNOS and NOCTIS masks; 22,555 pairs)}}\\\\\n"]
for name, label, ss in (("FoundationPose, per-call", "FoundationPose, per-call", "%.2f" % float(setsens["FoundationPose"])),
                        ("FoundationPose, alone", "FoundationPose, alone", "--"),
                        ("FoundationPose, joint", "FoundationPose, joint", "--"),
                        ("MegaPose, alone", "MegaPose, alone", "%.2f" % float(setsens["MegaPose"])),
                        ("MegaPose, joint", "MegaPose, batch", "--"),
                        ("Depth consistency", "Depth consistency", "0.00"),
                        ("Rendered-mask IoU", "Rendered-mask IoU", "0.00"),
                        ("Truncated chamfer", "Truncated chamfer", "0.00")):
    acc, arv = s19rows[name]
    lines.append("%s & %s & %s & %s \\\\\n" % (label, ss, ("\\textbf{%s}" % acc) if "per-call" in name or "alone" in name and "Foundation" in name else acc, arv))
lines += ["\\midrule\n\\multicolumn{4}{@{}l}{\\emph{Best poses of WAPR (same masks and images; 22,578 pairs)}}\\\\\n",
          "WBPS, per-call & %.2f & %s & %s \\\\\n" % (m2["P90"], pw("wbps_raw"), arw("W1")),
          "WBPS, alone & -- & %s & %s \\\\\n" % (pw("wbps_single"), arw("W1a")),
          "FoundationPose, alone & -- & \\textbf{%s} & -- \\\\\n" % pw("fp_single"),
          "FoundationPose, joint & -- & %s & %s \\\\\n" % (pw("fp_joint"), arw("WF")),
          "Depth consistency & no (0) & %s & %s \\\\\n" % (pw("depth"), arw("WD")),
          "\\bottomrule\n\\end{tabular}\n"]
open(os.path.join(T, "tab_check.tex"), "w").write("".join(lines))

# ------------------------------------------------------------------ supplementary table: intervention by dataset
cs = e3["case_summary"]
lines = ["% generated by scripts/make_s22.py from data33_22/S22_E3.json -- do not edit by hand\n",
         "\\begin{tabular}{@{}l rrrrr@{}}\n\\toprule\n",
         "Dataset & R9 & R10, lateral & R10, rotation & R11 & R12 \\\\\n\\midrule\n"]
for ds in DS + ["pooled"]:
    c = cs[ds]
    lines.append("%s & %s \\\\\n" % (NAME.get(ds, "Pooled"), " & ".join("%d/%d" % (c[k]["positive"], c[k]["eligible"]) for k in ("R9", "R10_T_xy", "R10_R", "R11", "R12"))))
lines.append("\\bottomrule\n\\end{tabular}\n")
open(os.path.join(T, "supp_s22_e3.tex"), "w").write("".join(lines))

# ------------------------------------------------------------------ numbers drawn in Figure 2
with open(os.path.join(F, "fig2", "uses_data.tex"), "w") as h:
    h.write("%% generated by scripts/make_s22.py -- do not edit by hand\n")
    for k, v in (("APdet", uses["AP"]["ref"]), ("APfp", uses["AP"]["percall"]), ("APjoint", uses["AP"]["joint"]), ("APbest", uses["AP"]["detdepth"]),
                 ("Within", 100 * sum(ed["within_call_by_dataset"][d]["correct_first"] for d in DS) / sum(ed["within_call_by_dataset"][d]["opposite_label_pairs"] for d in DS)),
                 ("Cross", 100 * ed["cross_mask_combined"]["raw"]["accuracy"]),
                 ("TOPdet", uses["top1"]["ref"]), ("TOPfp", uses["top1"]["percall"]), ("TOPrand", uses["top1"]["random"]), ("TOPbest", uses["top1"]["detdepth"])):
        h.write("\\def\\fUse%s{%.1f}\n" % (k, v))

# ------------------------------------------------------------------ derived numbers
pc = e1["results"][cm]["P1"]["comparisons_AP_pp"]
derived = {
    "inputs": inputs,
    "E1_classic_P1_AP": {r: ap(cm, "P1", r) for r in ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7")},
    "E1_classic_P1_C1_minus_C0": pc["C1-C0"]["point_pp"],
    "E2_classic_P1_top1": {r: top1(cm, "P1", r) for r in ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7")},
    "E2_classic_P1_random": uses["top1"]["random"],
    "bins": binrows,
    "E3": {"instances": sum(len(e3["per_instance"][d]) for d in DS), "scored_poses": e3["scored_poses"], "out_of_frame": e3["out_of_frame"],
           "readings": {k: {"positive": cs["pooled"][k]["positive"], "eligible": cs["pooled"][k]["eligible"], "pct": cs["pooled"][k]["stats"]["point_pp"]} for k in ("R9", "R10_T_xy", "R10_R", "R11", "R12")},
           "posthoc_R10_R_failures": r10r_fail, "posthoc_R10_R_failures_with_correct_top": r10r_fail_correct,
           "ladders": lad},
    "E4": {"M2_P90": m2["P90"], "M2_median": m2["median"], "masks_valid": e4["mask_coverage"]["valid"], "masks_requested": e4["mask_coverage"]["requested"],
           "within_group_pct": e4["M1"]["pooled"]["within_group"]["ties_half"]["point_pp"],
           "pairwise": {k: {"pct": e4["M3"]["pooled"][k]["ties_half"]["point_pp"], "ci": e4["M3"]["pooled"][k]["ties_half"]["ci95_image_pp"], "pairs": e4["M3"]["pooled"][k]["pairs"]}
                        for k in ("wbps_raw", "wbps_neg", "wbps_single", "wbps_joint", "fp_single", "fp_joint", "depth")},
           "AR": {r: e4["M4"]["classic_mean"]["metrics"][r]["ar"]["point_pp"] for r in ("W0", "W1", "W1neg", "W1a", "W4", "WF", "WD")},
           "top_changed": sum(e4["M2"]["by_dataset"][d]["top_mask_changed_targets"] for d in DS),
           "targets": sum(e4["M2"]["by_dataset"][d]["total_targets"] for d in DS),
           "readings": e4["reading"]},
    "uses": uses,
}
json.dump(derived, open(os.path.join(D, "derived_s22.json"), "w"), indent=1, sort_keys=True)
print("wrote fig_intervention.pdf, fig2/uses_data.tex, tab_uses.tex, tab_check.tex, supp_s22_e3.tex, derived_s22.json")
print(json.dumps({k: derived[k] for k in ("E1_classic_P1_AP", "E2_classic_P1_top1")}, indent=0)[:800])
print("posthoc", r10r_fail, r10r_fail_correct, "top changed", derived["E4"]["top_changed"], derived["E4"]["targets"])
