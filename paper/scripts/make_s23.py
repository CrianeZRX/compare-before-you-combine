#!/usr/bin/env python3
"""Tables and derived numbers from the 33-23 deliverable (data33_23/).

Usage:  python scripts/make_s23.py          (run from the manuscript root)
Inputs: data33_23/S23_A.json, S23_B.json, S23_C.json (unchanged deliverables)
Outputs:
  tables/supp_s23_wbps.tex     WBPS and other scores as confidence and grasp order on WAPR poses
  tables/supp_s23_public.tex   public submissions: check and confidence replacement, per method
  tables/supp_s23_subset.tex   750-image sample against the full test sets
  data33_23/derived_s23.json   numbers quoted in the text

Notes:
  * Two localization files of FoundationPose(NVIDIA) carry one score for every pose. The
    deliverable labels them "reversed" because ties count as wrong orders under the strict
    accuracy; with ties counted half they are at 50%. They are reported here as constant
    scores, not as reversed ones.
  * AP is not a mean over images; its bootstrap distribution lies above the full-sample
    value. AP intervals are quoted only where they cover the point estimate.
"""
import json, os, hashlib, statistics as st
from collections import OrderedDict, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data33_23"); T = os.path.join(ROOT, "tables")
inputs = {}


def load(name):
    p = os.path.join(D, name)
    inputs["data33_23/" + name] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return json.load(open(p))


A, B, C = load("S23_A.json"), load("S23_B.json"), load("S23_C.json")
out = OrderedDict()
f2 = lambda v: "%.2f" % v


def iv(x):
    """point and interval of a BCa record; interval None if it misses the point."""
    return dict(point=x["point_pp"], ci=x["ci95_pp"] if x["point_in_ci"] else None,
                offset=x["bootstrap_min_median_max_pp"][1] - x["point_pp"])


# ---------------- B: WBPS and other scores on WAPR poses ----------------
ROWS = [("B-C0", "Detector confidence"), ("B-W", "WBPS, per call"), ("B-Wa", "WBPS, each pose alone"),
        ("B-Wneg", "WBPS, reversed"), ("B-F", "FoundationPose, each pose alone"),
        ("B-Fj", "FoundationPose, joint"), ("B-D", "Depth consistency"), ("B-C0D", "Detector $\\times$ depth")]
b = OrderedDict()
for pool in ("W0", "W1"):
    x = B["by_dataset"]["classic_mean"][pool]
    b[pool] = OrderedDict(
        ap={r: iv(x["ap"]["rows"][r]["image"]) for r, _ in ROWS},
        top1={r: iv(x["ranking"]["top1"]["rows"][r]["image"]) if "rows" in x["ranking"]["top1"]
              else iv(x["ranking"]["top1"][r]["image"]) for r, _ in ROWS},
        p3={r: iv(x["ranking"]["precision_at_3"]["rows"][r]["image"]) if "rows" in x["ranking"]["precision_at_3"]
            else iv(x["ranking"]["precision_at_3"][r]["image"]) for r, _ in ROWS},
        random_top1=x["random_top1"]["image"]["point_pp"],
        d_ap={k: iv(v["image"]) for k, v in x["ap"]["differences"].items()},
        reading=x["reading"])
    # top-1 differences are not stored as differences; take the reading-level record if present
out["B"] = b
# top-1 difference WBPS - detector from the report (image BCa), stored in readings of S23_B
for pool in ("W0", "W1"):
    x = B["by_dataset"]["classic_mean"][pool]["ranking"]["top1"]
    if "differences" in x:
        out["B"][pool]["d_top1"] = {k: iv(v["image"]) for k, v in x["differences"].items()}

lines = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
         r" & \multicolumn{3}{c}{CNOS masks (W0)} & \multicolumn{2}{c}{CNOS and NOCTIS masks (W1)} \\",
         r"\cmidrule(lr){2-4}\cmidrule(l){5-6}",
         r"Confidence & AP & First correct (\%) & Precision at 3 (\%) & AP & First correct (\%) \\", r"\midrule"]
for r, name in ROWS:
    w0, w1 = b["W0"], b["W1"]
    lines.append("%s & %s & %s & %s & %s & %s \\\\" % (
        name, f2(w0["ap"][r]["point"]), f2(w0["top1"][r]["point"]), f2(w0["p3"][r]["point"]),
        f2(w1["ap"][r]["point"]), f2(w1["top1"][r]["point"])))
lines += [r"\midrule", "Random order (expected) & -- & %s & -- & -- & %s \\\\" % (
    f2(b["W0"]["random_top1"]), f2(b["W1"]["random_top1"])), r"\bottomrule", r"\end{tabular}"]
open(os.path.join(T, "supp_s23_wbps.tex"), "w").write("\n".join(lines) + "\n")

# ---------------- A: public submissions ----------------
cases = []
for c in A["cases"]:
    a1, a2, s = c["A1"], c.get("A2"), c["submission"]
    acc = a1["accuracy_ties_half"]["image"]["point_pp"] if a1["accuracy_ties_half"]["image"] else None
    rec = dict(method=s["method_name"], dataset=s["dataset"], task=s["task"], group=s["group"],
               constant=a1["all_scores_equal"], distinct=a1["distinct_scores"], pairs=a1["pair_counts"][2],
               acc=None if a1["all_scores_equal"] else acc, oversupplied=a1["oversupplied_fraction"],
               eligible=bool(a2))
    if a2:
        R, Dd = a2["official"]["rows"], a2["official"]["differences"]
        rec.update(S=R["S"]["image"]["point_pp"], SD=R["S-D"]["image"]["point_pp"],
                   Sneg=R["S-neg"]["image"]["point_pp"], Srand=R["S-rand"]["image"]["point_pp"],
                   dD=iv(Dd["S-D-S"]["image"]), dN=iv(Dd["S-neg-S"]["image"]), R6=a2["R6"], R7=a2["R7"])
    cases.append(rec)
nonconst = [c for c in cases if c["acc"] is not None]
elig = [c for c in cases if c["eligible"]]
from collections import Counter
pub = OrderedDict(
    files=len(cases), methods=len({c["method"] for c in cases}),
    fp_files=sum(c["group"] == "FoundationPose_mentioned" for c in cases),
    control_files=sum(c["group"] != "FoundationPose_mentioned" for c in cases),
    constant_files=sum(c["constant"] for c in cases),
    no_effect_files=sum(not c["eligible"] for c in cases),
    with_pairs=len(nonconst), acc_min=min(c["acc"] for c in nonconst), acc_max=max(c["acc"] for c in nonconst),
    reversed_nonconstant=sum(c["acc"] < 50 for c in nonconst),
    eligible=len(elig),
    R6=dict(Counter(c["R6"] for c in elig)), R7=dict(Counter(c["R7"] for c in elig)),
    neg_raises=sum(c["dN"]["point"] > 0 for c in elig),
)
try:
    from scipy.stats import spearmanr
    ne = [c for c in elig if c["acc"] is not None]
    r = spearmanr([c["acc"] for c in ne], [c["dN"]["point"] for c in ne])
    pub["spearman_acc_vs_reversal_cost"] = float(r.statistic)
    pub["spearman_n"] = len(ne)
except Exception:
    pass
up = [c for c in elig if c["R6"] == "提高"]
pub["depth_raises"] = [dict(method=c["method"], dataset=c["dataset"], dD=c["dD"], acc=c["acc"]) for c in up]
out["A"] = pub
out["A_cases"] = cases

bym = OrderedDict()
for c in cases:
    bym.setdefault(c["method"], []).append(c)
lines = [r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
         r"Entry & Metric & Files & Pairwise (\%) & Submitted & Depth cons. & Reversed & Random \\", r"\midrule"]
mean = lambda v: st.mean(v) if v else None
for m, cs in bym.items():
    e = [c for c in cs if c["eligible"]]
    acc = [c["acc"] for c in cs if c["acc"] is not None]
    metric = "AR" if cs[0]["task"] == "localization" else "AP"
    if acc:
        a = f2(min(acc)) if len(acc) == 1 or abs(min(acc) - max(acc)) < 0.005 else "%s--%s" % (f2(min(acc)), f2(max(acc)))
    else:
        a = "const."
    vals = [mean([c[k] for c in e]) for k in ("S", "SD", "Sneg", "Srand")]
    name = m.replace("（", " (").replace("）", ")").replace("&", r"\&")
    if not e:
        lines.append("%s & %s & %d & %s & \\multicolumn{4}{c}{score cannot change %s} \\\\" % (name, metric, len(cs), a, metric))
    else:
        note = "" if len(e) == len(cs) else "$^{a}$"
        lines.append("%s%s & %s & %d & %s & %s & %s & %s & %s \\\\" % (
            name, note, metric, len(cs), a, *[f2(v) for v in vals]))
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(T, "supp_s23_public.tex"), "w").write("\n".join(lines) + "\n")

# ---------------- C: statistics ----------------
c3 = C["C3"]["by_dataset"]
sub = OrderedDict()
for ds, v in c3.items():
    sub[ds] = {r: dict(full=x["full_AR_pp"], subset=x["subset_AR_pp"], diff=x["subset_minus_full_pp"])
               for r, x in v["rows"].items()}
dsl = list(sub)
for r in ("A0", "A1", "A4"):
    full = st.mean(sub[d][r]["full"] for d in dsl); s = st.mean(sub[d][r]["subset"] for d in dsl)
    sub.setdefault("mean", {})[r] = dict(full=full, subset=s, diff=s - full)
out["C3"] = sub
NAMES = dict(lmo="LM-O", tless="T-LESS", tudl="TUD-L", icbin="IC-BIN", ycbv="YCB-V", mean="Mean")
lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{CNOS only} & \multicolumn{2}{c}{Per-call, pooled} & \multicolumn{2}{c}{Joint, pooled} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}",
         r"Dataset & Full & Sample & Full & Sample & Full & Sample \\", r"\midrule"]
for ds in ("lmo", "tless", "tudl", "icbin", "ycbv", "mean"):
    if ds == "mean":
        lines.append(r"\midrule")
    lines.append("%s & %s \\\\" % (NAMES[ds], " & ".join("%s & %s" % (f2(sub[ds][r]["full"]), f2(sub[ds][r]["subset"]))
                                                       for r in ("A0", "A1", "A4"))))
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(T, "supp_s23_subset.tex"), "w").write("\n".join(lines) + "\n")

e1 = C["C1_33_22_E1"]["by_dataset"]
out["C1"] = OrderedDict(
    classic_C1_C0=iv(e1["classic_mean"]["P1"]["differences"]["C1-C0"]["BCa"]["image"]),
    xyz_C1_C0=iv(e1["xyzibd"]["P1"]["differences"]["C1-C0"]["BCa"]["image"]),
    classic_C0_vs_C7=iv(e1["classic_mean"]["P1"]["versus_best"]["C0"]["BCa"]["image"]),
    E4_W1_W0=iv(C["C1_33_22_E4"]["by_dataset"]["classic_mean"]["differences"]["W1-W0"]["BCa"]["image"]),
    outside=len(C["BCa_point_estimates_outside_interval"]),
    outside_beyond_support=sum("outside the finite" in x["cause"] for x in C["BCa_point_estimates_outside_interval"]),
    reading_changes=len(C["reading_changes"]),
    xyz_P1_P2_same_poses=all(v.get("pose_and_score_multisets_identical", False) for v in C["C2"]["cases"].values()
                             if isinstance(v, dict) and "counts" in v),
    xyz_P1_P2_poses=C["C2"]["cases"]["C1"]["counts"][0],
)


def maxoffset(obj):
    best = 0.0
    def walk(x, path=""):
        nonlocal best
        if isinstance(x, dict):
            if "bootstrap_min_median_max_pp" in x and x.get("point_pp") is not None and "/rows/" in path and "/image" in path:
                best = max(best, abs(x["bootstrap_min_median_max_pp"][1] - x["point_pp"]))
            for k, v in x.items():
                walk(v, path + "/" + k)
    walk(obj)
    return best


out["C1"]["ap_offset_max_E1"] = maxoffset(e1)
out["C1"]["ap_offset_max_B"] = maxoffset(B["by_dataset"])
out["inputs"] = inputs
json.dump(out, open(os.path.join(D, "derived_s23.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: out[k] for k in ("A", "C1")}, indent=1, ensure_ascii=False)[:3000])
print("W0 AP", {k: round(v["point"], 2) for k, v in b["W0"]["ap"].items()})
print("W0 top1", {k: round(v["point"], 2) for k, v in b["W0"]["top1"].items()})
print("W1 AP", {k: round(v["point"], 2) for k, v in b["W1"]["ap"].items()})
print("dAP W0", {k: (round(v["point"], 2), v["ci"]) for k, v in b["W0"]["d_ap"].items()})
print("dAP W1", {k: (round(v["point"], 2), v["ci"]) for k, v in b["W1"]["d_ap"].items()})
print("mean subset", sub["mean"])
