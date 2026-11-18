#!/usr/bin/env python3
"""Recompute the numbers quoted in the running text from the deliverable tables and check that each appears in the text.
Usage: python3 scripts/check_numbers_v3.py [MANUSCRIPT_ROOT]
Without a manuscript root that contains sec/*.tex, the script prints the recomputed strings without checking them."""
import json, os, re, glob, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_tables_v3 import md_table, D
SRC = sorted(glob.glob(os.path.join(sys.argv[1] if len(sys.argv) > 1 else ".", "sec", "*.tex")))
text = " ".join(open(f).read() for f in SRC)
text = text.replace("$-$", "-").replace("$", "").replace("{,}", ",")
ar = {r[0]: [float(x) for x in r[1:]] for r in md_table("TABLES_P1.md", "E-A 与 X2 的 AR（百分点）")}
ed = json.load(open(os.path.join(D, "ED_MECH.json"))); dv = json.load(open(os.path.join(D, "derived_v3.json")))
x = lambda t: {r[0]: r for r in md_table("TABLES_P1.md", t)}
c = dv["counts"]; xb1, xb4, ib1, ib4 = c["xyzibd"]["B1"], c["xyzibd"]["B4"], c["ipd"]["B1"], c["ipd"]["B4"]
fp = lambda d: d["outputs"] - d["TP"]
checks = [
 ("CNOS only mean", f"{ar['A0'][-1]:.2f}"), ("per-call mean", f"{ar['A1'][-1]:.2f}"), ("joint mean", f"{ar['A4'][-1]:.2f}"),
 ("joint - CNOS", f"{ar['A4'][-1]-ar['A0'][-1]:.2f}"), ("TUD-L per-call", f"{ar['A1'][2]:.2f}"), ("LM-O per-call", f"{ar['A1'][0]:.2f}"),
 ("z-score", f"{ar['A7'][-1]:.2f}"), ("softmax", f"{ar['A8'][-1]:.2f}"), ("margin", f"{ar['A9'][-1]:.2f}"), ("MegaPose", f"{ar['A5'][-1]:.2f}"),
 ("pairwise raw", f"{100*ed['cross_mask_combined']['raw']['accuracy']:.2f}\\%"), ("pairwise joint", f"{100*ed['cross_mask_combined']['joint']['accuracy']:.2f}\\%"),
 ("pairwise megapose", f"{100*ed['cross_mask_combined']['megapose']['accuracy']:.2f}\\%"), ("pairs", f"{ed['cross_mask_combined']['raw']['opposite_label_pairs']:,}"),
 ("setsens FP", "0.03 units, but the 90th percentile is %.2f and the maximum %.2f" % (dv["setsens"]["FP"]["normalized_abs_delta"]["p90"], dv["setsens"]["FP"]["normalized_abs_delta"]["max"])),
 ("setsens MP", "%.3f, %.2f and %.2f" % (dv["setsens"]["MegaPose"]["normalized_abs_delta"]["median"], dv["setsens"]["MegaPose"]["normalized_abs_delta"]["p90"], dv["setsens"]["MegaPose"]["normalized_abs_delta"]["max"])),
 ("targets >=2", f"{dv['setsens']['FP']['targets_at_least_two']:,} targets"), ("top changes FP", f"{dv['setsens']['FP']['winner_changed']:,} of these"),
 ("correct top FP", f"{dv['setsens']['FP']['solo_correct_top']:,} to {dv['setsens']['FP']['joint_correct_top']:,}"),
 ("correct top MP", f"{dv['setsens']['MegaPose']['solo_correct_top']:,} targets either way" if dv['setsens']['MegaPose']['solo_correct_top']==dv['setsens']['MegaPose']['joint_correct_top'] else "MISMATCH"),
 ("raw medians", "%.1f for correct and %.1f for incorrect" % (dv["score_medians"]["pooled"]["s_raw"]["correct"], dv["score_medians"]["pooled"]["s_raw"]["incorrect"])),
 ("Fcov ratio LM-O", "35\\%" if round(100*1.89/5.42) == 35 and round(100*4.38/12.37) == 35 else "MISMATCH"),
 ("XYZ single repairs sum", f"{0.19+0.88+4.54+0.0+0.70+0.0:.2f}"),
 ("IPD B4 outputs", f"{ib4['outputs']:,}"), ("IPD background", f"{ib4['background']:,}"),
 ("IPD background share of outputs", f"{round(100*ib4['background']/ib4['outputs'])}\\% of all outputs"),
 ("IPD background share of false", f"{round(100*ib4['background']/fp(ib4))}\\% of the false"),
 ("IPD added TP / FP", f"{ib4['TP']-ib1['TP']} correct poses and {fp(ib4)-fp(ib1):,} false"), ("IPD ratio", f"about {round((fp(ib4)-fp(ib1))/(ib4['TP']-ib1['TP']))} false"),
 ("XYZ added TP / FP", f"{xb4['TP']-xb1['TP']} correct poses and {fp(xb4)-fp(xb1):,} false"), ("XYZ ratio", f"about {round((fp(xb4)-fp(xb1))/(xb4['TP']-xb1['TP']))} for each"),
 ("XYZ below one", f"{round(100*c['xyzibd']['share_below_1'])}\\% of the XYZ-IBD frames"), ("IPD below one", f"{round(100*c['ipd']['share_below_1'])}\\% of the IPD frames"),
 ("B0 IPD background", f"{round(100*c['ipd']['B0']['background']/c['ipd']['B0']['outputs'])}\\% of the detector"),
 ("density A4 rho", f"{dv['density_A4_basis_posthoc']['A4_spearman']['rho']:.2f} [{dv['density_A4_basis_posthoc']['A4_spearman']['ci95'][0]:.2f}, {dv['density_A4_basis_posthoc']['A4_spearman']['ci95'][1]:.2f}]"),
 ("T-LESS mean A4 gain", f"{dv['density_A4_basis_posthoc']['by_dataset_A4_mean']['tless']:.2f}"),
 ("A4-A5 by dataset", (lambda r: "T-LESS (%s) and LM-O (%s)" % (r[2], r[1]))(next(r for r in md_table("TABLES_P1.md", "E-A 的配对差（百分点）") if r[0] == "A4-A5"))),
 ("Y8 K16", "%.2f\\%%, %.2f\\%% and %.2f\\%% on all %s pairs" % tuple([float(r[i].split()[0]) for r in md_table("TABLES_Y8.md", "K=16 跨调用成对准确率（%）") if r[0] == "五集合并" for i in (2, 3, 4)] + [f"{36852:,}"])),
 ("Y8 K252", "%.2f\\%%, %.2f\\%% and %.2f\\%%" % tuple(float(r[3].split()[0]) for r in md_table("TABLES_Y8.md", "同一批胜者对的 K=16 与 K=252 准确率（%）") if r[0] == "五集合并")),
 ("Y8 FP", "%.2f\\%% for the z-score and %.2f\\%% for the margin" % tuple(float(r[i].split()[0]) for r in md_table("TABLES_Y8.md", "K=16 跨调用成对准确率（%）") if r[0] == "五集合并" for i in (5, 7))),
 ("Y8 AR", "%.2f, %.2f and %.2f" % tuple(float(r[2].split()[0]) for r in md_table("TABLES_Y8.md", "五集 AR 与分项（pp）") if r[0] == "五集等权" and r[1] in ("A12", "A13", "A14"))),
 ("minimal B1->B4", f"+{50.44-41.90:.1f}"), ("instances per frame", f"{3990/150:.1f}"), ("IPD instances per frame", f"{548/88:.1f}"),
] + (lambda m: [
 ("S18 coverage", "from %d\\%% to %d--%d\\%%" % (round(100*m['V0']['identity_correct_cover']), round(100*min(m[k]['identity_correct_cover'] for k in m if k != 'V0')), round(100*max(m[k]['identity_correct_cover'] for k in m if k != 'V0')))),
 ("S18 false", "from %.1f to %d--%d" % (m['V0']['false_detections_per_frame_mean'], round(min(m[k]['false_detections_per_frame_mean'] for k in m if k != 'V0')), round(max(m[k]['false_detections_per_frame_mean'] for k in m if k != 'V0')))),
])(json.load(open(os.path.join(os.path.dirname(__file__), '..', 'data33_18', 'S18_2D_V11.json')))['cohorts']['Cprime']) + [
 ("EC doubling", "about 9" if round(float(x('EC 配对差')['ECD-ECB'][1].split()[0])) == 9 and round(float(x('EC 配对差')['ECE-ECB'][1].split()[0])) == 9 else "MISMATCH"),
]
if not SRC:
    for name, s in checks:
        print("----", name, "->", s)
    sys.exit(0)
bad = 0
for name, s in checks:
    ok = s in text
    bad += not ok
    print(("OK  " if ok else "MISS"), name, "->", s)
print("missing:", bad)
