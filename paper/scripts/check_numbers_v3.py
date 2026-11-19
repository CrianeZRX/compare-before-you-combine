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
 ("IPD added TP / FP", f"{ib4['TP']-ib1['TP']} correct poses and {fp(ib4)-fp(ib1):,} false"), ("IPD ratio", f"{(fp(ib4)-fp(ib1))/(ib4['TP']-ib1['TP']):.1f} false"),
 ("XYZ added TP / FP", f"{xb4['TP']-xb1['TP']} correct poses and {fp(xb4)-fp(xb1):,} false"), ("XYZ ratio", f"{(fp(xb4)-fp(xb1))/(xb4['TP']-xb1['TP']):.1f} for each"),
 ("XYZ below one", f"{round(100*c['xyzibd']['share_below_1'])}\\% of the XYZ-IBD frames"), ("IPD below one", f"{round(100*c['ipd']['share_below_1'])}\\% of the IPD frames"),
 ("B0 IPD background", f"{round(100*c['ipd']['B0']['background']/c['ipd']['B0']['outputs'])}\\% of the detector"),
 ("density A4 rho", f"{dv['density_A4_basis_posthoc']['A4_spearman']['rho']:.2f} [{dv['density_A4_basis_posthoc']['A4_spearman']['ci95'][0]:.2f}, {dv['density_A4_basis_posthoc']['A4_spearman']['ci95'][1]:.2f}]"),
 ("T-LESS mean A4 gain", f"{dv['density_A4_basis_posthoc']['by_dataset_A4_mean']['tless']:.2f}"),
 ("A4-A5 by dataset", (lambda r: "T-LESS (%s) and LM-O (%s)" % (r[2], r[1]))(next(r for r in md_table("TABLES_P1.md", "E-A 的配对差（百分点）") if r[0] == "A4-A5"))),
 ("Y8 K16", "%.2f\\%%, %.2f\\%% and %.2f\\%% of all %s pairs" % tuple([float(r[i].split()[0]) for r in md_table("TABLES_Y8.md", "K=16 跨调用成对准确率（%）") if r[0] == "五集合并" for i in (2, 3, 4)] + [f"{36852:,}"])),
 ("Y8 FP", "inverted (%.2f\\%% and %.2f\\%%)" % tuple(float(r[i].split()[0]) for r in md_table("TABLES_Y8.md", "K=16 跨调用成对准确率（%）") if r[0] == "五集合并" for i in (5, 7))),
 ("minimal B1->B4", f"+{50.44-41.90:.1f}"), ("instances per frame", f"{3990/150:.1f}"), ("IPD instances per frame", f"{548/88:.1f}"),
] + (lambda m: [
 ("S18 coverage", "from %d\\%% to %d--%d\\%%" % (round(100*m['V0']['identity_correct_cover']), round(100*min(m[k]['identity_correct_cover'] for k in m if k != 'V0')), round(100*max(m[k]['identity_correct_cover'] for k in m if k != 'V0')))),
 ("S18 false", "from %.1f to %d--%d" % (m['V0']['false_detections_per_frame_mean'], round(min(m[k]['false_detections_per_frame_mean'] for k in m if k != 'V0')), round(max(m[k]['false_detections_per_frame_mean'] for k in m if k != 'V0')))),
])(json.load(open(os.path.join(os.path.dirname(__file__), '..', 'data33_18', 'S18_2D_V11.json')))['cohorts']['Cprime']) + [
 ("EC doubling", "about 9" if round(float(x('EC 配对差')['ECD-ECB'][1].split()[0])) == 9 and round(float(x('EC 配对差')['ECE-ECB'][1].split()[0])) == 9 else "MISMATCH"),
 ("slots k C-B", "segments lead by %.2f points" % (float(x('EC 评测')['ECC'][1]) - float(x('EC 评测')['ECB'][1]))),
 ("count given candidates", "adds %.1f AP points" % float({r[0]: r for r in md_table('TABLES_P2.md', 'Y1 配对差')}['B4-B3'][1].split()[0])),
 ("joint over det x depth", "and %.1f over the best ranking" % (ar['A4'][-1] - ar['A10'][-1]) if 'A10' in ar else "MISMATCH"),
]
# ---- 33-19 (data33_19/derived_s19.json, written by scripts/make_s19.py)
s19 = json.load(open(os.path.join(os.path.dirname(__file__), '..', 'data33_19', 'derived_s19.json')))
e1, e2c, e3 = s19["E1"], {(c["ds"], round(c["tau"], 3)): c for c in s19["E2"]["conditions"]}, s19["E3"]
pct = lambda v: f"{v:.2f}\\%"
neg = lambda c: f"{-c['delta']:.2f} [{-c['ci'][1]:.2f}, {-c['ci'][0]:.2f}]"
checks += [
 ("S19 bins", f"{e1['bins']['xyzibd']['B1']['cross']:.2f}\\% of the cross-call pairs correctly on XYZ-IBD and {e1['bins']['ipd']['B1']['cross']:.2f}\\% on IPD, against {e1['bins']['xyzibd']['B1']['within']:.2f}\\% and {e1['bins']['ipd']['B1']['within']:.2f}\\%"),
 ("S19 sample pairs", f"{e1['pairs']:,} cross-call pairs"),
 ("S19 alone", f"{e1['FPsingle']['pooled']:.2f}\\% [{e1['FPsingle']['ci'][0]:.2f}, {e1['FPsingle']['ci'][1]:.2f}]"),
 ("S19 raw same pairs", f"({e1['FPraw']['pooled']:.2f}\\%; {e1['FPsingle']['wins']:,} against {e1['FPraw']['wins']:,} correct orders)"),
 ("S19 AR alone/raw", f"({e1['FPsingle']['ar']:.2f} against {e1['FPraw']['ar']:.2f})"),
 ("S19 MegaPose", f"{e1['MPsingle']['pooled']:.2f}\\% of the cross-call pairs correctly when it scores each pose alone and {e1['MPjoint']['pooled']:.2f}\\%"),
 ("S19 fit range", f"{round(min(e1[k]['pooled'] for k in ('Depth','MaskIoU','Chamfer')))}--{round(max(e1[k]['pooled'] for k in ('Depth','MaskIoU','Chamfer')))}\\% of the pairs"),
 ("S19 GigaPose", f"{e1['native']['gigapose']['pooled']:.2f}\\% (GigaPose, {e1['native']['gigapose']['pairs']:,} pairs)"),
 ("S19 SAM-6D", f"{e1['native']['sam6d']['pooled']:.2f}\\% (SAM-6D, {e1['native']['sam6d']['pairs']:,} pairs)"),
 ("S19 IPD ratios", "from %.2f to %.2f, %.2f and %.2f" % tuple(e2c[('ipd', t)]['ratio'] for t in (0.2, 0.26, 0.29, 0.32))),
 ("S19 IPD loss official", neg(e2c[('ipd', 0.2)]) + " points at %.2f" % e2c[('ipd', 0.2)]['ratio']),
 ("S19 IPD losses", "to %.2f and %.2f" % (-e2c[('ipd', 0.26)]['delta'], -e2c[('ipd', 0.29)]['delta'])),
 ("S19 IPD low", "+%.2f [-%.2f, %.2f] at %.2f" % (e2c[('ipd', 0.32)]['delta'], -e2c[('ipd', 0.32)]['ci'][0], e2c[('ipd', 0.32)]['ci'][1], e2c[('ipd', 0.32)]['ratio'])),
 ("S19 IC-BIN", neg(e2c[('icbin', 0.2)]) + " points on IC-BIN (ratio %.2f)" % e2c[('icbin', 0.2)]['ratio']),
 ("S19 TUD-L", neg(e2c[('tudl', 0.2)]) + " on TUD-L (%.1f)" % e2c[('tudl', 0.2)]['ratio']),
 ("S19 XYZ max ratio", "reached only %.2f, and the gain stayed at %.2f points" % (e2c[('xyzibd', 0.0)]['ratio'], e2c[('xyzibd', 0.0)]['delta'])),
 ("S19 B1 cost", "up to %.2f points" % (e2c[('ipd', 0.2)]['b1'] - e2c[('ipd', 0.32)]['b1'])),
 ("S19 false per correct", "%.1f and %.1f on XYZ-IBD, against %.1f to %.1f on IPD, %.1f on IC-BIN" % (e2c[('xyzibd', 0.2)]['false_per_correct'], e2c[('xyzibd', 0.0)]['false_per_correct'], e2c[('ipd', 0.32)]['false_per_correct'], e2c[('ipd', 0.2)]['false_per_correct'], e2c[('icbin', 0.2)]['false_per_correct'])),
 ("S19 rule 3 range", "at most about %d false poses" % round(max(e2c[('xyzibd', t)]['false_per_correct'] for t in (0.0, 0.2))) if min(c['false_per_correct'] for c in s19['E2']['conditions'] if c['ds'] != 'xyzibd' and c['false_per_correct']) >= 9 else "MISMATCH"),
 ("S19 sweep vs Table 4", "by at most %.2f points" % max(abs(e2c[('xyzibd', 0.2)]['delta'] - (50.44 - 41.90)), abs(e2c[('ipd', 0.2)]['delta'] - (e3['ipd']['ap']['B4'] - e3['ipd']['ap']['B1'])))),
 ("S19 sweep XYZ ratio", "ratio is %.2f instead of %.2f" % (e2c[('xyzibd', 0.2)]['ratio'], c['xyzibd']['median_ratio'])),
 ("S19 rejection", "lost %.2f [%.2f, %.2f] points" % (-e3['ipd']['delta'], -e3['ipd']['ci'][1], -e3['ipd']['ci'][0])),
 ("S19 lowest classic ratio", "the lowest and the highest median count ratio" if min(s19['E2']['classic_median_ratios'].values()) >= 1.0 else "MISMATCH"),
]

# ---- 33-20 (data33_20/derived_s20.json, written by scripts/make_s20.py)
s20 = json.load(open(os.path.join(os.path.dirname(__file__), '..', 'data33_20', 'derived_s20.json')))
d20 = lambda k: "%.2f [%.2f, %.2f]" % tuple(s20["diff"][k])
checks += [
 ("S20 reversed pairs", "orders %.2f\\%% of the cross-call pairs correctly and reaches %.2f AR" % (s20["pairwise_reversed_strict_pct"], s20["AR_full_mean"]["A1neg"])),
 ("S20 reversed - joint", d20("full:A1neg-A4") + " points above joint scoring"),
 ("S20 calibrated - joint", "(" + d20("k10:A1cal10-A4") + ";"),
 ("S20 slopes", "negative slope in all 1,000 random draws" if s20["sign_learnability"]["10"]["combos_at_100"] == 10 and s20["sign_learnability"]["10"]["unfit"] == 0 else "MISMATCH"),
 ("S20 categories", "order correctly %.2f\\%% of the pairs whose incorrect pose lies on the background or on clutter and %.2f\\%% of those whose incorrect pose lies on another object, but %.2f\\%%" % (s20["categories"]["c"]["strict_pct"], s20["categories"]["a"]["strict_pct"], s20["categories"]["b"]["strict_pct"])),
 ("S20 share", "form %.2f\\%% of the reversed pairs" % s20["categories"]["c"]["share_pct"]),
 ("S20 readings", "on par with joint scoring" if s20["readings"]["R1"] == "持平" and s20["readings"]["R3"] == "持平" and s20["readings"]["R5"] else "MISMATCH"),
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
