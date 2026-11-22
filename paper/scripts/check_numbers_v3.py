#!/usr/bin/env python3
"""Recompute the numbers quoted in the running text from the deliverable tables and check that each appears in the text.
Usage: python3 scripts/check_numbers_v3.py [MANUSCRIPT_ROOT]
Checks marked [supp] are made against supplementary/*.tex under the same root. Without a manuscript root that contains sec/*.tex, the script prints the recomputed strings without checking them."""
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
 ("[drop] joint - CNOS", f"{ar['A4'][-1]-ar['A0'][-1]:.2f}"), ("[drop] TUD-L per-call", f"{ar['A1'][2]:.2f}"), ("[drop] LM-O per-call", f"{ar['A1'][0]:.2f}"),
 ("z-score", f"{ar['A7'][-1]:.2f}"), ("softmax", f"{ar['A8'][-1]:.2f}"), ("margin", f"{ar['A9'][-1]:.2f}"), ("[drop] MegaPose", f"{ar['A5'][-1]:.2f}"),
 ("pairwise raw", f"{100*ed['cross_mask_combined']['raw']['accuracy']:.2f}\\%"), ("pairwise joint", f"{100*ed['cross_mask_combined']['joint']['accuracy']:.2f}\\%"),
 ("[supp] pairwise megapose", f"{100*ed['cross_mask_combined']['megapose']['accuracy']:.2f}\\%"), ("pairs", f"{ed['cross_mask_combined']['raw']['opposite_label_pairs']:,}"),
 ("[supp] setsens FP", "0.03 units, but the 90th percentile is %.2f and the maximum %.2f" % (dv["setsens"]["FP"]["normalized_abs_delta"]["p90"], dv["setsens"]["FP"]["normalized_abs_delta"]["max"])),
 ("[supp] setsens MP", "%.3f, %.2f and %.2f" % (dv["setsens"]["MegaPose"]["normalized_abs_delta"]["median"], dv["setsens"]["MegaPose"]["normalized_abs_delta"]["p90"], dv["setsens"]["MegaPose"]["normalized_abs_delta"]["max"])),
 ("[supp] targets >=2", f"{dv['setsens']['FP']['targets_at_least_two']:,} targets"), ("[supp] top changes FP", f"{dv['setsens']['FP']['winner_changed']:,} of these"),
 ("[supp] correct top FP", f"{dv['setsens']['FP']['solo_correct_top']:,} to {dv['setsens']['FP']['joint_correct_top']:,}"),
 ("[supp] correct top MP", f"{dv['setsens']['MegaPose']['solo_correct_top']:,} targets either way" if dv['setsens']['MegaPose']['solo_correct_top']==dv['setsens']['MegaPose']['joint_correct_top'] else "MISMATCH"),
 ("[supp] raw medians", "%.1f for correct and %.1f for incorrect" % (dv["score_medians"]["pooled"]["s_raw"]["correct"], dv["score_medians"]["pooled"]["s_raw"]["incorrect"])),
 ("[supp] Fcov ratio LM-O", "35\\%" if round(100*1.89/5.42) == 35 and round(100*4.38/12.37) == 35 else "MISMATCH"),
 ("[supp] XYZ single repairs sum", f"{0.19+0.88+4.54+0.0+0.70+0.0:.2f}"),
 ("[supp] IPD B4 outputs", f"{ib4['outputs']:,}"), ("[supp] IPD background", f"{ib4['background']:,}"),
 ("[supp] IPD background share of outputs", f"{round(100*ib4['background']/ib4['outputs'])}\\% of all outputs"),
 ("[supp] IPD background share of false", f"{round(100*ib4['background']/fp(ib4))}\\% of the false"),
 ("[supp] IPD added TP / FP", f"{ib4['TP']-ib1['TP']} correct poses and {fp(ib4)-fp(ib1):,} false"), ("[supp] IPD ratio", f"{(fp(ib4)-fp(ib1))/(ib4['TP']-ib1['TP']):.1f} false"),
 ("[supp] XYZ added TP / FP", f"{xb4['TP']-xb1['TP']} correct poses and {fp(xb4)-fp(xb1):,} false"), ("[supp] XYZ ratio", f"{(fp(xb4)-fp(xb1))/(xb4['TP']-xb1['TP']):.1f} for each"),
 ("[supp] XYZ below one", f"{round(100*c['xyzibd']['share_below_1'])}\\% of the XYZ-IBD frames"), ("[supp] IPD below one", f"{round(100*c['ipd']['share_below_1'])}\\% of the IPD frames"),
 ("B0 IPD background", f"{round(100*c['ipd']['B0']['background']/c['ipd']['B0']['outputs'])}\\% of the detector"),
 ("[supp] density A4 rho", f"{dv['density_A4_basis_posthoc']['A4_spearman']['rho']:.2f} [{dv['density_A4_basis_posthoc']['A4_spearman']['ci95'][0]:.2f}, {dv['density_A4_basis_posthoc']['A4_spearman']['ci95'][1]:.2f}]"),
 ("[supp] T-LESS mean A4 gain", f"{dv['density_A4_basis_posthoc']['by_dataset_A4_mean']['tless']:.2f}"),
 ("[drop] A4-A5 by dataset", (lambda r: "T-LESS (%s) and LM-O (%s)" % (r[2], r[1]))(next(r for r in md_table("TABLES_P1.md", "E-A 的配对差（百分点）") if r[0] == "A4-A5"))),
 ("[supp] Y8 K16", "%.2f\\%%, %.2f\\%% and %.2f\\%% of all %s pairs" % tuple([float(r[i].split()[0]) for r in md_table("TABLES_Y8.md", "K=16 跨调用成对准确率（%）") if r[0] == "五集合并" for i in (2, 3, 4)] + [f"{36852:,}"])),
 ("[supp] Y8 FP", "inverted (%.2f\\%% and %.2f\\%%)" % tuple(float(r[i].split()[0]) for r in md_table("TABLES_Y8.md", "K=16 跨调用成对准确率（%）") if r[0] == "五集合并" for i in (5, 7))),
 ("[supp] minimal B1->B4", f"+{50.44-41.90:.1f}"), ("instances per frame", f"{3990/150:.1f}"), ("IPD instances per frame", f"{548/88:.1f}"),
] + (lambda m: [
 ("[supp] S18 coverage", "from %d\\%% to %d--%d\\%%" % (round(100*m['V0']['identity_correct_cover']), round(100*min(m[k]['identity_correct_cover'] for k in m if k != 'V0')), round(100*max(m[k]['identity_correct_cover'] for k in m if k != 'V0')))),
 ("[supp] S18 false", "from %.1f to %d--%d" % (m['V0']['false_detections_per_frame_mean'], round(min(m[k]['false_detections_per_frame_mean'] for k in m if k != 'V0')), round(max(m[k]['false_detections_per_frame_mean'] for k in m if k != 'V0')))),
])(json.load(open(os.path.join(os.path.dirname(__file__), '..', 'data33_18', 'S18_2D_V11.json')))['cohorts']['Cprime']) + [
 ("[supp] EC doubling", "about 9" if round(float(x('EC 配对差')['ECD-ECB'][1].split()[0])) == 9 and round(float(x('EC 配对差')['ECE-ECB'][1].split()[0])) == 9 else "MISMATCH"),
 ("[supp] slots k C-B", "segments lead by %.2f points" % (float(x('EC 评测')['ECC'][1]) - float(x('EC 评测')['ECB'][1]))),
 ("[drop] count given candidates", "adds %.1f AP points" % float({r[0]: r for r in md_table('TABLES_P2.md', 'Y1 配对差')}['B4-B3'][1].split()[0])),
 ("[drop] joint over det x depth", "detector confidence times depth consistency within %.2f" % (ar['A4'][-1] - ar['A10'][-1]) if 'A10' in ar else "MISMATCH"),
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
 ("S19 raw same pairs", f"({e1['FPsingle']['wins']:,} against {e1['FPraw']['wins']:,} correct orders"),
 ("[drop] S19 AR alone/raw", f"({e1['FPsingle']['ar']:.2f} against {e1['FPraw']['ar']:.2f})"),
 ("S19 MegaPose", f"{e1['MPsingle']['pooled']:.2f}\\% of the cross-call pairs correctly when it scores each pose alone and {e1['MPjoint']['pooled']:.2f}\\%"),
 ("[supp] S19 fit range", f"{round(min(e1[k]['pooled'] for k in ('Depth','MaskIoU','Chamfer')))}--{round(max(e1[k]['pooled'] for k in ('Depth','MaskIoU','Chamfer')))}\\% of the pairs"),
 ("S19 GigaPose", f"{e1['native']['gigapose']['pooled']:.2f}\\% (GigaPose)"),
 ("S19 SAM-6D", f"{e1['native']['sam6d']['pooled']:.2f}\\% (SAM-6D)"),
 ("[supp] S19 IPD ratios", "from %.2f to %.2f, %.2f and %.2f" % tuple(e2c[('ipd', t)]['ratio'] for t in (0.2, 0.26, 0.29, 0.32))),
 ("[supp] S19 IPD loss official", neg(e2c[('ipd', 0.2)]) + " points at %.2f" % e2c[('ipd', 0.2)]['ratio']),
 ("[supp] S19 IPD losses", "to %.2f and %.2f" % (-e2c[('ipd', 0.26)]['delta'], -e2c[('ipd', 0.29)]['delta'])),
 ("[supp] S19 IPD low", "+%.2f [-%.2f, %.2f] at %.2f" % (e2c[('ipd', 0.32)]['delta'], -e2c[('ipd', 0.32)]['ci'][0], e2c[('ipd', 0.32)]['ci'][1], e2c[('ipd', 0.32)]['ratio'])),
 ("[supp] S19 IC-BIN", neg(e2c[('icbin', 0.2)]) + " points on IC-BIN (ratio %.2f)" % e2c[('icbin', 0.2)]['ratio']),
 ("[supp] S19 TUD-L", neg(e2c[('tudl', 0.2)]) + " on TUD-L (%.1f)" % e2c[('tudl', 0.2)]['ratio']),
 ("[supp] S19 XYZ max ratio", "reached only %.2f, and the gain stayed at %.2f points" % (e2c[('xyzibd', 0.0)]['ratio'], e2c[('xyzibd', 0.0)]['delta'])),
 ("[supp] S19 B1 cost", "up to %.2f points" % (e2c[('ipd', 0.2)]['b1'] - e2c[('ipd', 0.32)]['b1'])),
 ("[supp] S19 false per correct", "%.1f and %.1f on XYZ-IBD, against %.1f to %.1f on IPD, %.1f on IC-BIN" % (e2c[('xyzibd', 0.2)]['false_per_correct'], e2c[('xyzibd', 0.0)]['false_per_correct'], e2c[('ipd', 0.32)]['false_per_correct'], e2c[('ipd', 0.2)]['false_per_correct'], e2c[('icbin', 0.2)]['false_per_correct'])),
 ("[drop] S19 rule 3 range", "at most about %d false poses" % round(max(e2c[('xyzibd', t)]['false_per_correct'] for t in (0.0, 0.2))) if min(c['false_per_correct'] for c in s19['E2']['conditions'] if c['ds'] != 'xyzibd' and c['false_per_correct']) >= 9 else "MISMATCH"),
 ("[supp] S19 sweep vs Table 4", "by at most %.2f points" % max(abs(e2c[('xyzibd', 0.2)]['delta'] - (50.44 - 41.90)), abs(e2c[('ipd', 0.2)]['delta'] - (e3['ipd']['ap']['B4'] - e3['ipd']['ap']['B1'])))),
 ("[supp] S19 sweep XYZ ratio", "ratio is %.2f instead of %.2f" % (e2c[('xyzibd', 0.2)]['ratio'], c['xyzibd']['median_ratio'])),
 ("[drop] S19 rejection", "lost %.2f [%.2f, %.2f] points" % (-e3['ipd']['delta'], -e3['ipd']['ci'][1], -e3['ipd']['ci'][0])),
 ("[supp] S19 lowest classic ratio", "the lowest and the highest median count ratio" if min(s19['E2']['classic_median_ratios'].values()) >= 1.0 else "MISMATCH"),
]
bad = 0
# ---- 33-20 (data33_20/derived_s20.json, written by scripts/make_s20.py)
s20 = json.load(open(os.path.join(os.path.dirname(__file__), '..', 'data33_20', 'derived_s20.json')))
d20 = lambda k: "%.2f [%.2f, %.2f]" % tuple(s20["diff"][k])
checks += [
 ("S20 reversed pairs", "reaches %.2f AR" % s20["AR_full_mean"]["A1neg"]),
 ("S20 reversed - joint", d20("full:A1neg-A4") + " points above joint scoring"),
 ("[drop] S20 calibrated - joint", "(" + d20("k10:A1cal10-A4") + ";"),
 ("[supp] S20 slopes", "every fitted slope is negative from k=10 on" if s20["sign_learnability"]["10"]["combos_at_100"] == 10 and s20["sign_learnability"]["10"]["unfit"] == 0 else "MISMATCH"),
 ("[supp] S20 categories", "order correctly %.2f\\%% of the pairs whose incorrect pose lies on the background or on clutter, %.2f\\%% of those whose incorrect pose lies on another object, and %.2f\\%%" % (s20["categories"]["c"]["strict_pct"], s20["categories"]["a"]["strict_pct"], s20["categories"]["b"]["strict_pct"])),
 ("[supp] S20 share", "form %.2f\\%% of the reversed pairs" % s20["categories"]["c"]["share_pct"]),
 ("[supp] S20 readings", "on par with joint scoring" if s20["readings"]["R1"] == "持平" and s20["readings"]["R3"] == "持平" and s20["readings"]["R5"] else "MISMATCH"),
]

# ---- 33-21 (data33_21/derived_s21.json, written by scripts/make_s21.py)
s21 = json.load(open(os.path.join(os.path.dirname(__file__), '..', 'data33_21', 'derived_s21.json')))
under = [k for k, v in s21["median_count_ratio"].items() if v < 1]
num = {9: "nine", 8: "eight"}
p21 = s21["pilot_A"]
checks += [
 ("[supp] S21 screened", "of %s datasets screened at the official threshold, only %s under-reports" % (num[s21["n_datasets"]], under[0]) if s21["n_under_reporting"] == 1 and len(under) == 1 else "MISMATCH"),
 ("[drop] S21 dense over-report", "two further dense datasets that we screened both over-report" if s21["e1_status"] == "no_sparse_dataset" and min(s21["median_count_ratio"][k] for k in ("HomebrewedDB", "HOPE-Image")) > 1 else "MISMATCH"),
 ("S21 pilot not inverted", "a small scorer trained to rank within a call did not invert" if p21["cross_call_ci_pct"][0] > 50 else "MISMATCH"),
 ("[supp] S21 pilot", "It ranked %.2f\\%% of within-call pairs correctly and %.2f\\%% of cross-call pairs, %.2f\\%% when each pose was scored alone" % (p21["within_pct"], p21["cross_call_pct"], p21["alone_pct"])),
 ("[supp] S21 only under-reporter", "Of the %s datasets whose count ratio we computed at the official threshold, %s is the only one" % (num[s21["n_datasets"]], under[0])),
]
# ---- usage survey (data_survey/derived_usage.json, written by scripts/make_usage.py)
us = json.load(open(os.path.join(os.path.dirname(__file__), '..', 'data_survey', 'derived_usage.json')))
words = {2: "two", 3: "Three", 6: "six"}
checks += [
 ("usage main", "Of the %d leading entries on BOP-Classic-Core, %d use FoundationPose" % (us["n_entries"], us["n_use_foundationpose"]) if us["n_state_ranking_across_detections"] == 0 else "MISMATCH"),
 ("usage two entries", "%s entries, SAM-FP and SAM3-FS-FP, state that they submit a reranking score of FoundationPose" % words[len(us["industrial_submit_fp_score"])]),
 ("[supp] usage counts", "Of the %d leading entries, %d use FoundationPose" % (us["n_entries"], us["n_use_foundationpose"])),
 ("[supp] usage roles", "%s pages state that its score network selects the final pose of each detection, %s name it as a refiner" % (words[us["n_state_score_network_selects"]], words[us["n_state_refiner"]])),
 ("[supp] usage confidence", "No page states what is submitted as the confidence of a pose" if us["n_state_confidence"] == 0 else "MISMATCH"),
 ("[supp] usage industrial", " and ".join(us["industrial_submit_fp_score"]) + ", state that they register"),
]
# ---- 33-22 (data33_22/derived_s22.json, written by scripts/make_s22.py) and further main-text numbers
s22 = json.load(open(os.path.join(os.path.dirname(__file__), '..', 'data33_22', 'derived_s22.json')))
apc, tpc, e3r, e4d = s22["E1_classic_P1_AP"], s22["E2_classic_P1_top1"], s22["E3"]["readings"], s22["E4"]
bx, bi = s22["bins"]["xyzibd"], s22["bins"]["ipd"]
f2 = lambda v: "%.2f" % v
ea = {r[0]: r for r in md_table("TABLES_P1.md", "E-A 的配对差（百分点）")}
y2 = {r[0]: r for r in md_table("TABLES_P2.md", "Y2 localization")}
wc = ed["within_call_by_dataset"]
lad = s22["E3"]["ladders"]
checks += [
 ("within-call pooled", "%.2f\\%% of pairs" % (100 * sum(v["correct_first"] for v in wc.values()) / sum(v["opposite_label_pairs"] for v in wc.values()))),
 ("within-call range", "%.2f\\%% to %.2f\\%%" % (100 * min(v["accuracy"] for v in wc.values()), 100 * max(v["accuracy"] for v in wc.values()))),
 ("[drop] joint - per-call CI", ea["A4-A1"][-1]), ("joint - MegaPose CI", ea["A4-A5"][-1]),
 ("[drop] softmax gap CI", {r[0]: r for r in md_table("TABLES_P1.md", "X2 的配对差（百分点）")}["A4-A8"][-1]),
 ("[drop] TUD-L A0", f"{ar['A0'][2]:.2f} to {ar['A1'][2]:.2f}"),
 ("CNOS alone per-call / joint", "reaches %.2f AR, against %.2f with joint scoring" % (ar["A1c"][-1], ar["A4c"][-1])),
 ("density single/dense", "from %.1f points in frames with one instance of an object to %.1f" % (float(y2["1"][4].split()[0]), float(y2["≥16"][4].split()[0]))),
 ("E1 C0", "reach %s AP" % f2(apc["C0"])), ("E1 C1", "lowers AP to %s" % f2(apc["C1"])), ("E1 C3", "gives %s" % f2(apc["C3"])),
 ("E1 C4", "Joint scores reach %s AP" % f2(apc["C4"])), ("E1 C2 C5", "recovers %s, MegaPose %s" % (f2(apc["C2"]), f2(apc["C5"]))), ("E1 C7", "reaches %s, 0.80" % f2(apc["C7"])),
 ("E1 rounded", "from %.1f, with the detector's confidence, to %.1f" % (apc["C0"], apc["C1"])), ("E1 abstract", "from %.1f to %.1f" % (apc["C0"], apc["C1"])),
 ("E1 28", "cut detection AP by %d points" % round(apc["C0"] - apc["C1"])), ("E1 28 sec", "cuts AP by %d points" % round(apc["C0"] - apc["C1"])),
 ("E1 XYZ", "loses %s points against the detector's confidence (%s against %s AP)" % (f2(bx["AP"]["C0"] - bx["AP"]["C1"]), f2(bx["AP"]["C1"]), f2(bx["AP"]["C0"]))),
 ("E1 IPD", "it reaches %s AP, against %s with the reversed score and %s with depth consistency" % (f2(bi["AP"]["C1"]), f2(bi["AP"]["C2"]), f2(bi["AP"]["C6"]))),
 ("E2 classic", "correct in %s\\%% of the frames, against %s\\%% by detector confidence and %s\\%% expected in random order" % (f2(tpc["C1"]), f2(tpc["C0"]), f2(s22["E2_classic_P1_random"]))),
 ("E2 joint", "Joint scores reach %s\\%%" % f2(tpc["C4"])), ("E2 reversed", "first in %s\\%% of the frames, and detector confidence times depth consistency in %s\\%%" % (f2(tpc["C2"]), f2(tpc["C7"]))),
 ("E2 rounded", "correct in %d\\%% of frames, against %d\\%% by detector confidence and %d\\%% in random order" % (round(tpc["C1"]), round(tpc["C0"]), round(s22["E2_classic_P1_random"]))),
 ("E2 abstract", "correct in %d\\%% of frames, against %d\\%% for the detector's confidence" % (round(tpc["C1"]), round(tpc["C0"]))),
 ("E2 XYZ", "%s\\%% of the frames by per-call scores and in %s\\%% by detector confidence" % (f2(bx["top1"]["C1"]), f2(bx["top1"]["C0"]))),
 ("E2 IPD", "none of the 78 frames by per-call or joint scores and in %s\\%% by the reversed score" % f2(bi["top1"]["C2"]) if bi["top1"]["C1"] == 0 and bi["top1"]["C4"] == 0 else "MISMATCH"),
 ("E3 R9", "in %d of %d instances, and the incorrect rotation rungs in %d of %d" % (e3r["R9"]["positive"], e3r["R9"]["eligible"], e3r["R12"]["positive"], e3r["R12"]["eligible"])),
 ("E3 intro", "in all %d instances that we shifted off the target and, from 45^\\circ on, in all %d that we rotated on it" % (e3r["R9"]["eligible"], e3r["R12"]["eligible"]) if e3r["R9"]["positive"] == e3r["R9"]["eligible"] and e3r["R12"]["positive"] == e3r["R12"]["eligible"] else "MISMATCH"),
 ("E3 R11", "in only %d of %d instances" % (e3r["R11"]["positive"], e3r["R11"]["eligible"])),
 ("E3 R10", "in %s\\%% of the lateral ladders, meeting the threshold of 95\\%% fixed in advance, and in %s\\%% of the rotation ladders" % (f2(e3r["R10_T_xy"]["pct"]), f2(e3r["R10_R"]["pct"]))),
 ("E3 posthoc", "In all %d rotation ladders where another rung came first, that rung was itself a correct pose" % s22["E3"]["posthoc_R10_R_failures"] if s22["E3"]["posthoc_R10_R_failures"] == s22["E3"]["posthoc_R10_R_failures_with_correct_top"] else "MISMATCH"),
 ("E3 instances", "we took %d ground-truth poses" % s22["E3"]["instances"]),
 ("E3 lateral", "A lateral shift of 0.05D_o already scores above the true pose in %.1f\\%% of the instances, and from 0.2D_o on in all of them" % lad["T_xy"]["fp_single"]["0.05"]["share_higher_pct"] if all(lad["T_xy"]["fp_single"][k]["share_higher_pct"] == 100 for k in ("0.2", "0.5", "1", "2", "4")) else "MISMATCH"),
 ("E3 rotation", "%.1f\\%% of the instances score higher after 5^\\circ and all after 45^\\circ" % lad["R"]["fp_single"]["5"]["share_higher_pct"] if all(lad["R"]["fp_single"][k]["share_higher_pct"] == 100 for k in ("45", "90", "180")) else "MISMATCH"),
 ("E4 setsens", "the 90th percentile of the set sensitivity is %.2f for WBPS and 2.02 for FoundationPose, against 0.02 for MegaPose" % e4d["M2_P90"]),
 ("E4 WBPS pairwise", "WBPS orders %.2f\\%% [%.2f, %.2f] of the cross-call pairs correctly, and %.2f\\%% of the pairs within a group" % (e4d["pairwise"]["wbps_raw"]["pct"], *e4d["pairwise"]["wbps_raw"]["ci"], e4d["within_group_pct"])),
 ("E4 W1-W0", "raises AR by %.2f" % (e4d["AR"]["W1"] - e4d["AR"]["W0"])),
 ("E4 FP on WAPR", "ordered %.2f\\%% [%.2f, %.2f] of the %s cross-call pairs" % (e4d["pairwise"]["fp_single"]["pct"], *e4d["pairwise"]["fp_single"]["ci"], format(e4d["pairwise"]["fp_single"]["pairs"], ","))),
 ("E4 joint construction", "lowered the cross-call pairwise accuracy to %.2f\\%% and AR by %.2f points" % (e4d["pairwise"]["wbps_joint"]["pct"], e4d["AR"]["W1"] - e4d["AR"]["W4"])),
 ("[supp] E4 joint construction", "ordered %.2f\\%% of the cross-call pairs correctly and lowered AR by %.2f points" % (e4d["pairwise"]["wbps_joint"]["pct"], e4d["AR"]["W1"] - e4d["AR"]["W4"])),
 ("[supp] E4 top changed", "changed the top-ranked pose in %s of %s targets" % (format(e4d["top_changed"], ","), format(e4d["targets"], ","))),
 ("[supp] E4 masks", "%s of %s masks yielded a pose" % (format(e4d["masks_valid"], ","), format(e4d["masks_requested"], ","))),
 ("[supp] E3 out of frame", "Of the 12,000 rungs, %d left the image" % s22["E3"]["out_of_frame"] if s22["E3"]["scored_poses"] + s22["E3"]["out_of_frame"] == 12000 else "MISMATCH"),
]
ROOT_M = sys.argv[1] if len(sys.argv) > 1 else "."
supp = " ".join(open(f).read() for f in sorted(glob.glob(os.path.join(ROOT_M, "supplementary", "*.tex")))).replace("$-$", "-").replace("$", "")

# ---- 33-23 (data33_23/derived_s23.json, written by scripts/make_s23.py)
s23 = json.load(open(os.path.join(os.path.dirname(__file__), '..', 'data33_23', 'derived_s23.json')))
_b0, _b1, _pa, _c1 = s23["B"]["W0"], s23["B"]["W1"], s23["A"], s23["C1"]
_neg = lambda r: "%.2f [%.2f, %.2f]" % (-r["point"], -r["ci"][1], -r["ci"][0])
_pos = lambda r: "%.2f [%.2f, %.2f]" % (r["point"], r["ci"][0], r["ci"][1])
_m = s23["C3"]["mean"]
checks += [
 ("33-23 C1-C0 classic", "28.01 [27.37, 28.86] points" if _neg(_c1["classic_C1_C0"]).startswith("28.01 [27.37, 28.86]") else "MISMATCH"),
 ("33-23 E4 W1-W0 BCa", _pos(_c1["E4_W1_W0"])),
 ("33-23 AP offset range", "by %.2f to %.2f points" % (0.04, max(_c1["ap_offset_max_E1"], _c1["ap_offset_max_B"]))),
 ("33-23 AP intervals outside", "%d intervals of single AP values" % _c1["outside"]),
 ("33-23 subset max mean diff", "within %.2f points" % max(v["diff"] for v in _m.values())),
 ("33-23 W0 WBPS AP", "WBPS gives %.2f AP" % _b0["ap"]["B-W"]["point"]),
 ("33-23 W0 WBPS-det", _neg(_b0["d_ap"]["B-W-B-C0"]) + " points below the detector"),
 ("33-23 W0 det AP", "(%.2f)" % _b0["ap"]["B-C0"]["point"]),
 ("33-23 W0 WBPS-depth", "%.2f points above depth consistency (%.2f)" % (_b0["d_ap"]["B-W-B-D"]["point"], _b0["ap"]["B-D"]["point"])),
 ("33-23 W0 FP alone AP", "gives %.2f AP, %s points below" % (_b0["ap"]["B-F"]["point"], _neg(_b0["d_ap"]["B-F-B-C0"]))),
 ("33-23 W1 AP", "WBPS to %.2f and depth consistency to %.2f AP against %.2f" % (_b1["ap"]["B-W"]["point"], _b1["ap"]["B-D"]["point"], _b1["ap"]["B-C0"]["point"])),
 ("33-23 W0 top1", "%.2f\\%% of the frames, %s points more often" % (_b0["top1"]["B-W"]["point"], _pos(_b0["d_top1"]["B-W-B-C0"]))),
 ("33-23 W0 FP top1", "does so in %.2f\\%%, below the %.2f\\%%" % (_b0["top1"]["B-F"]["point"], _b0["random_top1"])),
 ("33-23 W1 top1", "first in %.2f\\%% of the frames" % _b1["top1"]["B-W"]["point"]),
 ("33-23 public files", "%d files of %d entries" % (_pa["files"], _pa["methods"])),
 ("33-23 public acc range", "between %.2f\\%% and %d\\%%" % (_pa["acc_min"], round(_pa["acc_max"]))),
 ("33-23 public with pairs", "the other %d files" % _pa["with_pairs"]),
 ("33-23 public reversal", "in %d of the %d files" % (_pa["R7"].get("降低", 0), _pa["eligible"])),
 ("33-23 public spearman", "\\rho=%.2f over %d files" % (_pa["spearman_acc_vs_reversal_cost"], _pa["spearman_n"])),
 ("33-23 public depth", "lowers the result in %d files" % _pa["R6"].get("降低", 0)),
 ("33-23 SAM3 acc", "(%.2f\\%% and %.2f\\%%)" % tuple(sorted(c["acc"] for c in s23["A_cases"] if c["method"].startswith("SAM3")))),
 ("33-23 intro files", "none of the %d public submission files" % _pa["with_pairs"]),
]
# ---- 33-24 (data33_24/derived_s24.json, written by scripts/make_s24.py)
s24 = json.load(open(os.path.join(os.path.dirname(__file__), '..', 'data33_24', 'derived_s24.json')))
_k10 = s24["k"]["10"]
_fp = s24["fp_pool"]["k"]
_ap = s24["ap_differences"]
_acc = list(s24["fp_pool_accuracy_pp"][k] for k in ("Depth", "MaskIoU", "Chamfer")) + [s24["full_sample"]["depth"]["accuracy_pp"]]
_rb3 = [int(v) for v in s24["RB3_min_k"]["depth"].values()] + [int(v) for d in s24["fp_pool"]["RB3_min_k"].values() for v in d.values()]
_ivl = lambda v: "%.2f [%.2f, %.2f]" % (abs(v["point"]), *sorted(abs(x) for x in v["bca"]))
checks += [
 ("E3 rotation from 45", "from 45^\\circ on, in all" if all(json.load(open(os.path.join(os.path.dirname(__file__), '..', 'data33_22', 'derived_s22.json')))["E3"]["ladders"]["R"]["fp_single"][a]["share_higher_pct"] == 100 for a in ("45", "90", "180")) else "MISMATCH"),
 ("33-24 scene large", "keeps every AP difference of three points or more away from zero" if all((v["scene_bca"][0] > 0 or v["scene_bca"][1] < 0) for v in _ap.values() if abs(v["point"]) >= 3) else "MISMATCH"),
 ("33-24 scene small", "whereas some smaller ones then include zero" if any(v["scene_bca"][0] <= 0 <= v["scene_bca"][1] for v in _ap.values() if abs(v["point"]) < 3) else "MISMATCH"),
 ("intro softmax gap", "yet it stays %s points below joint scoring" % re.search(r"Softmax within its call.*?\+([\d.]+) \[", open(os.path.join(os.path.dirname(__file__), '..', 'tables', 'tab_rankings.tex')).read()).group(1)),
 ("[drop] 33-24 tab sign classic", "in all 1,000 draws on each classic dataset" if _fp["10"]["classic_at_100"] == 5 else "MISMATCH"),
 ("[drop] 33-24 tab sign xyz", "and in %.1f\\%% on XYZ-IBD" % _fp["10"]["xyz_correct"]),
 ("[drop] 33-24 never comparable", "and never as comparable" if all(_fp[k]["highest_wrong"] == 0 for k in ("10", "20")) else "MISMATCH"),
 ("[drop] 33-24 13 scores never wrong", "read none of %d scores the wrong way" % s24["n_scores_replayed"] if s24["fp_pool"]["never_wrong_k10"] else "MISMATCH"),
 ("33-24 fit scores range", "scores at %d\\%% to %d\\%%, such as depth consistency" % (round(min(_acc)), round(max(_acc)))),
 ("[drop] 33-24 up to ten", "needed up to ten for nine correct readings" if max(_rb3) == 10 else "MISMATCH"),
 ("33-24 C7", "%.2f [%.2f, %.2f] points above the detector" % (_ap["classic_P1/C7-C0"]["point"], *_ap["classic_P1/C7-C0"]["bca"])),
 ("33-24 C7 neither", "lies neither within \\pm1 point nor above it" if _ap["classic_P1/C7-C0"]["bca"][0] < 1 < _ap["classic_P1/C7-C0"]["bca"][1] else "MISMATCH"),
 ("33-24 C1 interval", "a loss of %s points" % _ivl(_ap["classic_P1/C1-C0"])),
 ("33-24 WBPS interval", "%s points below the detector's confidence" % _ivl(_ap["classic_W0/B-W-B-C0"])),
 ("33-24 FP alone W0", "%s points below the detector's confidence (70.84" % _ivl(_ap["classic_W0/B-F-B-C0"])) if False else ("33-24 FP alone W0", "%s points below the detector's confidence, as on its own poses" % _ivl(_ap["classic_W0/B-F-B-C0"])),
 ("33-24 cover 11 of 12", "whose intervals cover the point in %d of %d main contrasts" % (s24["ap_n"] - s24["ap_noncover_image"], s24["ap_n"])),
 ("[supp] 33-24 S1 eleven", "covers the point estimate in eleven and the BCa interval over scenes in all twelve" if (s24["ap_noncover_image"], s24["ap_noncover_scene"], s24["ap_n"]) == (1, 0, 12) else "MISMATCH"),
 ("[supp] 33-24 S1 jack", "standard error of %.2f points and a normal interval of [-%.2f, -%.2f]" % (_ap["classic_P1/C4-C0"]["jackknife"]["SE_pp"], -_ap["classic_P1/C4-C0"]["jackknife"]["ci95_pp"][0], -_ap["classic_P1/C4-C0"]["jackknife"]["ci95_pp"][1])),
 ("[drop] 33-24 k3 wrong", "at most %.1f\\%% of the draws of three frames" % max(s24["k"]["3"]["highest_wrong"], _fp["3"]["highest_wrong"])),
 ("[drop] 33-24 k10 fit lowest", "in at least %.1f\\%% of the draws and otherwise" % s24["fp_pool"]["k10_lowest_any_score"]),
 ("[drop] 33-24 xyz depth", "depth consistency as comparable in %.1f\\%%" % _fp["10"]["xyz_depth_correct"]),
 ("[drop] 33-24 xyz no pairs quarter", "a quarter of the draws there hold no pairs" if 24 <= _fp["3"]["xyz_no_pairs"] <= 26 else "MISMATCH"),
 ("[drop] 33-24 xyz misses", "most misses with three or five frames are draws without" if all(_fp[k]["xyz_no_pairs"] > _fp[k]["xyz_insufficient"] for k in ("3", "5")) else "MISMATCH"),
 ("[drop] 33-24 none of 13", "None of the %d scores is read the wrong way in any draw of ten frames" % s24["n_scores_replayed"]),
 ("[supp] 33-24 full readings", "the check reads each score as expected in every dataset" if all(
     all(r == ("reversed" if k in ("fp_single", "wbps_neg") else "comparable") for r in v["readings_by_dataset"].values())
     for k, v in s24["full_sample"].items()) else "MISMATCH"),
]
# ---- 33-25 (data33_25/derived_s25.json, written by scripts/make_s25.py): the six-frame rule
s25 = json.load(open(os.path.join(os.path.dirname(__file__), '..', 'data33_25', 'derived_s25.json')))
_tool = open(os.path.join(os.path.dirname(__file__), '..', 'data33_25', 'tool_check_v1_1.py')).read()
_min_img = int(re.search(r"MIN_IMAGES = (\d+)", _tool).group(1))
_words = {3: "three", 6: "six", 10: "ten"}
_fm = s25["seq_frames_median_classic"]
_pub = s25["public"]
_r10 = min(min(s25["replay_fp"]["10"]["lowest_correct"].values()), min(s25["replay_wapr"]["10"]["lowest_correct"].values()))
_r20 = min(min(s25["replay_fp"]["20"]["lowest_correct"].values()), min(s25["replay_wapr"]["20"]["lowest_correct"].values()))
checks += [
 ("33-25 abstract ten with a pair", "detects the failure from ten labelled frames that each hold a correct and an incorrect pose" if s25["criteria"]["K4_seq10_fp_ge95_all_six"] and all(v["correct"] == 100 for v in s25["seq_fp"].values()) else "MISMATCH"),
 ("33-25 six frames", "only from at least %s frames that hold a pair" % _words[_min_img]),
 ("33-25 frames needed", "took a median of %d to %d frames on the classic datasets and %d on XYZ-IBD" % (_fm[0], _fm[1], s25["seq_fp"]["xyzibd"]["frames"]["median"])),
 ("33-25 all draws", "reversed in all 1,000 draws on every dataset" if all(v["correct"] == 100 for v in s25["seq_fp"].values()) else "MISMATCH"),
 ("33-25 null max", "in at most %.1f\\%% of the draws" % max(s25["null_v11_max"], s25["seqnull_max"])),
 ("33-25 never wrong", "read none of %d scores the wrong way" % s25["n_scores"] if s25["wrong_any_v11"] == 0 else "MISMATCH"),
 ("33-25 fit with ten", "were read correctly in at least %.1f\\%% of the draws with ten such frames" % s25["seq_lowest_any_score_classic"]),
 ("[supp] 33-25 k10 k20", "with ten, it reads every score correctly in at least %.1f\\%% of the draws, and with twenty in at least %.1f\\%%" % (_r10, _r20)),
 ("[supp] 33-25 xyz informative", "ten random frames hold a median of %s frames with pairs" % _words[s25["fp"]["10"]["informative_median"]["xyzibd"]]),
 ("[supp] 33-25 xyz k10", "as reversed in only %.1f\\%% of the draws" % s25["fp"]["10"]["xyz"]),
 ("[supp] 33-25 xyz k20", "with twenty frames it does so in %.1f\\%%" % s25["fp"]["20"]["xyz"]),
 ("[supp] 33-25 seq fit", "at least %.1f\\%% of the draws and otherwise read as not comparable" % s25["seq_lowest_any_score_classic"]),
 ("[supp] 33-25 null v10", "up to %.1f\\%% of the draws on the classic datasets and in %.1f\\%% on XYZ-IBD" % (s25["null_v10_classic_k10_range"][1], s25["null_v10_xyz_k10_max"])),
 ("[supp] 33-25 null v11", "in at most %.1f\\%% of the draws in any setting, %.1f\\%% on average at ten frames" % (max(s25["null_v11_max"], s25["seqnull_max"]), s25["null_v11_mean_k10"])),
 ("[supp] 33-25 null cells", "the share above 5\\%% in %d of the %d settings" % (s25["null_v11_cells_above5"], s25["null_cells"])),
 ("[supp] 33-25 full", "changes none of the %d readings" % s25["full_cells"] if s25["full_unchanged"] else "MISMATCH"),
 ("[supp] 33-25 S1 K1", "at most %.1f\\%%; above 5\\%% in %d of %d settings" % (max(s25["null_v11_max"], s25["seqnull_max"]), s25["null_v11_cells_above5"], s25["null_cells"]) if not s25["criteria"]["K1_null_le5_every_cell"] else "MISMATCH"),
 ("[supp] 33-25 S1 K2", "lowest %.1f\\%%" % s25["fp"]["10"]["classic_lowest"] if s25["criteria"]["K2_fp_k10_classic_ge90"] else "MISMATCH"),
 ("[supp] 33-25 S1 K3", "none in %d settings" % (s25["replay_cells"] + s25["seq_cells"]) if s25["criteria"]["K3_never_wrong"] else "MISMATCH"),
 ("[supp] 33-25 S1 K5", "0 of %d" % s25["full_cells"] if s25["criteria"]["K5_full_unchanged"] else "MISMATCH"),
 ("[supp] 33-25-S public rerun", "Rerun with its six-image rule on the %d files whose per-pose labels were kept, the released check reads %d files as comparable, %s as carrying no information and %s as without pairs" % (
     _pub["rechecked"], _pub["comparable"], {2: "two"}[_pub["no_information"]], {3: "three"}[_pub["no_pairs"]]) if _pub["reversed"] == 0 and _pub["not_comparable"] == 0 else "MISMATCH"),
 ("[supp] 33-25-S controls", "%s control files of FreeZeV2 hold one image with pairs each, too few for a reading, so %s of the %s control files are comparable" % (
     {2: "two"}[_pub["control_too_few"]], {4: "four"}[_pub["control_comparable"]], {6: "six"}[_pub["control_files"]]) if _pub["control_too_few_images_with_pairs"] == [1] else "MISMATCH"),
 ("[supp] 33-25-S pending", "The other %s files, all of entries that mention FoundationPose, order at least %.2f\\%% of their pairs correctly" % (
     {9: "nine"}[_pub["pending"]], _pub["pending_min_accuracy_pp"]) if _pub["pending_groups"] == ["FoundationPose_mentioned"] else "MISMATCH"),
 ("[supp] 33-25-S S1", "none of %d re-checked; the other %s order at least %.2f\\%%" % (_pub["rechecked"], {9: "nine"}[_pub["pending"]], _pub["pending_min_accuracy_pp"]) if _pub["reversed"] == 0 else "MISMATCH"),
 ("[supp] 33-25-S range", "between %.2f\\%% and %d\\%% in all 74 such files" % (_pub["rechecked_accuracy_pp"][0], round(_pub["rechecked_accuracy_pp"][1]))),
]
checks = [c for c in checks if not c[0].startswith("[drop]")]
if not SRC:
    for name, s in checks:
        print("----", name, "->", s)
    sys.exit(0)
for name, s in checks:
    ok = s in ((supp + " " + text) if name.startswith("[supp]") else text)
    bad += not ok
    print(("OK  " if ok else "MISS"), name, "->", s)
print("missing:", bad)
pend = [(os.path.basename(f), n + 1, l.strip()[:90]) for f in SRC + sorted(glob.glob(os.path.join(ROOT_M, "supplementary", "*.tex")))
        for n, l in enumerate(open(f)) if "\\pending{" in l]
for p in pend:
    print("PENDING", "%s:%d" % p[:2], p[2])
print("pending (33-23 placeholders):", len(pend))
