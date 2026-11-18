"""Table of the 33-18 SAM 3 gate (XYZ-IBD validation frames C') from data33_18/S18_2D_V11.json."""
import json, os
D = os.path.join(os.path.dirname(__file__), '..', 'data33_18')
m = json.load(open(os.path.join(D, 'S18_2D_V11.json')))['cohorts']['Cprime']
sel = json.load(open(os.path.join(D, 'S18_GATE.json')))['selected']
rows = [('V0', 'NOCTIS'),
        ('V1', '+ SAM~3 text proposals'),
        ('V2a03', '+ exemplar propagation, 1 box, $\\tau_e=0.3$'),
        ('V2a05', '+ exemplar propagation, 1 box, $\\tau_e=0.5$'),
        ('V2b03', '+ exemplar propagation, $\\le 3$ boxes, $\\tau_e=0.3$'),
        ('V2b05', '+ exemplar propagation, $\\le 3$ boxes, $\\tau_e=0.5$')]
out = ['\\setlength{\\tabcolsep}{2.5pt}', '\\begin{tabular}{lrrrrrr}', '\\toprule',
       '2D source & \\multicolumn{2}{c}{Coverage (\\%)} & Count ratio & False det. & Identities & Mask AP \\\\',
       '\\cmidrule(lr){2-3}',
       ' & identity & any & (median) & per frame & per frame & (\\%) \\\\', '\\midrule']
for k, name in rows:
    r = m[k]
    mark = '\\textsuperscript{\\dag}' if k == sel else ''
    out.append(f"{name}{mark} & {100*r['identity_correct_cover']:.1f} & {100*r['class_agnostic_cover']:.1f} & "
               f"{r['count_ratio_median']:.2f} & {r['false_detections_per_frame_mean']:.1f} & "
               f"{r['detected_object_kinds_mean']:.2f} & {100*r['segmentation_COCO']['AP']:.1f} \\\\")
out += ['\\bottomrule', '\\end{tabular}']
os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'tables'), exist_ok=True)
open(os.path.join(os.path.dirname(__file__), '..', 'tables', 'supp_sam3.tex'), 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
