#!/usr/bin/env python3
"""Export a small, explicitly selected aggregate artifact from the paper snapshot."""

import argparse
import hashlib
import json
from pathlib import Path


def estimates(rows, metric):
    return {name: float(row["estimate_pp"][metric]) for name, row in rows.items()}


def comparisons(rows):
    return {name: {"estimate_pp": float(row["estimate_pp"]),
                   "ci95_pp": [float(x) for x in row["ci95_pp"]]}
            for name, row in rows.items()}


def extract(paper):
    source_hashes = {}

    def load(relative):
        path = paper / relative
        raw = path.read_bytes()
        source_hashes[relative] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    def study(name):
        return load(f"data33_17/{name}.json")

    ea, x2 = study("EA_RESULTS"), study("X2_RESULTS")
    pair = study("ED_MECH")
    derived = study("derived_v3")
    counts = {}
    for dataset, source in (("xyzibd", "Y1_RESULTS"), ("ipd", "EB_RESULTS")):
        detection = study(source)
        brief = derived["counts"][dataset]
        counts[dataset] = {
            "ap_pp": estimates(detection["metrics"], "ap"),
            "paired_ap_differences": comparisons(detection["comparisons"]),
            "frames": brief["frames"],
            "truth_instances": brief["truth_instances"],
            "median_detector_to_gt_ratio": brief["median_ratio"],
            "share_frames_below_one": brief["share_below_1"],
            "outputs": {key: brief[key] for key in ("B0", "B1", "B3", "B4")},
        }
        if (detection["metrics"]["B1"]["source"]["sha256"] != detection["metrics"]["B2"]["source"]["sha256"]
                or detection["details"]["output_counts"]["B1"] != detection["details"]["output_counts"]["B2"]):
            raise ValueError(f"Expected B1 and B2 to use the same predictions on {dataset}.")
        counts[dataset]["outputs"]["B2"] = dict(brief["B1"])

    sam3 = load("data33_18/S18_2D_V11.json")["cohorts"]["Cprime"]
    gate = load("data33_18/S18_GATE.json")
    return {
        "format_version": 1,
        "ranking_ar_pp": {
            dataset: estimates(ea["per_dataset"][dataset], "ar")
            | estimates(x2["per_dataset"][dataset], "ar")
            for dataset in ea["per_dataset"]
        },
        "ranking_dataset_mean_ar_pp": estimates(ea["average"], "ar") | estimates(x2["average"], "ar"),
        "ranking_paired_differences": {
            name: comparisons(by_dataset)
            for name, by_dataset in (ea["comparisons"] | x2["comparisons"]).items()
        },
        "cross_call_pairs": pair["cross_mask_by_dataset"],
        "within_call_pairs": pair["within_call_by_dataset"],
        "cross_call_pooled": pair["cross_mask_combined"],
        "counting": counts,
        "coverage_gain_by_density_posthoc": derived["density_A4_basis_posthoc"],
        "candidate_slots_k8": derived["slots"],
        "sam3_2d_gate": {
            "selected_variant": gate["selected"],
            "variants": {
                key: {"identity_correct_coverage": row["identity_correct_cover"],
                      "class_agnostic_coverage": row["class_agnostic_cover"],
                      "count_ratio_median": row["count_ratio_median"],
                      "false_detections_per_frame": row["false_detections_per_frame_mean"],
                      "mean_detected_identities": row["detected_object_kinds_mean"],
                      "mask_ap": row["segmentation_COCO"]["AP"]}
                for key, row in sam3.items()
            },
        },
        "source_sha256": source_hashes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = extract(args.paper)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
