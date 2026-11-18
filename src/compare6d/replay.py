"""Check and export the manuscript's aggregate evidence."""

import csv
from math import isclose
from statistics import fmean

DATASETS = {"lmo": "LM-O", "tless": "T-LESS", "tudl": "TUD-L", "icbin": "IC-BIN",
            "ycbv": "YCB-V", "xyzibd": "XYZ-IBD", "ipd": "IPD"}
CLASSIC = ("lmo", "tless", "tudl", "icbin", "ycbv")
SCORES = ("raw", "detection", "c2", "megapose", "joint")
COUNT_ARMS = ("B0", "B1", "B2", "B3", "B4")


def ratio(wins, pairs):
    if not 0 <= wins <= pairs or pairs <= 0:
        raise ValueError("Invalid archived pair counts.")
    return wins / pairs


def validate(data):
    if data["format_version"] != 1:
        raise ValueError("Unsupported paper results version.")
    ranking = data["ranking_ar_pp"]
    for arm, mean in data["ranking_dataset_mean_ar_pp"].items():
        if not isclose(mean, fmean(ranking[ds][arm] for ds in CLASSIC), abs_tol=1e-7):
            raise ValueError(f"Archived dataset mean differs from its components: {arm}")
    cross = data["cross_call_pairs"]
    for score in SCORES:
        parts = [cross[dataset][score] for dataset in CLASSIC]
        for row in parts:
            if not isclose(row["accuracy"], ratio(row["correct_first"], row["opposite_label_pairs"])):
                raise ValueError(f"Invalid per-dataset accuracy: {score}")
        wins = sum(row["correct_first"] for row in parts)
        pairs = sum(row["opposite_label_pairs"] for row in parts)
        pooled = data["cross_call_pooled"][score]
        if wins != pooled["correct_first"] or pairs != pooled["opposite_label_pairs"]:
            raise ValueError(f"Cannot pool the archived cross-call counts for {score}.")
        if not isclose(pooled["accuracy"], ratio(wins, pairs)):
            raise ValueError(f"Invalid pooled accuracy: {score}")
    for dataset in CLASSIC:
        row = data["within_call_pairs"][dataset]
        if not isclose(row["accuracy"], ratio(row["correct_first"], row["opposite_label_pairs"])):
            raise ValueError(f"Invalid within-call accuracy: {dataset}")
    for dataset, row in data["counting"].items():
        for arm, outputs in row["outputs"].items():
            if sum(outputs[key] for key in ("TP", "identity", "pose", "duplicate", "background")) != outputs["outputs"]:
                raise ValueError(f"Archived output classes do not add up: {dataset}/{arm}")
        for key, difference in row["paired_ap_differences"].items():
            a, b = key.split("-")
            if not isclose(difference["estimate_pp"], row["ap_pp"][a] - row["ap_pp"][b], abs_tol=1e-7):
                raise ValueError(f"AP contrast does not match archived endpoints: {dataset}/{key}")


def put_csv(path, columns, rows):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def replay(data, output):
    validate(data)
    output.mkdir(parents=True, exist_ok=False)
    ranking = data["ranking_ar_pp"]
    means = data["ranking_dataset_mean_ar_pp"]
    put_csv(output / "ranking_ar.csv", ("arm", *[DATASETS[ds] for ds in CLASSIC], "mean_ar_pp"),
            ((arm, *[ranking[ds][arm] for ds in CLASSIC], means[arm]) for arm in means))

    put_csv(output / "ranking_contrasts.csv", ("contrast", "dataset", "delta_ar_pp", "ci_low", "ci_high"),
            ((name, DATASETS.get(ds, ds), row["estimate_pp"], *row["ci95_pp"])
             for name, group in data["ranking_paired_differences"].items() for ds, row in group.items()))

    cross = data["cross_call_pairs"] | {"pooled": data["cross_call_pooled"]}
    within = dict(data["within_call_pairs"])
    wins = sum(x["correct_first"] for x in within.values())
    pairs = sum(x["opposite_label_pairs"] for x in within.values())
    within["pooled"] = {"accuracy": ratio(wins, pairs), "opposite_label_pairs": pairs}
    put_csv(output / "pairwise.csv",
            ("dataset", "cross_call_pairs", *SCORES, "within_call_pairs", "within_call_accuracy"),
            ((DATASETS.get(ds, ds), cross[ds]["raw"]["opposite_label_pairs"],
              *[cross[ds][score]["accuracy"] for score in SCORES],
              within[ds]["opposite_label_pairs"], within[ds]["accuracy"])
             for ds in (*CLASSIC, "pooled")))

    counting = data["counting"]
    put_csv(output / "counting_ap.csv", ("dataset", "arm", "ap_pp", "outputs", "outputs_per_instance"),
            ((DATASETS[ds], arm, row["ap_pp"][arm], row["outputs"][arm]["outputs"],
              row["outputs"][arm]["outputs"] / row["truth_instances"])
             for ds, row in counting.items() for arm in COUNT_ARMS))
    put_csv(output / "counting_contrasts.csv", ("dataset", "contrast", "delta_ap_pp", "ci_low", "ci_high"),
            ((DATASETS[ds], key, contrast["estimate_pp"], *contrast["ci95_pp"])
             for ds, row in counting.items() for key, contrast in row["paired_ap_differences"].items()))

    density = data["coverage_gain_by_density_posthoc"]
    put_csv(output / "coverage_density_posthoc.csv",
            ("max_instances_of_one_object", "frames", "per_call_gain_pp", "joint_gain_pp",
             "per_call_ci_low", "per_call_ci_high", "joint_ci_low", "joint_ci_high"),
            ((group, density["frames"][i], density["A1_mean"][i], density["A4_mean"][i],
              *density["A1_ci"][i], *density["A4_ci"][i]) for i, group in enumerate(density["bins"])))

    slots = data["candidate_slots_k8"]
    put_csv(output / "candidate_slots_k8.csv",
            ("dataset", "source", "gain_pp", "recovered_per_100_registrations", "recall_base", "recall_k8"),
            ((DATASETS[ds], arm, row["gain_k8"], row["recovered_per_100_registrations"],
              row["recall_base"], row["recall_k8"])
             for ds, rows in slots.items() for arm, row in rows.items()))

    sam3_columns = ("identity_correct_coverage", "class_agnostic_coverage", "count_ratio_median",
                    "false_detections_per_frame", "mean_detected_identities", "mask_ap")
    gate = data["sam3_2d_gate"]
    put_csv(output / "sam3_2d_gate.csv", ("arm", "selected", *sam3_columns),
            ((key, key == gate["selected_variant"], *[row[col] for col in sam3_columns])
             for key, row in gate["variants"].items()))

    pooled = cross["pooled"]
    text = [
        "# Results from the paper snapshot", "",
        f"- Cross-call accuracy over {pooled['raw']['opposite_label_pairs']:,} pairs was "
        f"{pooled['raw']['accuracy']:.2%} with per-call scores and {pooled['joint']['accuracy']:.2%} with joint scores.",
        f"- Within-call accuracy was {within['pooled']['accuracy']:.2%} over {pairs:,} pairs.",
        f"- Equal-weight mean AR was {means['A1']:.2f} with per-call scores and {means['A4']:.2f} with joint scores.",
    ]
    for ds in ("xyzibd", "ipd"):
        contrast = counting[ds]["paired_ap_differences"]["B4-B0"]
        lo, hi = contrast["ci95_pp"]
        text.append(f"- On {DATASETS[ds]}, B4 − B0 was {contrast['estimate_pp']:+.2f} AP points "
                    f"(95% CI {lo:+.2f} to {hi:+.2f}).")
    text.extend(["", "BOP metrics and uncertainty intervals come from the archived evaluation.",
                 "The replay checks arithmetic and exports the recorded results.", ""])
    with (output / "SUMMARY.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(text))
