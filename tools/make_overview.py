#!/usr/bin/env python3
"""Render the README overview from the paper's aggregate results."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    pooled = data["cross_call_pooled"]
    within = data["within_call_pairs"].values()
    within_accuracy = sum(x["correct_first"] for x in within) / sum(x["opposite_label_pairs"] for x in within)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#526274", "text.color": "#202e41",
                         "axes.labelcolor": "#202e41", "xtick.color": "#202e41",
                         "ytick.color": "#202e41"})
    fig, (ranking, counting) = plt.subplots(1, 2, figsize=(10.5, 2.7), layout="constrained")
    blue, teal, orange, grey = "#284b73", "#197b76", "#bb5d25", "#e2e8ef"
    accuracy = [100 * v for v in (within_accuracy, pooled["raw"]["accuracy"], pooled["joint"]["accuracy"])]
    ranking.barh(range(3), accuracy, height=0.52, color=[blue, orange, teal])
    ranking.invert_yaxis()
    ranking.set_yticks(range(3), ["Within call", "Across calls", "Joint re-score"])
    ranking.set_xlim(0, 113)
    ranking.set_xticks(range(0, 101, 25))
    ranking.set_xlabel("Pairwise accuracy (%)")
    ranking.set_title("How poses are scored", loc="left", weight="bold")
    for index, value in enumerate(accuracy):
        ranking.text(value + 1.4, index, f"{value:.2f}", va="center", weight="bold")

    for index, ds in enumerate(("xyzibd", "ipd")):
        contrast = data["counting"][ds]["paired_ap_differences"]["B4-B0"]
        point, (low, high) = contrast["estimate_pp"], contrast["ci95_pp"]
        color = teal if point > 0 else orange
        counting.errorbar(point, index, xerr=[[point - low], [high - point]],
                          fmt="o", capsize=4, color=color, markersize=7, linewidth=2.2)
        counting.text(point, index - 0.19, f"{point:+.2f}", va="center", ha="center", weight="bold")
    counting.axvline(0, color="#526274", linestyle="--", linewidth=1)
    counting.set_yticks(range(2), ["XYZ-IBD", "IPD"])
    counting.set_ylim(1.5, -0.5)
    counting.set_xlim(-11, 12)
    counting.set_xlabel("Change in AP (points, 95% CI)")
    counting.set_title("Adding candidates and pose-assisted counts", loc="left", weight="bold")
    for axis in (ranking, counting):
        axis.grid(axis="x", color=grey, linewidth=0.65)
        axis.set_axisbelow(True)
        axis.tick_params(axis="y", length=0)
    fig.savefig(args.output, dpi=180, facecolor="white", metadata={"Software": "Compare6D"})


if __name__ == "__main__":
    main()
