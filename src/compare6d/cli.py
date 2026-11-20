"""CPU commands for the cached-pose diagnostic and manuscript aggregates."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .check import check, load_bop_errors, load_table
from .ranking import Pairwise, best_index, pairwise, relative_scores
from .replay import replay


def audit(document):
    if not document["targets"]:
        raise ValueError("Expected at least one target.")
    all_calls = [call for target in document["targets"] for call in target["calls"]]
    joint_available = all("joint_score" in call for call in all_calls)
    if not joint_available and any("joint_score" in call for call in all_calls):
        raise ValueError("Supply joint scores for all calls or omit them for all calls.")
    methods = ["per_call", "z_score", "softmax", "margin"]
    if joint_available:
        methods.append("joint")
    counts = {name: Pairwise() for name in methods}
    within = Pairwise()
    details = []
    targets = set()
    for target in document["targets"]:
        key = (target["image_id"], target["object_id"])
        if key in targets:
            raise ValueError(f"Duplicate target: {key}")
        targets.add(key)
        calls = target["calls"]
        mask_ids = [call["mask_id"] for call in calls]
        if not calls or len(mask_ids) != len(set(mask_ids)):
            raise ValueError(f"Each target needs calls with unique mask IDs: {key}")
        winners, labels = [], []
        for call in calls:
            scores, correct = call["scores"], call["correct"]
            within += pairwise(scores, correct)
            winner = best_index(scores)
            scores_for_winner = {"per_call": scores[winner], **relative_scores(scores, winner)}
            if joint_available:
                scores_for_winner["joint"] = call["joint_score"]
            winners.append(scores_for_winner)
            labels.append(correct[winner])
        top = {}
        for name in methods:
            values = [winner[name] for winner in winners]
            counts[name] += pairwise(values, labels)
            chosen = best_index(values)
            top[name] = {"mask_id": mask_ids[chosen], "correct": labels[chosen]}
        details.append({"image_id": key[0], "object_id": key[1], "top": top})

    def summary(row):
        return {**asdict(row), "accuracy": row.accuracy,
                "accuracy_half_credit": row.accuracy_half_credit}

    return {"targets": details, "cross_call": {name: summary(row) for name, row in counts.items()},
            "within_call_per_call_score": summary(within)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit_parser = commands.add_parser("audit", help="Compare scores on already registered hypotheses.")
    audit_parser.add_argument("input", type=Path)
    replay_parser = commands.add_parser("replay", help="Export summaries from archived paper aggregates.")
    replay_parser.add_argument("input", type=Path)
    replay_parser.add_argument("--out", required=True, type=Path)
    check_parser = commands.add_parser(
        "check", help="Cross-call pairwise accuracy of one score per output pose.")
    source = check_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--table", type=Path, help="CSV with scene_id, im_id, obj_id, score, correct.")
    source.add_argument("--bop-errors", type=Path, help="Folder of errors_<scene>.json files (BOP toolkit, MSSD).")
    check_parser.add_argument("--models-info", type=Path, help="models_info.json of the dataset; needed with --bop-errors.")
    check_parser.add_argument("--threshold", type=float, default=0.1, help="Correct if MSSD < threshold x diameter.")
    check_parser.add_argument("--draws", type=int, default=2000)
    check_parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            if args.bop_errors and not args.models_info:
                raise ValueError("--bop-errors needs --models-info.")
            rows = load_table(args.table) if args.table else load_bop_errors(
                args.bop_errors, args.models_info, args.threshold)
            print(json.dumps(check(rows, args.draws, args.seed), indent=2, allow_nan=False))
            return 0
        data = json.loads(args.input.read_text(encoding="utf-8"))
        if args.command == "audit":
            print(json.dumps(audit(data), indent=2, allow_nan=False))
        else:
            replay(data, args.out)
            print(f"Paper tables written to {args.out}")
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"compare6d: {error}", file=sys.stderr)
        return 1
    return 0
