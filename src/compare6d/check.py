"""Cross-call pairwise accuracy from one score per output pose.

Each row is the best pose of one scoring call (one mask): an image key, an
object identifier, the score that the pipeline would compare or submit, and
whether the pose is correct. A pair is two rows of the same image and object
with opposite labels. The accuracy is the share of pairs in which the correct
pose has the strictly higher score.
"""

import csv
import json
from math import isfinite
from pathlib import Path
import random
import re

from .ranking import Pairwise, pairwise

TRUE, FALSE = {"1", "true", "t", "yes"}, {"0", "false", "f", "no"}


def load_table(path):
    """Read rows from a CSV with columns scene_id, im_id, obj_id, score, correct.

    An optional dataset column keeps the images of several datasets apart.
    """
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        needed = {"scene_id", "im_id", "obj_id", "score", "correct"}
        if not reader.fieldnames or not needed <= set(reader.fieldnames):
            raise ValueError(f"The table needs the columns {sorted(needed)}.")
        for line in reader:
            label = line["correct"].strip().lower()
            if label not in TRUE | FALSE:
                raise ValueError(f"Unreadable correctness label: {line['correct']!r}")
            image = (line.get("dataset", ""), line["scene_id"].strip(), line["im_id"].strip())
            rows.append((image, line["obj_id"].strip(), float(line["score"]), label in TRUE))
    return rows


def load_bop_errors(errors_dir, models_info, threshold=0.1):
    """Read rows from the MSSD error files of the BOP toolkit.

    errors_dir holds the errors_<scene>.json files that eval_calc_errors.py
    writes for the error type mssd; models_info is the models_info.json of the
    dataset, which gives the object diameters. A pose is correct if its MSSD to
    any annotated instance of its object is strictly below threshold times the
    diameter, in the same length unit.
    """
    diameters = {str(key): value["diameter"] for key, value in
                 json.loads(Path(models_info).read_text(encoding="utf-8")).items()}
    files = sorted(Path(errors_dir).glob("errors_*.json"))
    if not files:
        raise ValueError(f"No errors_*.json files in {errors_dir}.")
    rows = []
    for file in files:
        scene = re.fullmatch(r"errors_(\d+)", file.stem)
        if not scene:
            raise ValueError(f"Unexpected file name: {file.name}")
        for record in json.loads(file.read_text(encoding="utf-8")):
            errors = [value[0] if isinstance(value, list) else value for value in record["errors"].values()]
            diameter = diameters[str(record["obj_id"])]
            correct = any(0 <= error < threshold * diameter for error in errors)
            rows.append((("", scene.group(1), str(record["im_id"])), str(record["obj_id"]),
                         float(record["score"]), correct))
    return rows


def check(rows, draws=2000, seed=0):
    """Pool opposite-label pairs over targets and resample images for the interval."""
    targets = {}
    for image, obj, score, correct in rows:
        if not isfinite(score):
            raise ValueError("Scores must be finite.")
        scores, labels = targets.setdefault((image, obj), ([], []))
        scores.append(score)
        labels.append(bool(correct))
    per_image = {}
    for (image, _), (scores, labels) in targets.items():
        per_image[image] = per_image.get(image, Pairwise()) + pairwise(scores, labels)
    total = sum(per_image.values(), Pairwise())
    if not total.pairs:
        raise ValueError("No image and object has both a correct and an incorrect pose, so there is nothing to compare.")
    counted = [value for value in per_image.values() if value.pairs]
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        wins = pairs = 0
        for value in rng.choices(counted, k=len(counted)):
            wins += value.wins
            pairs += value.pairs
        samples.append(wins / pairs)
    samples.sort()
    low, high = samples[int(0.025 * draws)], samples[min(draws - 1, int(0.975 * draws))]
    if high < 0.5:
        reading = "reversed"
        advice = ("Across calls these scores order pairs worse than chance. Do not compare or submit them as they are. "
                  "Score the candidates of a target in one call, use a score that rates each pose on its own, "
                  "or use the reversed score after checking its sign on each new detector and dataset.")
    elif low > 0.5:
        reading = "comparable"
        advice = "Across calls these scores point the right way on these data. The accuracy says how reliably."
    else:
        reading = "not_comparable"
        advice = ("The interval includes 50%. These scores cannot be compared across calls on this evidence, "
                  "and normalizing each call is no substitute. Use more labelled images or another score.")
    return {
        "pairs": total.pairs, "wins": total.wins, "ties": total.ties,
        "accuracy": total.accuracy, "accuracy_half_credit": total.accuracy_half_credit,
        "interval_95": [low, high], "images_with_pairs": len(counted), "images": len(per_image),
        "bootstrap": {"unit": "image", "draws": draws, "seed": seed},
        "reading": reading, "advice": advice,
    }
