"""Reference selection and count rules for the paper's cached detections."""

from dataclasses import dataclass
from math import dist, isfinite
from typing import Iterable, Mapping


@dataclass(frozen=True)
class Candidate:
    object_id: int
    mask_id: str
    translation_m: tuple[float, float, float]
    joint_score: float
    depth_confidence: float
    detector_confidence: float | None = None


def translation_nms(rows: Iterable[Candidate], min_distance_m: float) -> list[Candidate]:
    """Greedy suppression in input order, suppressing distances below the radius."""
    if not isfinite(min_distance_m) or min_distance_m < 0:
        raise ValueError("The NMS radius must be finite and non-negative.")
    kept = []
    for row in rows:
        if len(row.translation_m) != 3 or not all(map(isfinite, row.translation_m)):
            raise ValueError("Each translation needs three finite coordinates in metres.")
        if all(dist(row.translation_m, other.translation_m) >= min_distance_m for other in kept):
            kept.append(row)
    return kept


def select_detections(
    candidates: Iterable[Candidate],
    detector_counts: Mapping[int, int],
    diameters_m: Mapping[int, float],
    variant: str,
) -> list[Candidate]:
    """Apply B0–B4 to admitted masks and cached best poses from one image.

    Each admitted detection has a detector confidence. Supplementary masks
    use None. Supply joint scores from the matching pool for each variant.
    Input order breaks score ties within an object.
    """
    if variant not in {"B0", "B1", "B2", "B3", "B4"}:
        raise ValueError("Unknown detection variant.")
    if any(type(count) is not int or count < 0 for count in detector_counts.values()):
        raise ValueError("Detector counts must be non-negative integers.")
    grouped = {object_id: [] for object_id, count in detector_counts.items() if count > 0}
    seen = set()
    for row in candidates:
        if row.object_id not in grouped:
            continue
        key = (row.object_id, row.mask_id)
        if key in seen:
            raise ValueError(f"Duplicate mask {key} in one image.")
        seen.add(key)
        if row.detector_confidence is not None and not isfinite(row.detector_confidence):
            raise ValueError("Detector confidence must be finite.")
        if variant in {"B3", "B4"} or row.detector_confidence is not None:
            grouped[row.object_id].append(row)

    if variant == "B0":
        rows = [row for target in grouped.values() for row in target]
        return sorted(rows, key=lambda row: (-row.detector_confidence, row.object_id))

    selected = []
    for object_id, rows in grouped.items():
        if not rows:
            continue
        diameter = diameters_m.get(object_id)
        if diameter is None or not isfinite(diameter) or diameter <= 0:
            raise ValueError(f"Missing positive diameter in metres for object {object_id}.")
        if any(not isfinite(row.joint_score) or not isfinite(row.depth_confidence) for row in rows):
            raise ValueError("Joint scores and depth confidence must be finite.")
        rank = sorted(rows, key=lambda row: -row.joint_score)
        survivors = translation_nms(rank, 0.15 * diameter)
        detector_count = detector_counts[object_id]
        quota = max(detector_count, len(survivors)) if variant in {"B2", "B4"} else detector_count
        by_confidence = sorted(survivors[:quota], key=lambda row: -row.depth_confidence)
        selected.extend(by_confidence[:2 * detector_count + 8])
    return sorted(selected, key=lambda row: (-row.depth_confidence, row.object_id))[:100]
