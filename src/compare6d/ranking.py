"""Scoring and pair diagnostics on candidates that were already registered."""

from dataclasses import dataclass
from itertools import combinations
from math import exp, fsum, isfinite
from statistics import fmean, pstdev
from typing import Callable, Iterable, Sequence


def relative_scores(scores: Sequence[float], winner: int = 0) -> dict[str, float]:
    """Score the same nominated pose by its call's z-score, softmax and margin.

    The population standard deviation and zero-spread convention follow the
    relative-score control in the paper. A nominated pose may lose under a
    second scorer, in which case the margin is negative.
    """
    if len(scores) < 2 or not 0 <= winner < len(scores) or not all(map(isfinite, scores)):
        raise ValueError("Relative scores need two or more finite values and a valid winner index.")
    value = scores[winner]
    spread = pstdev(scores)
    maximum = max(scores)
    return {
        "z_score": (value - fmean(scores)) / spread if spread else 0.0,
        "softmax": exp(value - maximum) / fsum(exp(x - maximum) for x in scores),
        "margin": value - max(x for index, x in enumerate(scores) if index != winner),
    }


def best_index(scores: Sequence[float]) -> int:
    """Return the first maximum in the original hypothesis order."""
    if not len(scores) or not all(map(isfinite, scores)):
        raise ValueError("A scoring call needs at least one finite hypothesis score.")
    return max(range(len(scores)), key=scores.__getitem__)


def correct_from_mssd(mssd: float, diameter: float) -> bool:
    """Apply the paper's strict correctness threshold in a common length unit."""
    if not isfinite(mssd) or not isfinite(diameter) or mssd < 0 or diameter <= 0:
        raise ValueError("Expected finite non-negative MSSD and a positive diameter.")
    return mssd < 0.1 * diameter


def score_jointly(
    pose_sets: Sequence[Sequence[object]],
    call_scores: Sequence[Sequence[float]],
    scorer: Callable[[Sequence[object]], Iterable[float]],
) -> tuple[float, ...]:
    """Re-score one fixed winner per mask in a single call for one target.

    Pose objects are passed through unchanged, allowing the caller to wrap a
    scorer's expected pose representation. Supply one image-object target.
    """
    if not len(pose_sets) or len(pose_sets) != len(call_scores):
        raise ValueError("Provide aligned pose and score sets for one target.")
    winners = []
    for poses, scores in zip(pose_sets, call_scores):
        if len(poses) != len(scores):
            raise ValueError("Each call must have one score per pose.")
        winners.append(poses[best_index(scores)])
    joint = tuple(scorer(winners))
    if len(joint) != len(winners) or not all(map(isfinite, joint)):
        raise ValueError("The joint scorer must return one finite score per fixed winner.")
    return joint


@dataclass(frozen=True)
class Pairwise:
    wins: int = 0
    ties: int = 0
    pairs: int = 0

    @property
    def accuracy(self) -> float | None:
        return self.wins / self.pairs if self.pairs else None

    @property
    def accuracy_half_credit(self) -> float | None:
        return (self.wins + 0.5 * self.ties) / self.pairs if self.pairs else None

    def __add__(self, other: "Pairwise") -> "Pairwise":
        return Pairwise(self.wins + other.wins, self.ties + other.ties, self.pairs + other.pairs)


def pairwise(scores: Sequence[float], correct: Sequence[bool]) -> Pairwise:
    """Count opposite-label pairs in one call or among winners of one target."""
    if len(scores) != len(correct) or not all(map(isfinite, scores)):
        raise ValueError("Align correctness labels with finite scores.")
    if not all(isinstance(value, bool) for value in correct):
        raise ValueError("Correctness labels must be booleans.")
    wins = ties = pairs = 0
    for i, j in combinations(range(len(scores)), 2):
        if correct[i] == correct[j]:
            continue
        positive, negative = (scores[i], scores[j]) if correct[i] else (scores[j], scores[i])
        pairs += 1
        wins += positive > negative
        ties += positive == negative
    return Pairwise(wins, ties, pairs)
