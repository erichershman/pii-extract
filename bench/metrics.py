"""Span-level scoring. Pure functions: no models, no datasets."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

GoldSpan = tuple[int, int, str]  # (start, end, label) - the canonical span key


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def support(self) -> int:
        return self.tp + self.fn

    def add(self, other: Counts) -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn


def score_exact(gold: Iterable[GoldSpan], predicted: Iterable[GoldSpan]) -> dict[str, Counts]:
    """Strict matching: a prediction counts only if start, end and label all agree."""
    gold_set, predicted_set = set(gold), set(predicted)
    counts: dict[str, Counts] = defaultdict(Counts)
    for span in predicted_set:
        if span in gold_set:
            counts[span[2]].tp += 1
        else:
            counts[span[2]].fp += 1
    for span in gold_set - predicted_set:
        counts[span[2]].fn += 1
    return counts


def _overlaps(a: GoldSpan, b: GoldSpan) -> bool:
    return a[2] == b[2] and a[0] < b[1] and b[0] < a[1]


def score_overlap(gold: Sequence[GoldSpan], predicted: Sequence[GoldSpan]) -> dict[str, Counts]:
    """Relaxed matching: same label and at least one shared character.

    Precision is over predictions (a prediction is right if it touches a gold
    span); recall is over gold spans (a gold span is found if any prediction
    touches it). This tolerates boundary-convention differences such as
    whether a phone extension belongs to the number.
    """
    gold, predicted = sorted(set(gold)), sorted(set(predicted))
    counts: dict[str, Counts] = defaultdict(Counts)
    for p in predicted:
        if any(_overlaps(p, g) for g in gold):
            counts[p[2]].tp += 1
        else:
            counts[p[2]].fp += 1
    for g in gold:
        if not any(_overlaps(p, g) for p in predicted):
            counts[g[2]].fn += 1
    return counts


def micro_average(per_label: dict[str, Counts]) -> Counts:
    total = Counts()
    for counts in per_label.values():
        total.add(counts)
    return total


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile, q in [0, 100]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, -(-len(ordered) * q // 100))  # ceil without importing math
    return ordered[int(rank) - 1]
