"""Merging detections from overlapping windows into one flat span list.

This is the only place in the package where spans are deduplicated or where
overlaps are resolved. Input candidates are already in original-text offsets.
The output is sorted by position and contains no two overlapping spans, which
is exactly what redaction needs.

Rules, applied in this order
----------------------------
1. Same label, IoU >= ``iou_threshold``  ->  DUPLICATES of one entity.
   Keep the single best candidate and its boundaries; discard the rest.
   "Best" is the priority order below.

2. Same label, overlapping, IoU < ``iou_threshold``  ->  BOUNDARY DISAGREEMENT.
   The windows agree these characters are PII of this type but disagree on
   where the entity ends. Merge them into their union (score = max). For a
   redactor this is the fail-closed choice: no character that any window
   flagged under this label is left exposed.

3. Different labels, any overlap  ->  CONFLICT. Winner takes all: the best
   candidate by the priority order is kept and every overlapping candidate
   with a different label is dropped.

Priority order (highest first)
------------------------------
  a. not clipped  - a span touching a window's cut edge may be a truncated
                    fragment ("4111 1111" of a full card number), so it loses
                    to any overlapping span that was seen whole;
  b. higher score;
  c. longer span;
  d. earlier start;
  e. label, alphabetically (a pure tie-break so results are deterministic).

The ordering is total, so the result does not depend on input order.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from pii_extract.spans import Span, iou

DEFAULT_IOU_THRESHOLD = 0.5


@dataclass(frozen=True, slots=True)
class Candidate:
    """A span plus what the merge stage needs to know about where it came from."""

    span: Span
    clipped: bool = False


def _priority(candidate: Candidate) -> tuple:
    span = candidate.span
    return (candidate.clipped, -span.score, -span.length, span.start, span.label)


def _suppress_duplicates(candidates: list[Candidate], iou_threshold: float) -> list[Candidate]:
    """Rule 1, within a single label: greedy non-maximum suppression."""
    kept: list[Candidate] = []
    for candidate in sorted(candidates, key=_priority):
        if all(iou(candidate.span, k.span) < iou_threshold for k in kept):
            kept.append(candidate)
    return kept


def _union_overlapping(candidates: list[Candidate], text: str) -> list[Candidate]:
    """Rule 2, within a single label: merge any remaining overlaps into unions."""
    merged: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda c: (c.span.start, c.span.end)):
        if merged and candidate.span.start < merged[-1].span.end:
            previous = merged[-1]
            start = previous.span.start
            end = max(previous.span.end, candidate.span.end)
            union = Span(
                start=start,
                end=end,
                label=previous.span.label,
                score=max(previous.span.score, candidate.span.score),
                text=text[start:end],
            )
            # The union is only as trustworthy as its most trustworthy part.
            merged[-1] = Candidate(union, clipped=previous.clipped and candidate.clipped)
        else:
            merged.append(candidate)
    return merged


def _resolve_label_conflicts(candidates: list[Candidate]) -> list[Candidate]:
    """Rule 3, across labels: greedy winner-takes-all on any overlap."""
    kept: list[Candidate] = []
    for candidate in sorted(candidates, key=_priority):
        if not any(candidate.span.overlaps(k.span) for k in kept):
            kept.append(candidate)
    return kept


def merge_spans(
    candidates: Iterable[Candidate],
    text: str,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> list[Span]:
    """Collapse per-window candidates into a flat, sorted, non-overlapping list.

    ``text`` is the original input; it is needed to read the text of a union
    span (rule 2) straight from the source rather than stitching fragments.
    """
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")

    by_label: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_label[candidate.span.label].append(candidate)

    per_label: list[Candidate] = []
    for group in by_label.values():
        survivors = _suppress_duplicates(group, iou_threshold)
        per_label.extend(_union_overlapping(survivors, text))

    winners = _resolve_label_conflicts(per_label)
    return sorted((c.span for c in winners), key=lambda s: (s.start, s.end, s.label))
