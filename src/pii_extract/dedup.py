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

3. Different labels, any overlap  ->  CONFLICT. The best candidate by the
   priority order below wins the LABEL. The loser is not simply discarded:
   any of its characters that no winner covers are absorbed into the winner
   next to them, so the winner grows to the union. A span cannot carry two
   labels, but every character either candidate flagged stays covered, which
   is what keeps redaction fail-closed.

   When a loser's leftover characters lie between two winners, they go to the
   higher-priority one. Losers are absorbed best-first, each against the
   winners as already grown, so the output never overlaps.

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

What can be discarded
---------------------
Rules 2 and 3 never drop a flagged character. Rule 1 does, on purpose: when
two detections of one entity nearly coincide, the weaker one's extra
characters (a trailing comma, a neighbouring word) are treated as boundary
noise. ``iou_threshold`` sets how much agreement counts as "the same entity".
At 1.0 only exact duplicates qualify, everything else is unioned, and the
output covers every character any candidate flagged.

Cost
----
Only spans that overlap can interact, so candidates are first grouped into
clusters of transitively overlapping spans (one sort and one sweep). The
pairwise rules then run inside each cluster, which holds a handful of spans
regardless of document length. Overall cost is O(n log n).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator
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


def _overlap_clusters(candidates: Iterable[Candidate]) -> Iterator[list[Candidate]]:
    """Group candidates into maximal runs of transitively overlapping spans."""
    cluster: list[Candidate] = []
    reach = -1  # furthest end offset seen in the current cluster
    for candidate in sorted(candidates, key=lambda c: (c.span.start, c.span.end)):
        if cluster and candidate.span.start >= reach:
            yield cluster
            cluster = []
        cluster.append(candidate)
        reach = max(reach, candidate.span.end)
    if cluster:
        yield cluster


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
            # A union counts as seen whole if any of its parts was.
            merged[-1] = Candidate(union, clipped=previous.clipped and candidate.clipped)
        else:
            merged.append(candidate)
    return merged


def _uncovered(start: int, end: int, extents: list[list[int]]) -> list[tuple[int, int]]:
    """The parts of ``[start, end)`` that no extent covers, left to right."""
    gaps: list[tuple[int, int]] = []
    cursor = start
    for extent_start, extent_end in sorted(extents):
        if extent_end <= cursor:
            continue
        if extent_start >= end:
            break
        if extent_start > cursor:
            gaps.append((cursor, extent_start))
        cursor = max(cursor, extent_end)
    if cursor < end:
        gaps.append((cursor, end))
    return gaps


def _resolve_label_conflicts(candidates: list[Candidate], text: str) -> list[Candidate]:
    """Rule 3, across labels: the best span wins the label and absorbs the rest."""
    winners: list[Candidate] = []
    losers: list[Candidate] = []
    for candidate in sorted(candidates, key=_priority):
        if any(candidate.span.overlaps(w.span) for w in winners):
            losers.append(candidate)
        else:
            winners.append(candidate)
    if not losers:
        return winners

    # Winners are in priority order, so a lower index means a stronger claim.
    extents = [[w.span.start, w.span.end] for w in winners]
    for loser in losers:
        for gap_start, gap_end in _uncovered(loser.span.start, loser.span.end, extents):
            # A gap is bounded by the loser's own edge or by a winner. The
            # loser overlaps at least one winner, so at least one side of
            # every gap is a winner; with two, the stronger one takes it.
            left = next((i for i, e in enumerate(extents) if e[1] == gap_start), None)
            right = next((i for i, e in enumerate(extents) if e[0] == gap_end), None)
            if right is None or (left is not None and left < right):
                extents[left][1] = gap_end
            else:
                extents[right][0] = gap_start

    resolved: list[Candidate] = []
    for winner, (start, end) in zip(winners, extents, strict=True):
        if (start, end) != (winner.span.start, winner.span.end):
            grown = winner.span.model_copy(
                update={"start": start, "end": end, "text": text[start:end]}
            )
            winner = Candidate(grown, clipped=winner.clipped)
        resolved.append(winner)
    return resolved


def merge_spans(
    candidates: Iterable[Candidate],
    text: str,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> list[Span]:
    """Collapse per-window candidates into a flat, sorted, non-overlapping list.

    ``text`` is the original input; it is needed to read the text of a union
    or grown span (rules 2 and 3) straight from the source rather than
    stitching fragments together.
    """
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")

    by_label: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_label[candidate.span.label].append(candidate)

    per_label: list[Candidate] = []
    for group in by_label.values():
        for cluster in _overlap_clusters(group):
            survivors = _suppress_duplicates(cluster, iou_threshold)
            per_label.extend(_union_overlapping(survivors, text))

    winners: list[Candidate] = []
    for cluster in _overlap_clusters(per_label):
        winners.extend(_resolve_label_conflicts(cluster, text))
    return sorted((c.span for c in winners), key=lambda s: (s.start, s.end, s.label))
