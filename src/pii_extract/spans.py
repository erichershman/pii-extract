"""The canonical span schema.

Every stage of the pipeline (backends, windowing, dedup, redaction, API, CLI)
speaks this one type. Offsets are Python ``str`` indices, i.e. Unicode code
points, half-open ``[start, end)``, and always refer to the ORIGINAL input
text exactly as the caller supplied it. Nothing in this package normalizes,
strips, or re-encodes that text.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field, model_validator

SpanKey = tuple[int, int, str]


class OffsetError(ValueError):
    """A span's offsets do not address the text they claim to."""


class Span(BaseModel):
    """One detected entity: ``text == original[start:end]``."""

    model_config = ConfigDict(frozen=True)

    start: int = Field(ge=0, description="Inclusive code-point offset into the original text.")
    end: int = Field(gt=0, description="Exclusive code-point offset into the original text.")
    label: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    text: str = Field(description="Exactly original[start:end].")

    @model_validator(mode="after")
    def _check_shape(self) -> Span:
        if self.end <= self.start:
            raise ValueError(f"empty or inverted span: [{self.start}, {self.end})")
        if len(self.text) != self.end - self.start:
            raise ValueError(
                f"text length {len(self.text)} does not match span length {self.end - self.start}"
            )
        return self

    @property
    def key(self) -> SpanKey:
        """The one span-identity convention used everywhere: (start, end, label)."""
        return (self.start, self.end, self.label)

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: Span) -> bool:
        return self.start < other.end and other.start < self.end

    def shifted(self, delta: int) -> Span:
        """The same span in a coordinate system offset by ``delta`` characters."""
        return self.model_copy(update={"start": self.start + delta, "end": self.end + delta})


def span_from_offsets(source: str, start: int, end: int, label: str, score: float) -> Span:
    """Build a span whose ``text`` is read from ``source``, never trusted from a model.

    Raises :class:`OffsetError` if the offsets fall outside ``source``.
    """
    if not (0 <= start < end <= len(source)):
        raise OffsetError(f"span [{start}, {end}) is outside text of length {len(source)}")
    score = min(1.0, max(0.0, float(score)))
    return Span(start=start, end=end, label=label, score=score, text=source[start:end])


def verify_spans(source: str, spans: Iterable[Span]) -> None:
    """Assert the core invariant: every span's text is the slice it points at."""
    for span in spans:
        if span.end > len(source) or source[span.start : span.end] != span.text:
            raise OffsetError(
                f"span {span.key} carries text {span.text!r} but the source has "
                f"{source[span.start : span.end]!r} at those offsets"
            )


def iou(a: Span, b: Span) -> float:
    """Character-level intersection over union of two spans (labels ignored)."""
    intersection = min(a.end, b.end) - max(a.start, b.start)
    if intersection <= 0:
        return 0.0
    union = max(a.end, b.end) - min(a.start, b.start)
    return intersection / union
