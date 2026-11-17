"""The detection pipeline: tokenize -> window -> infer -> remap -> merge -> validate.

Each step is delegated to the module that owns it; this file only wires them
together and is the one place where window-relative offsets become
original-text offsets.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

from pii_extract.backends.base import Backend
from pii_extract.chunking import Window, plan_windows
from pii_extract.dedup import DEFAULT_IOU_THRESHOLD, Candidate, merge_spans
from pii_extract.redaction import RedactionResult, RedactionStrategy, redact
from pii_extract.schema import PII, SchemaPack, apply_validators
from pii_extract.spans import Span, verify_spans

DEFAULT_LABELS: tuple[str, ...] = PII.labels
DEFAULT_THRESHOLD = 0.5
DEFAULT_OVERLAP_TOKENS = 64


def _touches_cut_edge(span: Span, window: Window) -> bool:
    """True if a window-relative span reaches an edge where the document continues."""
    window_length = window.end - window.start
    return (window.cut_start and span.start == 0) or (window.cut_end and span.end == window_length)


class Pipeline:
    """Detect (and optionally redact) PII in text of any length.

    The pipeline never modifies the input string. Returned offsets index the
    exact string passed in.
    """

    def __init__(
        self,
        backend: Backend,
        *,
        labels: Sequence[str] | None = None,
        schema: SchemaPack = PII,
        validators: Collection[str] = (),
        threshold: float = DEFAULT_THRESHOLD,
        window_tokens: int | None = None,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ) -> None:
        if window_tokens is not None and window_tokens <= 0:
            raise ValueError("window_tokens must be positive")
        if overlap_tokens < 0:
            raise ValueError("overlap_tokens must be non-negative")
        self.backend = backend
        self.schema = schema
        self.labels = tuple(schema.labels if labels is None else labels)
        # Canonical label keys whose validators run by default (none: off).
        self.validators = schema.select(validators)
        self.threshold = threshold
        self.window_tokens = window_tokens
        self.overlap_tokens = overlap_tokens
        self.iou_threshold = iou_threshold

    def _window_size(self, labels: Sequence[str]) -> tuple[int, int]:
        """(window budget, overlap) for this request, honouring the model's limit."""
        budget = self.backend.window_budget(labels)
        if self.window_tokens is not None:
            budget = min(budget, self.window_tokens)
        if budget < 1:
            raise ValueError(
                "the label prompt uses the model's whole input, leaving no window budget "
                "for text; use fewer or shorter labels"
            )
        # A too-large overlap is a configuration slip, not a reason to fail a
        # request: cap it at half the window so windows still advance quickly.
        return budget, min(self.overlap_tokens, budget // 2)

    def plan(self, text: str, labels: Sequence[str] | None = None) -> list[Window]:
        """The windows ``extract`` would run for this text."""
        budget, overlap = self._window_size(self._labels(labels))
        return plan_windows(self.backend.tokenize(text), budget, overlap)

    def _labels(self, labels: Sequence[str] | None) -> tuple[str, ...]:
        chosen = tuple(dict.fromkeys(labels)) if labels else self.labels
        if not chosen:
            raise ValueError("at least one label is required")
        return chosen

    def extract(
        self,
        text: str,
        labels: Sequence[str] | None = None,
        threshold: float | None = None,
        validate: bool | None = None,
    ) -> list[Span]:
        """Return non-overlapping spans, sorted by position, in original offsets.

        ``validate`` switches the schema pack's validators for this call:
        ``True`` runs all of them, ``False`` none, ``None`` the configured set.
        """
        chosen = self._labels(labels)
        cutoff = self.threshold if threshold is None else threshold
        if not 0.0 <= cutoff <= 1.0:
            raise ValueError("threshold must be between 0 and 1")

        budget, overlap = self._window_size(chosen)
        windows = plan_windows(self.backend.tokenize(text), budget, overlap)
        if not windows:
            return []

        per_window = self.backend.predict([w.slice(text) for w in windows], chosen, cutoff)

        candidates = [
            Candidate(span.shifted(window.start), clipped=_touches_cut_edge(span, window))
            for window, spans in zip(windows, per_window, strict=True)
            for span in spans
        ]
        merged = merge_spans(candidates, text, self.iou_threshold)
        verify_spans(text, merged)

        enabled = self._enabled_validators(validate)
        if enabled:
            merged, _rejected = apply_validators(merged, self.schema, enabled)
        return merged

    def _enabled_validators(self, validate: bool | None) -> frozenset[str]:
        if validate is None:
            return self.validators
        return frozenset(self.schema.validators) if validate else frozenset()

    def redact(
        self,
        text: str,
        strategy: RedactionStrategy,
        labels: Sequence[str] | None = None,
        threshold: float | None = None,
        validate: bool | None = None,
    ) -> RedactionResult:
        return redact(text, self.extract(text, labels, threshold, validate), strategy)
