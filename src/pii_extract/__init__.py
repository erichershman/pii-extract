"""pii-extract: PII detection and redaction on GLiNER-family zero-shot NER."""

from pii_extract.pipeline import DEFAULT_LABELS, Pipeline
from pii_extract.redaction import RedactionResult, make_strategy, redact
from pii_extract.spans import OffsetError, Span

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_LABELS",
    "OffsetError",
    "Pipeline",
    "RedactionResult",
    "Span",
    "__version__",
    "make_strategy",
    "redact",
]
