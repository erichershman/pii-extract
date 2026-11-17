"""Deterministic stand-ins for a model, so pipeline logic is testable without torch.

All data here is synthetic: 555 phone numbers, the 4111 test card, example.com
emails, and made-up names.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pii_extract.backends.base import checked_span
from pii_extract.chunking import Token
from pii_extract.spans import Span

EMAIL = "email"
PHONE = "phone number"
CARD = "credit card number"
PERSON = "person"
ALL_LABELS = (PERSON, EMAIL, PHONE, CARD)

FULL_CARD = "4111 1111 1111 1111"

# The card and name patterns deliberately match PREFIXES of the full entity,
# the way a real model reports a fragment when a window cuts an entity in two.
_PATTERNS: dict[str, re.Pattern[str]] = {
    EMAIL: re.compile(r"[A-Za-z0-9._+-]+@example\.(?:com|org)"),
    PHONE: re.compile(r"555-\d{3}-\d{4}"),
    CARD: re.compile(r"4111(?: 1111){0,3}"),
    PERSON: re.compile(r"Jane(?:\s+Doe)?|John(?:\s+Roe)?"),
}


class FakeBackend:
    """Regex 'model' with a whitespace tokenizer and a configurable window budget.

    A fragment (a card number or name cut short by the end of the text it was
    given) is scored HIGHER than a complete match, to prove that dedup prefers
    whole entities because of the clipping rule and not by luck of the scores.
    """

    name = "fake"
    model_id = "fake/regex"
    device = "cpu"

    def __init__(self, budget: int = 12, patterns: dict[str, re.Pattern[str]] | None = None):
        self.budget = budget
        self.patterns = patterns or _PATTERNS
        self.loaded = False
        self.calls: list[list[str]] = []

    def load(self) -> None:
        self.loaded = True

    def close(self) -> None:
        self.loaded = False

    def tokenize(self, text: str) -> list[Token]:
        return [Token(m.start(), m.end()) for m in re.finditer(r"\S+", text)]

    def window_budget(self, labels: Sequence[str]) -> int:
        return self.budget

    def predict(
        self, texts: Sequence[str], labels: Sequence[str], threshold: float
    ) -> list[list[Span]]:
        self.load()
        self.calls.append(list(texts))
        results: list[list[Span]] = []
        for text in texts:
            spans = []
            for label in labels:
                pattern = self.patterns.get(label)
                if pattern is None:
                    continue
                for match in pattern.finditer(text):
                    fragment = match.end() == len(text) and (
                        (label == CARD and match.group() != FULL_CARD)
                        or (label == PERSON and " " not in match.group().strip())
                    )
                    score = 0.99 if fragment else 0.9
                    if score >= threshold:
                        spans.append(
                            checked_span(
                                text, match.start(), match.end(), label, score, match.group()
                            )
                        )
            results.append(spans)
        return results


def span(text: str, value: str, label: str, score: float = 0.9, occurrence: int = 0) -> Span:
    """Build the span for the n-th occurrence of ``value`` in ``text``."""
    start = -1
    for _ in range(occurrence + 1):
        start = text.index(value, start + 1)
    return Span(start=start, end=start + len(value), label=label, score=score, text=value)
