"""Redaction: replace detected spans in the original text.

``redact`` does the splicing once, for every strategy. A strategy only
answers "what replaces this span?", so adding one means writing a small class
and registering it under a name.

Three strategies ship:

* ``mask``         every character of the span becomes ``*`` (length-preserving)
* ``hash``         ``[EMAIL:9f2c1a7b3d4e]`` - a salted (keyed) HMAC-SHA256 digest
* ``placeholder``  ``[PHONE_1]`` - typed, numbered per unique value

``hash`` and ``placeholder`` are consistent: the same value under the same
label always gets the same replacement within one strategy instance. "Same
value" means equal after whitespace collapsing and case folding, so
``Jane  Doe`` and ``jane doe`` share a placeholder.
"""

from __future__ import annotations

import hashlib
import hmac
import itertools
import re
import secrets
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from pii_extract.spans import Span, verify_spans


class RedactionStrategy(Protocol):
    """Maps a span to its replacement string. May keep per-document state."""

    name: str

    def replacement(self, span: Span) -> str: ...


class Replacement(BaseModel):
    """One applied substitution, addressable in both texts."""

    model_config = ConfigDict(frozen=True)

    span: Span
    replacement: str
    redacted_start: int
    redacted_end: int


class RedactionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    strategy: str
    replacements: list[Replacement]


def value_key(value: str) -> str:
    """Identity of a PII value for consistency purposes."""
    return " ".join(value.split()).casefold()


# Short tags for common labels; anything else is derived from the label itself.
DEFAULT_TAG_ALIASES: dict[str, str] = {
    "phone number": "PHONE",
    "phone_number": "PHONE",
    "email address": "EMAIL",
    "credit card number": "CREDIT_CARD",
    "card_number": "CREDIT_CARD",
    "payment_card": "CREDIT_CARD",
    "social security number": "SSN",
}


def label_tag(label: str, aliases: dict[str, str] | None = None) -> str:
    """``"phone number"`` -> ``"PHONE"``; ``"date of birth"`` -> ``"DATE_OF_BIRTH"``."""
    table = DEFAULT_TAG_ALIASES if aliases is None else aliases
    alias = table.get(label.strip().casefold())
    if alias:
        return alias
    tag = re.sub(r"[^0-9A-Za-z]+", "_", label).strip("_").upper()
    return tag or "PII"


class MaskStrategy:
    """Overwrite every character, preserving the span's length."""

    name = "mask"

    def __init__(self, mask_char: str = "*") -> None:
        if len(mask_char) != 1:
            raise ValueError("mask_char must be exactly one character")
        self._mask_char = mask_char

    def replacement(self, span: Span) -> str:
        return self._mask_char * span.length


class HashStrategy:
    """Keyed digest of the value: stable for joins, not reversible without the salt.

    The salt is used as an HMAC key, not concatenated, so the construction is a
    proper MAC. The label is mixed in, so the same digits seen as a phone and
    as an account number hash differently. With no salt supplied a random one
    is generated per instance: digests are then consistent within a document
    but unlinkable across documents.

    PII values are low-entropy (there are only 10^10 phone numbers), so the
    digest is only as private as the salt. Treat the salt as a secret.
    """

    name = "hash"

    def __init__(
        self,
        salt: str | bytes | None = None,
        digest_chars: int = 12,
        tag_aliases: dict[str, str] | None = None,
    ) -> None:
        if not 8 <= digest_chars <= 64:
            raise ValueError("digest_chars must be between 8 and 64")
        if salt is None or salt == "" or salt == b"":
            salt = secrets.token_bytes(32)
        self._key = salt.encode("utf-8") if isinstance(salt, str) else salt
        self._digest_chars = digest_chars
        self._tag_aliases = tag_aliases

    def replacement(self, span: Span) -> str:
        message = f"{span.label.casefold()}\x1f{value_key(span.text)}".encode()
        digest = hmac.new(self._key, message, hashlib.sha256).hexdigest()
        return f"[{label_tag(span.label, self._tag_aliases)}:{digest[: self._digest_chars]}]"


class PlaceholderStrategy:
    """``[PHONE_1]``, ``[PHONE_2]``, ... numbered per label in order of first use.

    Numbering is stateful: ask for replacements in document order (``redact``
    does) and the first distinct phone number is ``[PHONE_1]``. Reusing one
    instance across documents keeps the numbering consistent across them.
    """

    name = "placeholder"

    def __init__(self, tag_aliases: dict[str, str] | None = None) -> None:
        self._tag_aliases = tag_aliases
        self._numbers: dict[str, dict[str, int]] = defaultdict(dict)

    def replacement(self, span: Span) -> str:
        tag = label_tag(span.label, self._tag_aliases)
        seen = self._numbers[tag]
        number = seen.setdefault(value_key(span.text), len(seen) + 1)
        return f"[{tag}_{number}]"


STRATEGIES: dict[str, Callable[..., RedactionStrategy]] = {
    MaskStrategy.name: MaskStrategy,
    HashStrategy.name: HashStrategy,
    PlaceholderStrategy.name: PlaceholderStrategy,
}


def make_strategy(name: str, **options: object) -> RedactionStrategy:
    """Instantiate a registered strategy by name."""
    try:
        factory = STRATEGIES[name]
    except KeyError:
        raise ValueError(
            f"unknown redaction strategy {name!r}; choose from {sorted(STRATEGIES)}"
        ) from None
    return factory(**options)


def redact(text: str, spans: Sequence[Span], strategy: RedactionStrategy) -> RedactionResult:
    """Replace every span in ``text`` and report where each replacement landed.

    Spans must address ``text`` exactly and must not overlap; both are checked,
    because splicing with stale or overlapping offsets would silently corrupt
    the output (and could leave PII in it).
    """
    ordered = sorted(spans, key=lambda s: (s.start, s.end))
    verify_spans(text, ordered)
    for previous, current in itertools.pairwise(ordered):
        if current.start < previous.end:
            raise ValueError(
                f"overlapping spans cannot be redacted: {previous.key} and {current.key}"
            )

    pieces: list[str] = []
    replacements: list[Replacement] = []
    cursor = 0  # position in the original text
    written = 0  # length of the redacted text so far
    for span in ordered:
        untouched = text[cursor : span.start]
        substitute = strategy.replacement(span)
        pieces += (untouched, substitute)
        written += len(untouched)
        replacements.append(
            Replacement(
                span=span,
                replacement=substitute,
                redacted_start=written,
                redacted_end=written + len(substitute),
            )
        )
        written += len(substitute)
        cursor = span.end
    pieces.append(text[cursor:])
    return RedactionResult(text="".join(pieces), strategy=strategy.name, replacements=replacements)
