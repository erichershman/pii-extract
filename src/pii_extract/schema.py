"""Schema packs: a label set plus optional per-label post-processors.

A pack answers two questions for the pipeline: which entity types to ask the
model for by default, and how to sanity-check what comes back. The only kind
of post-processor that exists today is a validator, a predicate on the
detected text, so that is the type; nothing here transforms a span.

Validators are OFF unless switched on. A validator can only ever remove a
detection, and a detection that fails its format check may still be sensitive
(a card number with one mistyped digit fails Luhn and is still a card
number). Dropping it buys precision at the price of recall, which is the
wrong default for a redactor. Turn validators on for extraction and analytics,
where a wrongly typed entity is the costlier mistake.
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from pii_extract.spans import Span

Validator = Callable[[str], bool]


def normalize_label(label: str) -> str:
    """``"Phone_Number"`` -> ``"phone number"``: the key labels are compared by."""
    return " ".join(re.sub(r"[_\-]+", " ", label).split()).casefold()


# --- validators ---------------------------------------------------------------
#
# Each takes the detected text and says whether it has the shape its label
# promises. They tolerate the punctuation models commonly leave attached to a
# span (a trailing comma or full stop), because rejecting "jane@example.com,"
# as "not an email" would punish a boundary quirk, not a wrong detection.

_TRAILING = " \t\r\n.,;:"
_WRAPPING = " \t\r\n<>()[]\"'"

_EMAIL = re.compile(
    r"[A-Za-z0-9._%+\-']+@(?:[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,63}"
)
# Same separator (hyphen, space, or none) in both positions.
_SSN = re.compile(r"\d{3}([- ]?)\d{2}\1\d{4}")
_PHONE = re.compile(
    r"(?P<number>\+?[\d\s().\-/]+?)(?:\s*(?:x|ext\.?|extension)\s*\d{1,6})?", re.IGNORECASE
)
_CARD = re.compile(r"[\d \-]+")


def is_email(value: str) -> bool:
    return _EMAIL.fullmatch(value.strip(_WRAPPING).rstrip(_TRAILING)) is not None


def is_ssn(value: str) -> bool:
    """US SSN shape: nine digits grouped 3-2-4.

    Shape only, on purpose. The SSA's allocation rules (no area 000, 666 or
    9xx; no group 00) are not enforced, because synthetic fixtures use exactly
    those reserved ranges to stay harmless, and because a 9xx number is an
    ITIN, which is just as sensitive.
    """
    return _SSN.fullmatch(value.strip(_TRAILING)) is not None


def is_phone(value: str) -> bool:
    """Phone-number characters only, with 7 to 15 digits (15 is the E.164 maximum).

    An extension ("x7005", "ext. 12") is allowed and not counted.
    """
    match = _PHONE.fullmatch(value.strip(_TRAILING))
    if match is None:
        return False
    digits = sum(ch.isdigit() for ch in match["number"])
    return 7 <= digits <= 15


def luhn_valid(digits: str) -> bool:
    """The Luhn checksum over a string of decimal digits."""
    if not digits.isdigit():
        return False
    total = 0
    for position, char in enumerate(reversed(digits)):
        digit = int(char)
        if position % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def is_credit_card(value: str) -> bool:
    """12 to 19 digits, optionally grouped by spaces or hyphens, passing Luhn."""
    core = value.strip(_TRAILING)
    if _CARD.fullmatch(core) is None:
        return False
    digits = re.sub(r"\D", "", core)
    return 12 <= len(digits) <= 19 and luhn_valid(digits)


def is_ip_address(value: str) -> bool:
    try:
        ipaddress.ip_address(value.strip(_TRAILING))
    except ValueError:
        return False
    return True


# --- packs --------------------------------------------------------------------


@dataclass(frozen=True)
class SchemaPack:
    """A named label set with validators for some of its labels.

    ``validators`` and ``aliases`` are keyed by normalized label, so the pack
    recognises ``"phone_number"`` and ``"Phone Number"`` as the same type, and
    an alias such as ``"social security number"`` as ``"ssn"``. A label with no
    validator is never filtered.
    """

    name: str
    labels: tuple[str, ...]
    validators: Mapping[str, Validator] = field(default_factory=dict)
    aliases: Mapping[str, str] = field(default_factory=dict)

    def canonical(self, label: str) -> str:
        """The pack's key for a label, following aliases."""
        key = normalize_label(label)
        return self.aliases.get(key, key)

    def validator_for(self, label: str) -> Validator | None:
        return self.validators.get(self.canonical(label))

    def select(self, selection: Iterable[str]) -> frozenset[str]:
        """Resolve a validator selection to canonical label keys.

        ``["all"]`` means every validator the pack has; an empty selection
        means none; otherwise each entry must name a label that has one.
        """
        chosen = [item for item in selection if item]
        if any(normalize_label(item) == "all" for item in chosen):
            return frozenset(self.validators)
        keys = set()
        for item in chosen:
            key = self.canonical(item)
            if key not in self.validators:
                raise ValueError(
                    f"schema pack {self.name!r} has no validator for {item!r}; "
                    f"available: {sorted(self.validators)}"
                )
            keys.add(key)
        return frozenset(keys)


_ALIASES: dict[str, str] = {
    "email address": "email",
    "e mail": "email",
    "phone": "phone number",
    "telephone number": "phone number",
    "mobile number": "phone number",
    "social security number": "ssn",
    "us social security number": "ssn",
    "credit card": "credit card number",
    "card number": "credit card number",
    "payment card": "credit card number",
    "debit card number": "credit card number",
    "ip": "ip address",
    "ipv4": "ip address",
    "ipv6": "ip address",
    "ipv4 address": "ip address",
    "ipv6 address": "ip address",
}

PII = SchemaPack(
    name="pii",
    labels=("person", "email", "phone number", "credit card number", "ssn"),
    validators={
        "email": is_email,
        "phone number": is_phone,
        "credit card number": is_credit_card,
        "ssn": is_ssn,
    },
    aliases=_ALIASES,
)

PII_EXTENDED = SchemaPack(
    name="pii-extended",
    labels=(*PII.labels, "date of birth", "ip address", "medical record number"),
    validators={**PII.validators, "ip address": is_ip_address},
    aliases=_ALIASES,
)

PACKS: dict[str, SchemaPack] = {pack.name: pack for pack in (PII, PII_EXTENDED)}
DEFAULT_PACK = PII.name


def get_pack(name: str) -> SchemaPack:
    try:
        return PACKS[name]
    except KeyError:
        raise ValueError(f"unknown schema pack {name!r}; choose from {sorted(PACKS)}") from None


def apply_validators(
    spans: Sequence[Span], pack: SchemaPack, enabled: Collection[str]
) -> tuple[list[Span], list[Span]]:
    """Split spans into (kept, rejected) using the enabled validators.

    ``enabled`` holds canonical label keys, as returned by ``SchemaPack.select``.
    A span is rejected only if its label has an enabled validator and the
    span's text fails it.
    """
    kept: list[Span] = []
    rejected: list[Span] = []
    for span in spans:
        key = pack.canonical(span.label)
        validator = pack.validators.get(key) if key in enabled else None
        if validator is None or validator(span.text):
            kept.append(span)
        else:
            rejected.append(span)
    return kept, rejected
