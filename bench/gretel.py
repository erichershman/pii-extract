"""Gold spans from ``gretelai/gretel-pii-masking-en-v1`` (Apache-2.0, synthetic).

The dataset annotates each document with entity STRINGS and their types, not
offsets, so gold offsets are recovered by locating each string in the text.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from bench.metrics import GoldSpan

DATASET_ID = "gretelai/gretel-pii-masking-en-v1"

# benchmark label -> (prompt given to BOTH models, dataset types folded into it).
# The same natural-language prompt is used for both backends: this measures
# zero-shot behaviour, not either model's native label vocabulary.
# Address-like types are deliberately left out of scope.
LABELS: dict[str, tuple[str, tuple[str, ...]]] = {
    "person": ("person", ("name", "first_name", "last_name")),
    "email": ("email", ("email",)),
    "phone number": ("phone number", ("phone_number",)),
    "ssn": ("social security number", ("ssn",)),
    "credit card number": ("credit card number", ("credit_card_number",)),
    "date of birth": ("date of birth", ("date_of_birth",)),
    "ip address": ("ip address", ("ipv4", "ipv6")),
}
PROMPT_TO_LABEL = {prompt: label for label, (prompt, _) in LABELS.items()}
TYPE_TO_LABEL = {t: label for label, (_, types) in LABELS.items() for t in types}


def parse_entities(raw: Any) -> list[dict[str, Any]]:
    """The column is a stringified Python list in the published dataset."""
    if isinstance(raw, list):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return ast.literal_eval(raw)


def _occurrences(text: str, value: str) -> list[tuple[int, int]]:
    """Whole-token occurrences: 'Al' must not match inside 'Also'."""
    pattern = r"(?<![0-9A-Za-z])" + re.escape(value) + r"(?![0-9A-Za-z])"
    return [(m.start(), m.end()) for m in re.finditer(pattern, text)]


def gold_spans(text: str, entities: list[dict[str, Any]]) -> tuple[list[GoldSpan], int]:
    """Gold (start, end, label) spans, plus how many annotations could not be located.

    Two normalizations, both needed for a like-for-like comparison with span
    predictors:

    * A gold span that lies inside a longer gold span of the same label is
      dropped (the dataset lists "Jane", "Doe" and "Jane Doe" separately).
    * Same-label gold spans separated only by a single space are merged, so a
      first name followed by a last name is one "person", as the models
      report it.
    """
    found: set[GoldSpan] = set()
    missing = 0
    for entity in entities:
        value = str(entity.get("entity", "")).strip()
        labels = {TYPE_TO_LABEL[t] for t in entity.get("types", []) if t in TYPE_TO_LABEL}
        if not value or not labels:
            continue
        positions = _occurrences(text, value)
        if not positions:
            missing += 1
            continue
        for label in labels:
            found.update((start, end, label) for start, end in positions)

    by_label: dict[str, list[tuple[int, int]]] = {}
    for start, end, label in found:
        by_label.setdefault(label, []).append((start, end))

    result: list[GoldSpan] = []
    for label, spans in by_label.items():
        merged: list[list[int]] = []
        for start, end in sorted(spans, key=lambda s: (s[0], -s[1])):
            if merged and end <= merged[-1][1]:
                continue  # nested inside the previous span
            if merged and (start <= merged[-1][1] or text[merged[-1][1] : start] == " "):
                merged[-1][1] = max(merged[-1][1], end)  # overlapping or adjacent
            else:
                merged.append([start, end])
        result.extend((start, end, label) for start, end in merged)
    return sorted(result), missing
