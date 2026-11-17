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
LabelTable = dict[str, tuple[str, tuple[str, ...]]]

BASE_LABELS: LabelTable = {
    "person": ("person", ("name", "first_name", "last_name")),
    "email": ("email", ("email",)),
    "phone number": ("phone number", ("phone_number",)),
    "ssn": ("social security number", ("ssn",)),
    "credit card number": ("credit card number", ("credit_card_number",)),
    "date of birth": ("date of birth", ("date_of_birth",)),
    "ip address": ("ip address", ("ipv4", "ipv6")),
}

# Optional labels that can be added to a run with --extra-labels. Each gives
# the models somewhere else to put identifiers that are not in the base set.
EXTRA_LABELS: LabelTable = {
    "medical record number": ("medical record number", ("medical_record_number",)),
    "customer id": ("customer id", ("customer_id",)),
    "employee id": ("employee id", ("employee_id",)),
}


def label_table(extra: tuple[str, ...] | list[str] = ()) -> LabelTable:
    """The base labels plus the named extras, in a stable order."""
    table = dict(BASE_LABELS)
    for name in extra:
        if name not in EXTRA_LABELS:
            raise ValueError(f"unknown extra label {name!r}; choose from {sorted(EXTRA_LABELS)}")
        table[name] = EXTRA_LABELS[name]
    return table


def prompts(table: LabelTable) -> list[str]:
    return [prompt for prompt, _ in table.values()]


def prompt_to_label(table: LabelTable) -> dict[str, str]:
    return {prompt: label for label, (prompt, _) in table.items()}


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


def gold_spans(
    text: str, entities: list[dict[str, Any]], table: LabelTable = BASE_LABELS
) -> tuple[list[GoldSpan], int]:
    """Gold (start, end, label) spans, plus how many annotations could not be located.

    Two normalizations, both needed for a like-for-like comparison with span
    predictors:

    * A gold span that lies inside a longer gold span of the same label is
      dropped (the dataset lists "Jane", "Doe" and "Jane Doe" separately).
    * Same-label gold spans separated only by a single space are merged, so a
      first name followed by a last name is one "person", as the models
      report it.
    """
    type_to_label = {t: label for label, (_, types) in table.items() for t in types}
    found: set[GoldSpan] = set()
    missing = 0
    for entity in entities:
        value = str(entity.get("entity", "")).strip()
        labels = {type_to_label[t] for t in entity.get("types", []) if t in type_to_label}
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
