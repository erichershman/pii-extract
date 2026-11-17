import pytest

from fakes import CARD, EMAIL, PERSON, PHONE, span
from pii_extract.redaction import (
    HashStrategy,
    MaskStrategy,
    PlaceholderStrategy,
    label_tag,
    make_strategy,
    redact,
    value_key,
)
from pii_extract.spans import OffsetError

TEXT = (
    "Jane Doe (jane.doe@example.com, 555-867-5309) and John Roe (555-010-0199). "
    "Call Jane Doe back on 555-867-5309. Card: 4111 1111 1111 1111."
)


def all_spans():
    return [
        span(TEXT, "Jane Doe", PERSON),
        span(TEXT, "jane.doe@example.com", EMAIL),
        span(TEXT, "555-867-5309", PHONE),
        span(TEXT, "John Roe", PERSON),
        span(TEXT, "555-010-0199", PHONE),
        span(TEXT, "Jane Doe", PERSON, occurrence=1),
        span(TEXT, "555-867-5309", PHONE, occurrence=1),
        span(TEXT, "4111 1111 1111 1111", CARD),
    ]


def assert_consistent(result, original=TEXT):
    """Replacement bookkeeping must agree with both texts."""
    for item in result.replacements:
        assert original[item.span.start : item.span.end] == item.span.text
        assert result.text[item.redacted_start : item.redacted_end] == item.replacement


@pytest.mark.parametrize("name", ["mask", "hash", "placeholder"])
def test_no_detected_value_survives_and_the_rest_is_untouched(name):
    result = redact(TEXT, all_spans(), make_strategy(name))
    assert result.strategy == name
    for item in all_spans():
        assert item.text not in result.text
    for kept in [" (", ") and ", "Call ", " back on ", ". Card: "]:
        assert kept in result.text
    assert len(result.replacements) == len(all_spans())
    assert_consistent(result)


def test_no_spans_returns_the_text_unchanged():
    result = redact(TEXT, [], MaskStrategy())
    assert result.text == TEXT and result.replacements == []


def test_mask_preserves_length_and_therefore_offsets():
    result = redact(TEXT, all_spans(), MaskStrategy())
    assert len(result.text) == len(TEXT)
    for item in result.replacements:
        assert item.replacement == "*" * item.span.length
        assert (item.redacted_start, item.redacted_end) == (item.span.start, item.span.end)
    assert (
        redact(
            "x 555-867-5309", [span("x 555-867-5309", "555-867-5309", PHONE)], MaskStrategy("#")
        ).text
        == "x ############"
    )
    with pytest.raises(ValueError):
        MaskStrategy("**")


def test_placeholders_are_numbered_per_unique_value_in_document_order():
    result = redact(TEXT, all_spans(), PlaceholderStrategy())
    assert result.text == (
        "[PERSON_1] ([EMAIL_1], [PHONE_1]) and [PERSON_2] ([PHONE_2]). "
        "Call [PERSON_1] back on [PHONE_1]. Card: [CREDIT_CARD_1]."
    )


def test_placeholder_numbering_does_not_depend_on_span_order():
    forward = redact(TEXT, all_spans(), PlaceholderStrategy())
    backward = redact(TEXT, list(reversed(all_spans())), PlaceholderStrategy())
    assert forward == backward


def test_placeholder_identity_ignores_case_and_whitespace_runs():
    text = "Jane Doe, JANE  DOE, jane\tdoe and John Roe"
    spans = [
        span(text, "Jane Doe", PERSON),
        span(text, "JANE  DOE", PERSON),
        span(text, "jane\tdoe", PERSON),
        span(text, "John Roe", PERSON),
    ]
    assert redact(text, spans, PlaceholderStrategy()).text == (
        "[PERSON_1], [PERSON_1], [PERSON_1] and [PERSON_2]"
    )
    assert value_key(" Jane \n Doe ") == "jane doe"


def test_placeholder_numbering_is_separate_per_label():
    text = "5550100 and 5550100"
    spans = [span(text, "5550100", PHONE), span(text, "5550100", "account number", occurrence=1)]
    assert redact(text, spans, PlaceholderStrategy()).text == "[PHONE_1] and [ACCOUNT_NUMBER_1]"


def test_one_placeholder_instance_stays_consistent_across_documents():
    strategy = PlaceholderStrategy()
    first = redact("a 555-867-5309", [span("a 555-867-5309", "555-867-5309", PHONE)], strategy)
    text = "b 555-010-0199 c 555-867-5309"
    second = redact(
        text, [span(text, "555-010-0199", PHONE), span(text, "555-867-5309", PHONE)], strategy
    )
    assert first.text == "a [PHONE_1]"
    assert second.text == "b [PHONE_2] c [PHONE_1]"


def test_hash_is_deterministic_for_a_salt_and_changes_with_it():
    a = redact(TEXT, all_spans(), HashStrategy(salt="pepper-one"))
    b = redact(TEXT, all_spans(), HashStrategy(salt="pepper-one"))
    c = redact(TEXT, all_spans(), HashStrategy(salt="pepper-two"))
    assert a == b
    assert a.text != c.text
    assert a.text.startswith("[PERSON:")


def test_hash_is_consistent_per_value_and_distinct_across_values_and_labels():
    result = redact(TEXT, all_spans(), HashStrategy(salt="s3cret-salt"))
    by_value: dict[tuple[str, str], set[str]] = {}
    for item in result.replacements:
        by_value.setdefault((item.span.label, item.span.text), set()).add(item.replacement)
    assert all(len(v) == 1 for v in by_value.values())
    digests = [next(iter(v)) for v in by_value.values()]
    assert len(set(digests)) == len(digests)

    strategy = HashStrategy(salt="s3cret-salt")
    text = "5550100"
    as_phone = strategy.replacement(span(text, text, PHONE))
    as_account = strategy.replacement(span(text, text, "account number"))
    assert as_phone.split(":")[1] != as_account.split(":")[1]


def test_hash_without_a_salt_is_consistent_within_an_instance_only():
    target = span(TEXT, "555-867-5309", PHONE)
    one, two = HashStrategy(), HashStrategy()
    assert one.replacement(target) == one.replacement(target)
    assert one.replacement(target) != two.replacement(target)


def test_hash_format_and_options():
    target = span(TEXT, "555-867-5309", PHONE)
    assert len(HashStrategy(salt="k").replacement(target)) == len("[PHONE:]") + 12
    assert len(HashStrategy(salt="k", digest_chars=32).replacement(target)) == len("[PHONE:]") + 32
    assert "555" not in HashStrategy(salt=b"bytes-salt").replacement(target).split(":")[0]
    with pytest.raises(ValueError):
        HashStrategy(digest_chars=4)


def test_label_tag():
    assert label_tag("phone number") == "PHONE"
    assert label_tag("Phone_Number") == "PHONE"
    assert label_tag("credit card number") == "CREDIT_CARD"
    assert label_tag("date of birth") == "DATE_OF_BIRTH"
    assert label_tag("ssn") == "SSN"
    assert label_tag("  ??  ") == "PII"
    assert label_tag("phone number", aliases={}) == "PHONE_NUMBER"


def test_overlapping_spans_are_rejected():
    spans = [span(TEXT, "Jane Doe", PERSON), span(TEXT, "Doe (jane", EMAIL)]
    with pytest.raises(ValueError, match="overlapping"):
        redact(TEXT, spans, MaskStrategy())


def test_stale_offsets_are_rejected():
    stale = span(TEXT, "555-867-5309", PHONE).shifted(2)
    with pytest.raises(OffsetError):
        redact(TEXT, [stale], MaskStrategy())


def test_adjacent_spans_and_text_edges():
    text = "555-867-5309jane.doe@example.com"
    spans = [span(text, "555-867-5309", PHONE), span(text, "jane.doe@example.com", EMAIL)]
    result = redact(text, spans, PlaceholderStrategy())
    assert result.text == "[PHONE_1][EMAIL_1]"
    assert_consistent(result, text)


def test_unknown_strategy_name():
    with pytest.raises(ValueError, match="unknown redaction strategy"):
        make_strategy("rot13")
