import pytest

from fakes import CARD, EMAIL, PERSON, PHONE, span
from pii_extract.schema import (
    PACKS,
    PII,
    PII_EXTENDED,
    apply_validators,
    get_pack,
    is_credit_card,
    is_email,
    is_ip_address,
    is_phone,
    is_ssn,
    luhn_valid,
    normalize_label,
)


@pytest.mark.parametrize(
    "value",
    [
        "jane.doe@example.com",
        "JANE.DOE@EXAMPLE.COM",
        "jane+tag@mail.example.org",
        "o'brien@example.co.uk",
        "jane.doe@example.com,",  # trailing punctuation from a sloppy boundary
        "<jane.doe@example.com>",
    ],
)
def test_email_accepts(value):
    assert is_email(value)


@pytest.mark.parametrize(
    "value",
    [
        "jane.doe",
        "jane.doe@",
        "@example.com",
        "jane doe@example.com",
        "jane@example",
        "jane@@example.com",
        "",
    ],
)
def test_email_rejects(value):
    assert not is_email(value)


@pytest.mark.parametrize(
    "value", ["000-12-3456", "000 12 3456", "000123456", "900-12-3456", "000-12-3456."]
)
def test_ssn_accepts_the_shape_including_reserved_ranges(value):
    assert is_ssn(value)


@pytest.mark.parametrize(
    "value",
    [
        "000-12-345",
        "000-123-456",
        "000-12 3456",
        "0000-12-3456",
        "ZZ 12 34 56 T",
        "MRN-000123",
        "12345678",
        "",
    ],
)
def test_ssn_rejects(value):
    assert not is_ssn(value)


@pytest.mark.parametrize(
    "value",
    [
        "555-867-5309",
        "(555) 867-5309",
        "+1 555 867 5309",
        "555.867.5309",
        "555-0100",
        "+1-555-867-5309x7005",
        "555-867-5309 ext. 12",
        "+44 5550 100200",
    ],
)
def test_phone_accepts(value):
    assert is_phone(value)


@pytest.mark.parametrize(
    "value",
    [
        "555-867",
        "5",
        "555-867-5309 or so",
        "call me",
        "1234567890123456",
        "4111 1111 1111 1111",
        "",
    ],
)
def test_phone_rejects(value):
    assert not is_phone(value)


def test_luhn():
    assert luhn_valid("4111111111111111")
    assert luhn_valid("5555555555554444")
    assert luhn_valid("0")
    assert not luhn_valid("4111111111111112")
    assert not luhn_valid("411111111111111a")
    assert not luhn_valid("")


@pytest.mark.parametrize(
    "value",
    ["4111 1111 1111 1111", "4111-1111-1111-1111", "4111111111111111", "4111 1111 1111 1111."],
)
def test_credit_card_accepts(value):
    assert is_credit_card(value)


@pytest.mark.parametrize(
    "value",
    [
        "4111 1111 1111 1112",  # right shape, wrong checksum
        "4111 1111",  # a fragment
        "4111 1111 1111 1111 1111 1111",  # too long
        "card 4111 1111 1111 1111",
        "",
    ],
)
def test_credit_card_rejects(value):
    assert not is_credit_card(value)


def test_ip_address():
    assert (
        is_ip_address("192.0.2.17")
        and is_ip_address("2001:db8::1")
        and is_ip_address("192.0.2.17.")
    )
    assert not is_ip_address("999.0.2.17") and not is_ip_address("192.0.2")


def test_normalize_label():
    assert normalize_label("Phone_Number") == "phone number"
    assert normalize_label("  credit-card   NUMBER ") == "credit card number"


def test_packs_and_aliases():
    assert set(PACKS) == {"pii", "pii-extended"}
    assert get_pack("pii") is PII
    with pytest.raises(ValueError, match="unknown schema pack"):
        get_pack("nope")
    assert PII.labels == ("person", "email", "phone number", "credit card number", "ssn")
    assert PII.validator_for("social security number") is is_ssn
    assert PII.validator_for("card_number") is is_credit_card
    assert PII.validator_for("Phone Number") is is_phone
    assert PII.validator_for("person") is None
    assert PII.validator_for("ip address") is None
    assert PII_EXTENDED.validator_for("ipv4") is is_ip_address


def test_select():
    assert PII.select([]) == frozenset()
    assert PII.select(["all"]) == {"email", "phone number", "credit card number", "ssn"}
    assert PII.select(["SSN", "payment card"]) == {"ssn", "credit card number"}
    with pytest.raises(ValueError, match="no validator"):
        PII.select(["person"])


TEXT = "Jane Doe, jane.doe@example, 555-867-5309, 555-867, 4111 1111 1111 1111, 4111 1111 1111 1112"


def detected():
    return [
        span(TEXT, "Jane Doe", PERSON),
        span(TEXT, "jane.doe@example", EMAIL),
        span(TEXT, "555-867-5309", PHONE),
        span(TEXT, "555-867", "phone_number", occurrence=1),
        span(TEXT, "4111 1111 1111 1111", CARD),
        span(TEXT, "4111 1111 1111 1112", "card number"),
    ]


def test_apply_validators_rejects_only_spans_that_fail_an_enabled_validator():
    kept, rejected = apply_validators(detected(), PII, PII.select(["all"]))
    assert [s.text for s in kept] == ["Jane Doe", "555-867-5309", "4111 1111 1111 1111"]
    assert [s.text for s in rejected] == ["jane.doe@example", "555-867", "4111 1111 1111 1112"]


def test_apply_validators_respects_the_selection():
    kept, rejected = apply_validators(detected(), PII, PII.select(["credit card number"]))
    assert [s.text for s in rejected] == ["4111 1111 1111 1112"]
    assert len(kept) == 5

    kept, rejected = apply_validators(detected(), PII, frozenset())
    assert kept == detected() and rejected == []


def test_labels_without_a_validator_always_pass():
    odd = [span(TEXT, "Jane Doe", "employee nickname")]
    assert apply_validators(odd, PII, PII.select(["all"])) == (odd, [])
