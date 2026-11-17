import re

import pytest
from fastapi.testclient import TestClient

from fakes import _PATTERNS as FAKE_PATTERNS
from fakes import ALL_LABELS, CARD, FakeBackend
from pii_extract.api import create_app
from pii_extract.config import Settings
from pii_extract.pipeline import Pipeline
from pii_extract.spans import Span

TEXT = (
    "Contact Jane Doe at jane.doe@example.com or 555-867-5309. Card on file: 4111 1111 1111 1111."
)


def make_client(budget: int = 64, validators=(), **settings) -> tuple[TestClient, FakeBackend]:
    backend = FakeBackend(budget)
    pipeline = Pipeline(backend, labels=ALL_LABELS, overlap_tokens=4, validators=validators)
    app = create_app(settings=Settings(**settings), pipeline=pipeline)
    return TestClient(app), backend


@pytest.fixture
def client():
    test_client, _ = make_client()
    with test_client:
        yield test_client


def test_healthz(client):
    body = client.get("/healthz").json()
    assert body == {
        "status": "ok",
        "version": "0.1.0",
        "backend": "fake",
        "model": "fake/regex",
        "device": "cpu",
        "model_loaded": False,
        "schema_pack": "pii",
        "validators": [],
    }


def test_preload_loads_at_startup_and_shutdown_releases():
    test_client, backend = make_client(preload=True)
    with test_client:
        assert test_client.get("/healthz").json()["model_loaded"] is True
    assert backend.loaded is False


def test_extract_returns_spans_that_index_the_request_text(client):
    response = client.post("/extract", json={"text": TEXT})
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "fake/regex"
    assert [(e["label"], e["text"]) for e in body["entities"]] == [
        ("person", "Jane Doe"),
        ("email", "jane.doe@example.com"),
        ("phone number", "555-867-5309"),
        ("credit card number", "4111 1111 1111 1111"),
    ]
    for entity in body["entities"]:
        assert TEXT[entity["start"] : entity["end"]] == entity["text"]
        assert set(entity) == {"start", "end", "label", "score", "text"}


def test_extract_honours_labels_and_threshold(client):
    body = client.post("/extract", json={"text": TEXT, "labels": ["email"]}).json()
    assert [e["label"] for e in body["entities"]] == ["email"]
    body = client.post("/extract", json={"text": TEXT, "threshold": 0.95}).json()
    assert body["entities"] == []


def test_offsets_survive_json_transport_of_non_ascii_text(client):
    text = "﻿café \U0001f600\r\n mail jane.doe@example.com 你好 555-867-5309"
    entities = client.post("/extract", json={"text": text}).json()["entities"]
    assert [text[e["start"] : e["end"]] for e in entities] == [
        "jane.doe@example.com",
        "555-867-5309",
    ]


def test_long_request_is_windowed():
    test_client, backend = make_client(budget=10)
    text = " ".join(f"filler{i} 555-{100 + i}-{1000 + i}" for i in range(50))
    with test_client:
        entities = test_client.post("/extract", json={"text": text}).json()["entities"]
    assert len(entities) == 50
    assert len(backend.calls[0]) > 5, "expected several windows in one batch"


def test_redact_defaults_to_placeholders(client):
    text = TEXT + " Again: 555-867-5309."
    body = client.post("/redact", json={"text": text}).json()
    assert body["redacted_text"] == (
        "Contact [PERSON_1] at [EMAIL_1] or [PHONE_1]. Card on file: [CREDIT_CARD_1]. "
        "Again: [PHONE_1]."
    )
    assert body["strategy"] == "placeholder"
    for item in body["replacements"]:
        start, end = item["redacted_start"], item["redacted_end"]
        assert body["redacted_text"][start:end] == item["replacement"]
        assert text[item["span"]["start"] : item["span"]["end"]] == item["span"]["text"]


def test_redact_mask(client):
    body = client.post("/redact", json={"text": TEXT, "strategy": "mask", "mask_char": "#"}).json()
    assert len(body["redacted_text"]) == len(TEXT)
    assert "####################" in body["redacted_text"]
    assert "Jane" not in body["redacted_text"] and "4111" not in body["redacted_text"]


def test_redact_hash_uses_the_server_salt():
    salted, _ = make_client(hash_salt="server-side-pepper")
    request = {"text": TEXT, "strategy": "hash"}
    with salted:
        first = salted.post("/redact", json=request).json()["redacted_text"]
        second = salted.post("/redact", json=request).json()["redacted_text"]
    assert first == second and "[EMAIL:" in first

    unsalted, _ = make_client()
    with unsalted:
        a = unsalted.post("/redact", json=request).json()["redacted_text"]
        b = unsalted.post("/redact", json=request).json()["redacted_text"]
    assert a != b, "without a configured salt, digests must not link across requests"


def test_placeholder_numbering_restarts_for_each_request(client):
    first = client.post("/redact", json={"text": "call 555-867-5309"}).json()
    second = client.post("/redact", json={"text": "call 555-010-0199"}).json()
    assert first["redacted_text"] == second["redacted_text"] == "call [PHONE_1]"


@pytest.mark.parametrize(
    "path,payload",
    [
        ("/extract", {}),
        ("/extract", {"text": 5}),
        ("/extract", {"text": "x", "threshold": 2}),
        ("/redact", {"text": "x", "strategy": "rot13"}),
        ("/redact", {"text": "x", "mask_char": "**"}),
    ],
)
def test_invalid_requests_are_rejected(client, path, payload):
    assert client.post(path, json=payload).status_code == 422


def test_oversized_text_is_rejected():
    test_client, _ = make_client(max_text_chars=10)
    with test_client:
        assert test_client.post("/extract", json={"text": "x" * 11}).status_code == 413
        assert test_client.post("/redact", json={"text": "x" * 11}).status_code == 413
        assert test_client.post("/extract", json={"text": "x" * 10}).status_code == 200


def test_empty_text_is_fine(client):
    assert client.post("/extract", json={"text": ""}).json()["entities"] == []
    assert client.post("/redact", json={"text": ""}).json()["redacted_text"] == ""


def test_openapi_documents_the_labels_example(client):
    schema = client.get("/openapi.json").json()
    labels = schema["components"]["schemas"]["ExtractRequest"]["properties"]["labels"]
    assert ["person", "email", "phone number", "ssn"] in labels["examples"]
    assert {"/healthz", "/extract", "/redact"} <= set(schema["paths"])


def test_each_app_has_its_own_pipeline():
    a, backend_a = make_client()
    b, backend_b = make_client()
    with a, b:
        a.post("/extract", json={"text": TEXT})
    assert len(backend_a.calls) == 1 and backend_b.calls == []


class DriftingBackend(FakeBackend):
    """A model whose reported offsets are off by one."""

    def predict(self, texts, labels, threshold):
        results = super().predict(texts, labels, threshold)
        return [
            [
                Span.model_construct(**{**s.model_dump(), "start": s.start + 1, "end": s.end + 1})
                for s in spans
            ]
            for spans in results
        ]


def test_unverifiable_offsets_fail_the_request_instead_of_returning_wrong_spans():
    pipeline = Pipeline(DriftingBackend(64), labels=ALL_LABELS)
    with TestClient(create_app(settings=Settings(), pipeline=pipeline)) as test_client:
        for path in ("/extract", "/redact"):
            response = test_client.post(path, json={"text": TEXT})
            assert response.status_code == 500
            assert response.json() == {"detail": "span offset verification failed"}
            assert "Jane" not in response.text


# "4111 1111 1111 1112" has the shape of a card number but fails the Luhn check.
BAD_CARD_TEXT = "Cards 4111 1111 1111 1111 and 4111 1111 1111 1112, mail jane.doe@example.com"


def loose_card_client(validators=()) -> TestClient:
    backend = FakeBackend(64, patterns={**FAKE_PATTERNS, CARD: re.compile(r"\d{4}(?: \d{4}){3}")})
    pipeline = Pipeline(backend, labels=ALL_LABELS, validators=validators)
    return TestClient(create_app(settings=Settings(), pipeline=pipeline))


def cards(response) -> list[str]:
    return [e["text"] for e in response.json()["entities"] if e["label"] == CARD]


def test_validators_are_off_by_default_and_switchable_per_request():
    with loose_card_client() as test_client:
        assert test_client.get("/healthz").json()["validators"] == []
        default = test_client.post("/extract", json={"text": BAD_CARD_TEXT})
        assert cards(default) == ["4111 1111 1111 1111", "4111 1111 1111 1112"]

        checked = test_client.post("/extract", json={"text": BAD_CARD_TEXT, "validate": True})
        assert cards(checked) == ["4111 1111 1111 1111"]
        assert "jane.doe@example.com" in [e["text"] for e in checked.json()["entities"]]


def test_server_side_validators_apply_unless_the_request_turns_them_off():
    with loose_card_client(validators=["credit card number"]) as test_client:
        assert test_client.get("/healthz").json()["validators"] == ["credit card number"]
        assert cards(test_client.post("/extract", json={"text": BAD_CARD_TEXT})) == [
            "4111 1111 1111 1111"
        ]
        off = test_client.post("/extract", json={"text": BAD_CARD_TEXT, "validate": False})
        assert len(cards(off)) == 2


def test_redact_with_validation_leaves_the_rejected_value_in_place():
    # This is the trade-off validators make, stated as a test: the value that
    # fails its check is no longer redacted.
    with loose_card_client() as test_client:
        safe = test_client.post("/redact", json={"text": BAD_CARD_TEXT}).json()["redacted_text"]
        assert "1112" not in safe and "[CREDIT_CARD_2]" in safe
        strict = test_client.post("/redact", json={"text": BAD_CARD_TEXT, "validate": True})
        assert "4111 1111 1111 1112" in strict.json()["redacted_text"]
        assert "4111 1111 1111 1111" not in strict.json()["redacted_text"]


def test_extract_and_redact_merge_differently_on_purpose():
    from test_pipeline_offsets import DisagreeingBackend

    text = "Signed by Jane Doe Smith yesterday."
    pipeline = Pipeline(DisagreeingBackend(64), labels=["person"])
    with TestClient(create_app(settings=Settings(), pipeline=pipeline)) as test_client:
        extracted = test_client.post("/extract", json={"text": text}).json()["entities"]
        assert [e["text"] for e in extracted] == ["Jane Doe"]
        redacted = test_client.post("/redact", json={"text": text}).json()
        assert redacted["redacted_text"] == "Signed by [PERSON_1] yesterday."
        assert redacted["replacements"][0]["span"]["text"] == "Jane Doe Smith"
