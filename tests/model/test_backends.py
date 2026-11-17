"""Model-dependent tests: real weights, synthetic inputs only."""

from __future__ import annotations

import itertools
import threading

from pii_extract.pipeline import Pipeline
from pii_extract.redaction import PlaceholderStrategy
from pii_extract.spans import verify_spans

LABELS = ["person", "email", "phone number", "credit card number"]
SPEC_EXAMPLE = (
    "Contact Jane Doe at jane.doe@example.com or 555-867-5309. Card on file: 4111 1111 1111 1111."
)
FILLER = "The quarterly report was reviewed by the committee and no action was required. "


def by_label(spans) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for span in spans:
        out.setdefault(span.label, []).append(span.text)
    return out


def test_detects_the_spec_example(backend):
    spans = Pipeline(backend, labels=LABELS).extract(SPEC_EXAMPLE)
    verify_spans(SPEC_EXAMPLE, spans)
    assert by_label(spans) == {
        "person": ["Jane Doe"],
        "email": ["jane.doe@example.com"],
        "phone number": ["555-867-5309"],
        "credit card number": ["4111 1111 1111 1111"],
    }
    assert [s.start for s in spans] == [SPEC_EXAMPLE.index(s.text) for s in spans]


def test_tokens_align_with_the_original_text(backend):
    text = "﻿Name:  Jane   Doe\r\n\tE-MAIL: JANE.DOE@EXAMPLE.COM — (555) 867-5309 café"
    tokens = backend.tokenize(text)
    assert tokens, "expected some tokens"
    previous_end = 0
    for token in tokens:
        assert previous_end <= token.start < token.end <= len(text)
        assert not text[token.start : token.end].isspace()
        assert token.cost >= 1
        previous_end = token.end
    # Everything between tokens is whitespace or characters the splitter drops;
    # no letter or digit may fall through the cracks.
    covered = set()
    for token in tokens:
        covered.update(range(token.start, token.end))
    assert all(i in covered for i, ch in enumerate(text) if ch.isalnum())


def test_window_budget_is_within_the_encoder_limit(backend):
    budget = backend.window_budget(LABELS)
    assert 128 < budget < 512
    assert backend.window_budget(LABELS * 30) < budget, "more labels leave less room for text"
    if backend.name == "gliner":
        assert budget <= 384, "must stay under GLiNER's silent word-truncation cap"


def test_offsets_survive_bom_crlf_nbsp_and_case(backend):
    text = (
        "﻿Name:  Jane   Doe\r\n"
        "\tE-MAIL: JANE.DOE@EXAMPLE.COM  — tel. (555) 867-5309\r\n"
        "İstanbul café \U0001f600 card 4111 1111 1111 1111\r\n"
    )
    spans = Pipeline(backend, labels=LABELS, threshold=0.3).extract(text)
    verify_spans(text, spans)
    found = by_label(spans)
    assert "JANE.DOE@EXAMPLE.COM" in found["email"]
    assert "Jane   Doe" in found["person"], "inner whitespace must be preserved verbatim"
    email = next(s for s in spans if s.label == "email")
    assert email.start == text.index("JANE.DOE@EXAMPLE.COM")


def test_long_document_is_windowed_and_every_entity_is_found_once(backend):
    parts, expected_emails, expected_phones = [], [], []
    for i in range(60):
        parts.append(FILLER * 2)
        email, phone = f"agent{i}@example.com", f"555-{200 + i}-{1000 + i}"
        parts.append(f"Please reach Jane Doe at {email} or call {phone}. ")
        expected_emails.append(email)
        expected_phones.append(phone)
    text = "".join(parts)

    pipe = Pipeline(backend, labels=LABELS)
    windows = pipe.plan(text)
    assert len(windows) > 3, "the document must exceed one window"
    budget = backend.window_budget(LABELS)
    tokens = backend.tokenize(text)
    for window in windows:
        assert sum(t.cost for t in tokens[window.token_start : window.token_end]) <= budget

    spans = pipe.extract(text)
    verify_spans(text, spans)
    found = by_label(spans)
    # Phones sit in every window overlap; each must come back exactly once.
    assert found["phone number"] == expected_phones
    if backend.name == "gliner2":
        assert found["email"] == expected_emails
    # (urchade/gliner_multi_pii-v1 tags the local part of these synthetic
    # addresses as a person in some contexts. That is model quality, which the
    # benchmark measures; this test is about windowing.)
    assert len({s.key for s in spans}) == len(spans)
    for left, right in itertools.pairwise(spans):
        assert left.end <= right.start


def test_entity_split_across_a_window_boundary(backend):
    # Small windows so the card number is cut in half by the first window.
    lead = "Thank you for calling the support line about the recent order today . "
    text = lead + "The card number on file is 4111 1111 1111 1111 and it expires soon. " + FILLER
    card = "4111 1111 1111 1111"
    card_start = text.index(card)

    tokens = backend.tokenize(text)
    split_index = next(i for i, t in enumerate(tokens) if t.start >= card_start + 10)
    budget = sum(t.cost for t in tokens[:split_index])
    pipe = Pipeline(backend, labels=LABELS, window_tokens=budget, overlap_tokens=budget // 2)

    first = pipe.plan(text)[0]
    assert card_start < first.end < card_start + len(card), "window 0 must cut the card"

    cards = [s for s in pipe.extract(text) if s.label == "credit card number"]
    assert [(s.start, s.text) for s in cards] == [(card_start, card)]


def test_redaction_end_to_end(backend):
    text = SPEC_EXAMPLE + " Call 555-867-5309 again."
    result = Pipeline(backend, labels=LABELS).redact(text, PlaceholderStrategy())
    assert result.text == (
        "Contact [PERSON_1] at [EMAIL_1] or [PHONE_1]. "
        "Card on file: [CREDIT_CARD_1]. Call [PHONE_1] again."
    )


def test_concurrent_requests_on_one_session_agree(backend):
    pipe = Pipeline(backend, labels=LABELS)
    texts = [SPEC_EXAMPLE.replace("5309", f"{5300 + i}") for i in range(8)]
    expected = [pipe.extract(t) for t in texts]
    results: list = [None] * len(texts)
    errors: list[BaseException] = []

    def worker(i: int):
        try:
            for _ in range(3):
                results[i] = pipe.extract(texts[i])
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(texts))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert [[s.key for s in r] for r in results] == [[s.key for s in e] for e in expected]


def test_close_and_reload(gliner2_backend):
    # Runs on the default backend only; reloading twice would just be slower.
    pipe = Pipeline(gliner2_backend, labels=LABELS)
    before = pipe.extract(SPEC_EXAMPLE)
    gliner2_backend.close()
    assert not gliner2_backend.loaded
    after = pipe.extract(SPEC_EXAMPLE)  # lazy reload
    assert gliner2_backend.loaded
    assert [s.key for s in after] == [s.key for s in before]


def test_text_without_final_punctuation_keeps_offsets_in_range(backend):
    # gliner2 appends "." to unterminated input; a span ending at the very end
    # of the text must not swallow that synthetic character.
    texts = [
        "Reference code for the customer record: ZZ 12 34 56 T",
        "Card on file: 4111 1111 1111 1111",
        "Call Jane Doe on 555-867-5309",
        "Write to jane.doe@example.com",
    ]
    labels = [*LABELS, "reference code"]
    for text in texts:
        spans = Pipeline(backend, labels=labels, threshold=0.3).extract(text)
        verify_spans(text, spans)
        assert spans, f"expected a detection in {text!r}"
        assert all(s.end <= len(text) for s in spans)
        assert not any(s.text.endswith(".") for s in spans)


def test_validators_drop_only_malformed_detections(backend):
    # 4111 1111 1111 1112 has a card's shape but fails the Luhn check.
    text = (
        "Primary card 4111 1111 1111 1111, backup card 4111 1111 1111 1112. "
        "Contact Jane Doe at jane.doe@example.com or 555-867-5309."
    )
    off = Pipeline(backend, labels=LABELS)
    on = Pipeline(backend, labels=LABELS, validators=["all"])

    unchecked = by_label(off.extract(text))
    checked = by_label(on.extract(text))

    # Whatever card numbers the model reported, only the Luhn-valid one survives.
    # (urchade/gliner_multi_pii-v1 reports neither card in this sentence.)
    assert "4111 1111 1111 1112" not in checked.get("credit card number", [])
    if backend.name == "gliner2":
        assert unchecked["credit card number"] == ["4111 1111 1111 1111", "4111 1111 1111 1112"]
        assert checked["credit card number"] == ["4111 1111 1111 1111"]
    assert checked["email"] == unchecked["email"] == ["jane.doe@example.com"]
    assert checked["phone number"] == unchecked["phone number"] == ["555-867-5309"]
    assert checked["person"] == unchecked["person"], "labels without a validator are untouched"
    # Per-call switches override the configured default in both directions.
    assert by_label(off.extract(text, validate=True)) == checked
    assert by_label(on.extract(text, validate=False)) == unchecked
