"""End-to-end offset correctness, with a deterministic fake model.

These tests exercise the real tokenizer->window->remap->merge path; only the
model's forward pass is replaced.
"""

import random

import pytest

from fakes import ALL_LABELS, CARD, EMAIL, FULL_CARD, PERSON, PHONE, FakeBackend
from pii_extract.pipeline import Pipeline
from pii_extract.redaction import PlaceholderStrategy
from pii_extract.spans import verify_spans


def pipeline(budget: int = 12, overlap: int = 4, **kwargs) -> Pipeline:
    return Pipeline(FakeBackend(budget), labels=ALL_LABELS, overlap_tokens=overlap, **kwargs)


def found(spans) -> list[tuple[str, str]]:
    return [(s.label, s.text) for s in spans]


def occurrences(text: str, value: str) -> list[int]:
    starts, at = [], text.find(value)
    while at != -1:
        starts.append(at)
        at = text.find(value, at + 1)
    return starts


def test_short_text_single_window():
    text = "Contact Jane Doe at jane.doe@example.com or 555-867-5309. Card: 4111 1111 1111 1111."
    pipe = pipeline(budget=64)
    assert len(pipe.plan(text)) == 1
    spans = pipe.extract(text)
    assert found(spans) == [
        (PERSON, "Jane Doe"),
        (EMAIL, "jane.doe@example.com"),
        (PHONE, "555-867-5309"),
        (CARD, FULL_CARD),
    ]
    assert [s.start for s in spans] == [text.index(s.text) for s in spans]


def test_empty_and_whitespace_only_text_never_reaches_the_model():
    pipe = pipeline()
    assert pipe.extract("") == []
    assert pipe.extract("   \r\n\t  ") == []
    assert pipe.backend.calls == []


def test_entity_split_across_a_window_boundary_is_recovered_whole():
    # 8 filler words, then the card. With a budget of 10 the first window ends
    # after "4111 1111": the model sees half a card number there.
    text = "aa bb cc dd ee ff gg hh 4111 1111 1111 1111 ii jj kk ll"
    pipe = pipeline(budget=10, overlap=4)

    windows = pipe.plan(text)
    assert len(windows) >= 2
    first = windows[0].slice(text)
    assert first.endswith("4111 1111") and windows[0].cut_end, "fixture must cut the card"

    raw = pipe.backend.predict([w.slice(text) for w in windows], [CARD], 0.5)
    assert [s.text for s in raw[0]] == ["4111 1111"], "window 0 reports only a fragment"
    assert raw[0][0].score > max(s.score for s in raw[1]), "and scores it higher"

    spans = pipe.extract(text, labels=[CARD])
    assert found(spans) == [(CARD, FULL_CARD)]
    assert (spans[0].start, spans[0].end) == (text.index(FULL_CARD), text.index(FULL_CARD) + 19)
    assert text[spans[0].start : spans[0].end] == FULL_CARD


def test_split_name_is_recovered_whole():
    text = "w1 w2 w3 w4 w5 w6 w7 Jane Doe signed the form on behalf of the team today"
    pipe = pipeline(budget=8, overlap=3)
    assert pipe.plan(text)[0].slice(text).endswith("Jane")
    assert found(pipe.extract(text, labels=[PERSON])) == [(PERSON, "Jane Doe")]


def test_without_overlap_a_split_entity_is_only_seen_in_pieces():
    # Documents WHY overlap exists. With none, no window sees the whole card:
    # the first reports the leading half and the second cannot recognise the
    # trailing "1111 1111" at all, which would leak in a redaction.
    text = "aa bb cc dd ee ff gg hh 4111 1111 1111 1111 ii jj kk ll"
    spans = pipeline(budget=10, overlap=0).extract(text, labels=[CARD])
    assert [s.text for s in spans] == ["4111 1111"]


def test_entities_in_overlap_regions_are_reported_once():
    filler = "lorem ipsum dolor sit amet"
    text = " ".join(f"{filler} call 555-{100 + i}-{1000 + i} now" for i in range(40))
    pipe = pipeline(budget=10, overlap=5)
    assert len(pipe.plan(text)) > 20
    spans = pipe.extract(text, labels=[PHONE])
    assert [s.text for s in spans] == [f"555-{100 + i}-{1000 + i}" for i in range(40)]
    assert len({s.key for s in spans}) == 40
    verify_spans(text, spans)


MESSY = (
    "﻿Name:  Jane   Doe\r\n"
    "\tE-mail: jane.doe@example.com — tel. 555-867-5309\r\n\r\n"
    "não \U0001f600\U0001f600 café ​ zero​width  Card:   4111 1111 1111 1111\n"
    "שלום John\tRoe 你好 john.roe@example.org\x0b\x0c 555-010-0199   \n"
)


@pytest.mark.parametrize("budget,overlap", [(200, 0), (40, 8), (12, 6), (10, 5), (8, 4)])
def test_offsets_round_trip_through_bom_crlf_nbsp_emoji_and_combining_marks(budget, overlap):
    original = MESSY
    spans = pipeline(budget, overlap).extract(original)

    assert original == MESSY, "input must never be modified"
    verify_spans(original, spans)
    for span in spans:
        assert original[span.start : span.end] == span.text

    by_text = {s.text: s for s in spans}
    for value in [
        "jane.doe@example.com",
        "555-867-5309",
        FULL_CARD,
        "john.roe@example.org",
        "555-010-0199",
    ]:
        assert by_text[value].start == original.index(value)
    # Names separated by odd whitespace keep that whitespace verbatim.
    assert "Jane   Doe" in by_text and "John\tRoe" in by_text
    assert by_text["Jane   Doe"].start == original.index("Jane   Doe") == 8


def test_bom_shifts_offsets_by_exactly_one():
    body = "mail jane.doe@example.com now"
    plain = pipeline().extract(body, labels=[EMAIL])
    with_bom = pipeline().extract("﻿" + body, labels=[EMAIL])
    assert [(s.start + 1, s.end + 1) for s in plain] == [(s.start, s.end) for s in with_bom]


def test_crlf_and_lf_documents_differ_only_by_the_extra_characters():
    lf = "line one\n555-867-5309\nline three\njane.doe@example.com\n"
    crlf = lf.replace("\n", "\r\n")
    for text in (lf, crlf):
        spans = pipeline(budget=4, overlap=2).extract(text)
        assert [(s.start, s.text) for s in spans] == [
            (text.index("555-867-5309"), "555-867-5309"),
            (text.index("jane.doe@example.com"), "jane.doe@example.com"),
        ]


def test_random_documents_round_trip_exactly():
    rng = random.Random(20261116)
    gaps = [" ", "  ", "\n", "\r\n", "\t", " ", "   ", "\n\n"]
    fillers = ["lorem", "ipsum", "café", "não", "\U0001f600", "x", "the", "你好", "--"]
    for _ in range(150):
        parts: list[str] = [rng.choice(["", "﻿", "  ", "\r\n"])]
        expected: list[tuple[int, int, str, str]] = []
        for i in range(rng.randint(5, 60)):
            for _ in range(rng.randint(0, 9)):
                parts += [rng.choice(fillers), rng.choice(gaps)]
            label, value = rng.choice(
                [
                    (EMAIL, f"user{i}@example.com"),
                    (PHONE, f"555-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}"),
                    (CARD, FULL_CARD),
                    (PERSON, "Jane" + rng.choice([" ", "  ", "\t"]) + "Doe"),
                ]
            )
            start = sum(map(len, parts))
            expected.append((start, start + len(value), label, value))
            parts += [value, rng.choice(gaps)]
        text = "".join(parts)

        # An overlap of at least the longest entity (4 tokens) guarantees that
        # some window sees every entity whole. Overlap is capped at budget // 2.
        overlap = rng.randint(4, 8)
        budget = rng.randint(2 * overlap, 40)
        spans = pipeline(budget, overlap).extract(text)

        verify_spans(text, spans)
        assert [(s.start, s.end, s.label, s.text) for s in spans] == expected


def test_window_tokens_setting_can_only_shrink_the_model_budget():
    text = " ".join(["word"] * 50)
    assert len(Pipeline(FakeBackend(20), overlap_tokens=0).plan(text)) == 3
    assert len(Pipeline(FakeBackend(20), overlap_tokens=0, window_tokens=10).plan(text)) == 5
    assert len(Pipeline(FakeBackend(20), overlap_tokens=0, window_tokens=500).plan(text)) == 3


def test_oversized_overlap_is_capped_instead_of_failing():
    text = " ".join(["word"] * 100)
    windows = Pipeline(FakeBackend(20), overlap_tokens=64).plan(text)
    assert windows[1].token_start == 10  # overlap capped at half the window


def test_labels_and_threshold_validation():
    pipe = pipeline()
    with pytest.raises(ValueError, match="threshold"):
        pipe.extract("x", threshold=1.5)
    with pytest.raises(ValueError, match="label"):
        Pipeline(FakeBackend(), labels=()).extract("x")
    with pytest.raises(ValueError, match="window budget"):
        Pipeline(FakeBackend(budget=0)).extract("x")
    with pytest.raises(ValueError):
        Pipeline(FakeBackend(), window_tokens=0)


def test_request_labels_override_defaults_and_are_deduplicated():
    text = "Jane Doe 555-867-5309"
    pipe = pipeline()
    assert found(pipe.extract(text, labels=[PHONE, PHONE])) == [(PHONE, "555-867-5309")]
    assert found(pipe.extract(text)) == [(PERSON, "Jane Doe"), (PHONE, "555-867-5309")]


def test_threshold_is_passed_to_the_backend():
    assert pipeline().extract("555-867-5309", threshold=0.95) == []


def test_redaction_is_consistent_across_windows():
    # The same phone number appears in the first and the last window.
    middle = " ".join(f"filler{i}" for i in range(60))
    text = f"Call 555-867-5309 or 555-010-0199. {middle} Again: 555-867-5309, Jane Doe."
    pipe = pipeline(budget=10, overlap=4)
    assert len(pipe.plan(text)) > 5

    result = pipe.redact(text, PlaceholderStrategy())
    assert result.text == f"Call [PHONE_1] or [PHONE_2]. {middle} Again: [PHONE_1], [PERSON_1]."
    for item in result.replacements:
        assert text[item.span.start : item.span.end] == item.span.text
        assert result.text[item.redacted_start : item.redacted_end] == item.replacement
