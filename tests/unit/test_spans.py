import pytest
from pydantic import ValidationError

from pii_extract.spans import OffsetError, Span, iou, span_from_offsets, verify_spans

TEXT = "Call 555-867-5309 now"


def make(start: int, end: int, label: str = "phone number", score: float = 0.9) -> Span:
    return span_from_offsets(TEXT, start, end, label, score)


def test_span_text_is_read_from_the_source():
    span = make(5, 17)
    assert span.text == "555-867-5309"
    assert span.key == (5, 17, "phone number")
    assert span.length == 12


@pytest.mark.parametrize("start,end", [(5, 5), (9, 4), (-1, 3), (0, len(TEXT) + 1)])
def test_out_of_range_or_empty_offsets_are_rejected(start, end):
    with pytest.raises(OffsetError):
        span_from_offsets(TEXT, start, end, "phone number", 0.9)


def test_span_rejects_text_of_the_wrong_length():
    with pytest.raises(ValidationError):
        Span(start=0, end=4, label="x", score=0.5, text="toolong")


def test_span_is_immutable():
    with pytest.raises(ValidationError):
        make(5, 17).start = 0


def test_scores_are_clamped_into_unit_interval():
    assert span_from_offsets(TEXT, 0, 4, "x", 1.0000001).score == 1.0
    assert span_from_offsets(TEXT, 0, 4, "x", -0.1).score == 0.0


def test_shifted_moves_offsets_and_keeps_text():
    moved = make(5, 17).shifted(100)
    assert (moved.start, moved.end, moved.text) == (105, 117, "555-867-5309")


def test_verify_spans_detects_drift():
    good = make(5, 17)
    verify_spans(TEXT, [good])
    with pytest.raises(OffsetError):
        verify_spans(TEXT, [good.shifted(1)])
    with pytest.raises(OffsetError):
        verify_spans("short", [good])


def test_iou():
    assert iou(make(0, 10), make(0, 10)) == 1.0
    assert iou(make(0, 10), make(5, 15)) == pytest.approx(5 / 15)
    assert iou(make(0, 5), make(5, 10)) == 0.0  # touching is not overlapping
    assert iou(make(0, 4), make(10, 12)) == 0.0


def test_overlaps_is_half_open():
    assert make(0, 5).overlaps(make(4, 8))
    assert not make(0, 5).overlaps(make(5, 8))
