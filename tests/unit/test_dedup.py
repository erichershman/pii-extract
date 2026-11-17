import itertools
import random

import pytest

from fakes import CARD, EMAIL, PERSON, PHONE, span
from pii_extract.dedup import Candidate, merge_spans
from pii_extract.spans import Span, verify_spans

TEXT = "Dr Jane Doe Smith paid with 4111 1111 1111 1111, mail jane.doe@example.com, 555-867-5309."


def at(start: int, end: int, label: str, score: float = 0.9) -> Span:
    return Span(start=start, end=end, label=label, score=score, text=TEXT[start:end])


def merged(*candidates: Candidate | Span, iou_threshold: float = 0.5) -> list[Span]:
    wrapped = [c if isinstance(c, Candidate) else Candidate(c) for c in candidates]
    return merge_spans(wrapped, TEXT, iou_threshold)


def test_empty_input():
    assert merge_spans([], TEXT) == []


def test_disjoint_spans_are_all_kept_in_document_order():
    phone = span(TEXT, "555-867-5309", PHONE)
    email = span(TEXT, "jane.doe@example.com", EMAIL)
    person = span(TEXT, "Jane Doe", PERSON)
    assert merged(phone, email, person) == [person, email, phone]


def test_identical_detections_from_two_windows_collapse_to_the_best_score():
    low = span(TEXT, "555-867-5309", PHONE, score=0.7)
    high = span(TEXT, "555-867-5309", PHONE, score=0.95)
    assert merged(low, high) == [high]


def test_rule1_near_duplicates_keep_the_best_boundaries_not_their_union():
    tight = span(TEXT, "jane.doe@example.com", EMAIL, score=0.99)
    sloppy = span(TEXT, "jane.doe@example.com,", EMAIL, score=0.6)  # trailing comma
    assert merged(tight, sloppy) == [tight]
    assert merged(sloppy, tight) == [tight]


def test_rule1_clipped_fragment_loses_even_with_a_higher_score():
    full = span(TEXT, "4111 1111 1111 1111", CARD, score=0.90)
    fragment = span(TEXT, "4111 1111 1111", CARD, score=0.99)
    # Without clipping information, score decides and the last group leaks.
    assert merged(full, fragment) == [fragment]
    # With it, the entity that was seen whole wins.
    assert merged(full, Candidate(fragment, clipped=True)) == [full]


def test_rule1_two_clipped_candidates_fall_back_to_score():
    a = span(TEXT, "4111 1111 1111", CARD, score=0.8)
    b = span(TEXT, "4111 1111 1111 1111", CARD, score=0.7)
    assert merged(Candidate(a, clipped=True), Candidate(b, clipped=True)) == [a]


def test_rule2_same_label_boundary_disagreement_becomes_the_union():
    left = span(TEXT, "Jane Doe", PERSON, score=0.8)
    right = span(TEXT, "Doe Smith", PERSON, score=0.95)
    (union,) = merged(left, right)
    assert union.text == "Jane Doe Smith"
    assert union.label == PERSON
    assert union.score == 0.95
    verify_spans(TEXT, [union])


def test_rule2_small_fragment_is_absorbed_by_the_span_containing_it():
    full = span(TEXT, "4111 1111 1111 1111", CARD, score=0.9)
    fragment = span(TEXT, "4111", CARD, score=0.99)  # IoU 4/19, far below threshold
    assert merged(Candidate(fragment, clipped=True), full) == [
        full.model_copy(update={"score": 0.99})
    ]


def test_rule2_unions_chain_transitively():
    a, b, c = at(3, 11, PERSON), at(8, 17, PERSON, 0.7), at(15, 17, PERSON, 0.6)
    (union,) = merged(a, b, c, iou_threshold=0.9)
    assert (union.start, union.end, union.text) == (3, 17, "Jane Doe Smith")


def test_iou_threshold_decides_between_duplicate_and_union():
    best = span(TEXT, "Jane Doe", PERSON, score=0.95)
    longer = span(TEXT, "Jane Doe Smith", PERSON, score=0.6)  # IoU = 8/14 = 0.571
    assert merged(best, longer, iou_threshold=0.5) == [best]  # duplicates: trust the best
    (union,) = merged(best, longer, iou_threshold=0.6)  # distinct: fail closed
    assert union.text == "Jane Doe Smith"


def test_rule3_different_labels_higher_score_wins():
    email = span(TEXT, "jane.doe@example.com", EMAIL, score=0.99)
    person = span(TEXT, "jane.doe", PERSON, score=0.6, occurrence=0)
    assert merged(email, person) == [email]
    stronger_person = person.model_copy(update={"score": 0.999})
    assert merged(email, stronger_person) == [stronger_person]


def test_rule3_ties_go_to_the_longer_span_then_earlier_then_label():
    short = at(3, 11, PERSON, 0.9)
    long = at(3, 17, "organization", 0.9)
    assert merged(short, long) == [long]

    early = at(3, 11, PERSON, 0.9)
    late = at(9, 17, "organization", 0.9)
    assert merged(late, early) == [early]

    a = at(3, 11, "alias", 0.9)
    b = at(3, 11, PERSON, 0.9)
    assert merged(b, a) == [a]


def test_rule3_clipped_span_loses_to_unclipped_span_of_another_label():
    card = span(TEXT, "4111 1111 1111 1111", CARD, score=0.8)
    bogus_phone = span(TEXT, "4111 1111", PHONE, score=0.99)
    assert merged(card, Candidate(bogus_phone, clipped=True)) == [card]


def test_rule3_a_dropped_span_does_not_suppress_others():
    # Greedy runs best-first: A is kept; B overlaps A and is dropped; C overlaps
    # only the dropped B, so it survives. A loser never takes anything with it.
    a = at(0, 11, PERSON, 0.99)
    b = at(8, 20, "organization", 0.9)
    c = at(18, 27, "account", 0.8)
    assert merged(a, b, c) == [a, c]


def test_clipped_span_with_no_rival_is_kept():
    fragment = span(TEXT, "4111 1111", CARD, score=0.8)
    assert merged(Candidate(fragment, clipped=True)) == [fragment]


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_iou_threshold_must_be_a_fraction(bad):
    with pytest.raises(ValueError):
        merge_spans([], TEXT, iou_threshold=bad)


def test_result_is_flat_sorted_and_independent_of_input_order():
    rng = random.Random(7)
    labels = [PERSON, EMAIL, PHONE, CARD]
    for _ in range(200):
        candidates = []
        for _ in range(rng.randint(1, 25)):
            start = rng.randrange(0, len(TEXT) - 1)
            end = rng.randint(start + 1, min(len(TEXT), start + 25))
            candidates.append(
                Candidate(
                    at(start, end, rng.choice(labels), round(rng.random(), 2)),
                    clipped=rng.random() < 0.3,
                )
            )
        result = merge_spans(candidates, TEXT)

        verify_spans(TEXT, result)
        assert result == sorted(result, key=lambda s: (s.start, s.end, s.label))
        for left, right in itertools.pairwise(result):
            assert left.end <= right.start, "output spans must not overlap"

        shuffled = candidates[:]
        rng.shuffle(shuffled)
        assert merge_spans(shuffled, TEXT) == result
