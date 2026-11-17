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


def test_rule3_nested_spans_higher_score_wins_and_nothing_grows():
    email = span(TEXT, "jane.doe@example.com", EMAIL, score=0.99)
    person = span(TEXT, "jane.doe", PERSON, score=0.6, occurrence=0)
    assert merged(email, person) == [email]


def test_rule3_winner_inside_loser_keeps_its_label_and_covers_the_loser():
    email = span(TEXT, "jane.doe@example.com", EMAIL, score=0.6)
    person = span(TEXT, "jane.doe", PERSON, score=0.999)
    (result,) = merged(email, person)
    assert (result.label, result.score) == (PERSON, 0.999)
    assert result.text == "jane.doe@example.com", "the weaker email span must stay covered"
    verify_spans(TEXT, [result])


def test_rule3_partial_overlap_winner_grows_to_the_right():
    # Regression: the loser used to be dropped whole, exposing "Smith".
    winner = span(TEXT, "Jane Doe", PERSON, score=0.9)
    loser = span(TEXT, "Doe Smith", "organization", score=0.6)
    (result,) = merged(winner, loser)
    assert (result.label, result.score, result.text) == (PERSON, 0.9, "Jane Doe Smith")
    assert (result.start, result.end) == (winner.start, loser.end)


def test_rule3_partial_overlap_winner_grows_to_the_left():
    winner = span(TEXT, "Doe Smith", "organization", score=0.9)
    loser = span(TEXT, "Jane Doe", PERSON, score=0.6)
    (result,) = merged(winner, loser)
    assert (result.label, result.text) == ("organization", "Jane Doe Smith")


def test_rule3_winner_strictly_inside_loser_grows_on_both_sides():
    winner = span(TEXT, "Doe", PERSON, score=0.95)
    loser = span(TEXT, "Jane Doe Smith", "organization", score=0.5)
    (result,) = merged(loser, winner)
    assert (result.label, result.score, result.text) == (PERSON, 0.95, "Jane Doe Smith")


def test_rule3_ties_go_to_the_longer_span_then_earlier_then_label():
    short = at(3, 11, PERSON, 0.9)
    long = at(3, 17, "organization", 0.9)
    assert merged(short, long) == [long]

    early = at(3, 11, PERSON, 0.9)
    late = at(9, 17, "organization", 0.9)
    (result,) = merged(late, early)
    assert (result.label, result.start, result.end) == (PERSON, 3, 17)

    a = at(3, 11, "alias", 0.9)
    b = at(3, 11, PERSON, 0.9)
    assert merged(b, a) == [a]


def test_rule3_clipped_span_loses_to_unclipped_span_of_another_label():
    card = span(TEXT, "4111 1111 1111 1111", CARD, score=0.8)
    bogus_phone = span(TEXT, "4111 1111", PHONE, score=0.99)
    assert merged(card, Candidate(bogus_phone, clipped=True)) == [card]


def test_rule3_leftover_between_two_winners_goes_to_the_stronger_one():
    # B loses to A and to C. Its middle, [11, 18), is covered by neither; it
    # goes to A (the stronger winner), and C is left exactly as detected.
    a = at(0, 11, PERSON, 0.99)
    b = at(8, 20, "organization", 0.9)
    c = at(18, 27, "account", 0.8)
    result = merged(a, b, c)
    assert [(s.label, s.start, s.end) for s in result] == [(PERSON, 0, 18), ("account", 18, 27)]
    assert result[0].score == 0.99 and result[1] == c
    verify_spans(TEXT, result)

    # Same shape with the strengths reversed: now C takes the middle.
    a, c = at(0, 11, PERSON, 0.8), at(18, 27, "account", 0.99)
    result = merged(a, b, c)
    assert [(s.label, s.start, s.end) for s in result] == [(PERSON, 0, 11), ("account", 11, 27)]


def test_rule3_two_losers_never_make_the_winners_overlap():
    # Both losers leave characters in [11, 18); each is absorbed against the
    # winners as already grown, so the second finds nothing left to hand out.
    a = at(0, 11, PERSON, 0.99)
    c = at(18, 27, "account", 0.98)
    first = at(8, 16, "organization", 0.7)  # leftover [11, 16) -> A
    second = at(13, 22, "alias", 0.6)  # leftover [16, 18)  -> between A and C -> A
    result = merged(a, c, first, second)
    assert [(s.label, s.start, s.end) for s in result] == [(PERSON, 0, 18), ("account", 18, 27)]


def test_rule3_loser_sticking_out_past_the_winner_on_the_document_edge_side():
    winner = at(3, 11, PERSON, 0.9)
    loser = at(0, 6, "title", 0.4)  # "Dr Jan": starts before the winner
    (result,) = merged(winner, loser)
    assert (result.label, result.start, result.end, result.text) == (PERSON, 0, 11, "Dr Jane Doe")


def test_same_label_union_then_cross_label_absorption():
    # Rule 2 builds "Jane Doe Smith" (person); rule 3 then lets it absorb the
    # tail of a weaker, differently labelled span.
    left = span(TEXT, "Jane Doe", PERSON, score=0.8)
    right = span(TEXT, "Doe Smith", PERSON, score=0.95)
    other = span(TEXT, "Smith paid", "organization", score=0.5)
    (result,) = merged(left, right, other)
    assert (result.label, result.text) == (PERSON, "Jane Doe Smith paid")


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

        # The merge never invents coverage: every output character was flagged.
        flagged = {i for c in candidates for i in range(c.span.start, c.span.end)}
        covered = {i for s in result for i in range(s.start, s.end)}
        assert covered <= flagged

        # Rule 1 is the only rule that can discard flagged characters (it keeps
        # the best duplicate's boundaries). With iou_threshold=1.0 it applies
        # to exact duplicates only, and then coverage is complete.
        strict = merge_spans(candidates, TEXT, iou_threshold=1.0)
        assert {i for s in strict for i in range(s.start, s.end)} == flagged
        for left, right in itertools.pairwise(strict):
            assert left.end <= right.start

        shuffled = candidates[:]
        rng.shuffle(shuffled)
        assert merge_spans(shuffled, TEXT) == result


def test_large_inputs_stay_fast_because_only_neighbours_are_compared():
    # 30,000 candidates: 10,000 entities, each seen by three overlapping
    # windows. A quadratic merge would need ~10^9 comparisons here.
    unit = "555-867-5309 "
    text = unit * 10_000
    candidates = [
        Candidate(
            Span(start=i * 13, end=i * 13 + 12, label=PHONE, score=score, text="555-867-5309")
        )
        for i in range(10_000)
        for score in (0.7, 0.9, 0.8)
    ]
    result = merge_spans(candidates, text)
    assert len(result) == 10_000
    assert all(s.score == 0.9 for s in result)
    assert [s.start for s in result[:3]] == [0, 13, 26]


def test_only_rule1_discards_characters_and_iou_one_turns_that_off():
    best = span(TEXT, "Jane Doe", PERSON, score=0.95)
    wider = span(TEXT, "Jane Doe Smith", PERSON, score=0.6)
    other = span(TEXT, "Smith paid", "organization", score=0.5)
    # Default: the two person spans are duplicates, so "Jane Doe" keeps its
    # boundaries. The organization span no longer touches a person span.
    assert [(s.label, s.text) for s in merged(best, wider, other)] == [
        (PERSON, "Jane Doe"),
        ("organization", "Smith paid"),
    ]
    # Strict coverage: nothing any candidate flagged may be dropped.
    (strict,) = merged(best, wider, other, iou_threshold=1.0)
    assert (strict.label, strict.text) == (PERSON, "Jane Doe Smith paid")
