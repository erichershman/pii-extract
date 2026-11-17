import random
import re

import pytest

from pii_extract.chunking import Token, plan_windows


def words(text: str, cost: int = 1) -> list[Token]:
    return [Token(m.start(), m.end(), cost) for m in re.finditer(r"\S+", text)]


def test_no_tokens_means_no_windows():
    assert plan_windows([], max_tokens=8, overlap=2) == []


def test_text_that_fits_is_one_uncut_window():
    text = "  Jane Doe called 555-867-5309  "
    (window,) = plan_windows(words(text), max_tokens=10, overlap=2)
    assert window.slice(text) == "Jane Doe called 555-867-5309"
    assert (window.cut_start, window.cut_end) == (False, False)
    assert (window.token_start, window.token_end) == (0, 4)


def test_windows_slide_with_the_requested_overlap():
    text = " ".join(f"w{i}" for i in range(10))
    windows = plan_windows(words(text), max_tokens=4, overlap=1)
    assert [(w.token_start, w.token_end) for w in windows] == [(0, 4), (3, 7), (6, 10)]
    assert [w.slice(text) for w in windows] == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]
    assert [(w.cut_start, w.cut_end) for w in windows] == [
        (False, True),
        (True, True),
        (True, False),
    ]
    assert [w.index for w in windows] == [0, 1, 2]


def test_zero_overlap_partitions_the_tokens():
    tokens = words(" ".join("x" * 9))
    windows = plan_windows(tokens, max_tokens=4, overlap=0)
    assert [(w.token_start, w.token_end) for w in windows] == [(0, 4), (4, 8), (8, 9)]


def test_budget_counts_token_cost_not_token_count():
    # An email is one word but many subwords; the budget must see the subwords.
    tokens = [Token(0, 3, 1), Token(4, 24, 8), Token(25, 28, 1), Token(29, 32, 1)]
    windows = plan_windows(tokens, max_tokens=9, overlap=0)
    assert [(w.token_start, w.token_end) for w in windows] == [(0, 2), (2, 4)]


def test_oversized_token_gets_its_own_window():
    tokens = [Token(0, 2, 1), Token(3, 500, 50), Token(501, 503, 1)]
    windows = plan_windows(tokens, max_tokens=10, overlap=2)
    assert [(w.token_start, w.token_end) for w in windows] == [(0, 1), (1, 2), (2, 3)]


def test_overlap_is_measured_in_cost():
    tokens = [Token(i * 2, i * 2 + 1, cost) for i, cost in enumerate([1, 1, 1, 5, 1, 1, 1])]
    windows = plan_windows(tokens, max_tokens=8, overlap=3)
    # First window: 1+1+1+5 = 8. Stepping back, the cost-5 token does not fit
    # in an overlap of 3, so the next window starts right after it.
    assert [(w.token_start, w.token_end) for w in windows] == [(0, 4), (4, 7)]


@pytest.mark.parametrize("max_tokens,overlap", [(0, 0), (-1, 0), (4, -1), (4, 4), (4, 5)])
def test_invalid_sizes_are_rejected(max_tokens, overlap):
    with pytest.raises(ValueError):
        plan_windows(words("a b c"), max_tokens=max_tokens, overlap=overlap)


def test_window_text_is_an_exact_slice_of_messy_input():
    text = "﻿Name:  Jane   Doe\r\n\tmail jane.doe@example.com \r\n\r\n tel 555-867-5309\n"
    tokens = words(text)
    for window in plan_windows(tokens, max_tokens=3, overlap=1):
        piece = window.slice(text)
        assert piece == text[window.start : window.end]
        assert piece[0] == text[tokens[window.token_start].start]
        assert window.start == tokens[window.token_start].start
        assert window.end == tokens[window.token_end - 1].end
        assert not piece[0].isspace() and not piece[-1].isspace()


def test_invariants_hold_for_random_inputs():
    rng = random.Random(1234)
    for _ in range(300):
        n = rng.randint(1, 120)
        tokens, position = [], rng.randint(0, 3)
        for _ in range(n):
            length = rng.randint(1, 12)
            tokens.append(Token(position, position + length, rng.randint(1, 9)))
            position += length + rng.randint(1, 4)
        max_tokens = rng.randint(1, 40)
        overlap = rng.randint(0, max_tokens - 1)

        windows = plan_windows(tokens, max_tokens, overlap)

        covered = set()
        for i, window in enumerate(windows):
            members = tokens[window.token_start : window.token_end]
            assert members, "windows are never empty"
            cost = sum(t.cost for t in members)
            assert cost <= max_tokens or len(members) == 1
            assert window.start == members[0].start and window.end == members[-1].end
            assert window.cut_start == (window.token_start > 0)
            assert window.cut_end == (window.token_end < n)
            covered.update(range(window.token_start, window.token_end))
            if i:
                previous = windows[i - 1]
                assert window.token_start > previous.token_start, "must advance"
                assert window.token_start <= previous.token_end, "no token skipped"
                shared = tokens[window.token_start : previous.token_end]
                assert sum(t.cost for t in shared) <= overlap
        assert covered == set(range(n))
        assert windows[-1].token_end == n
