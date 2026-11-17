"""Sliding-window planning over a tokenized text.

This stage is pure arithmetic on token offsets. It does not tokenize (the
backend supplies tokens from the model's own splitter), does not run
inference, and does not merge results.

Offset contract
---------------
A window's text is ``original[window.start:window.end]``: an exact slice of
the caller's string, never stripped, joined, or normalized. A span the model
reports at ``[s, e)`` inside that slice therefore sits at
``[window.start + s, window.start + e)`` in the original. That single
addition is the whole offset mapping, and it stays correct regardless of
BOMs, CRLF, non-breaking spaces, or runs of whitespace, because those
characters are simply part of the slice.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Token:
    """A model token (a word, for GLiNER-family models) in original-text offsets.

    ``cost`` is how much of the window budget the token consumes. Backends set
    it to the token's subword count so that windows respect the encoder's real
    limit; a plain word-count budget is the special case ``cost == 1``.
    """

    start: int
    end: int
    cost: int = 1


@dataclass(frozen=True, slots=True)
class Window:
    """One slice of the original text, bounded by token edges.

    ``cut_start`` / ``cut_end`` record whether the window boundary falls inside
    the document (as opposed to at its first or last token). A span touching a
    cut edge may be a truncated fragment of a longer entity; dedup uses this.
    """

    index: int
    start: int
    end: int
    token_start: int
    token_end: int
    cut_start: bool
    cut_end: bool

    def slice(self, text: str) -> str:
        return text[self.start : self.end]


def plan_windows(tokens: Sequence[Token], max_tokens: int, overlap: int) -> list[Window]:
    """Cover ``tokens`` with overlapping windows of at most ``max_tokens`` budget.

    * Windows never split a token.
    * Consecutive windows share a suffix/prefix of at most ``overlap`` budget.
      An entity up to ``overlap`` tokens long that straddles one window's edge
      is therefore seen whole by the neighbouring window.
    * A single token costing more than ``max_tokens`` gets a window to itself
      (it cannot be split; the model will see it and handle it as it can).
    * Every token is in at least one window, and each window starts strictly
      after the previous one, so planning always terminates.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= max_tokens:
        raise ValueError("overlap must be smaller than max_tokens")

    windows: list[Window] = []
    n = len(tokens)
    first = 0
    while first < n:
        # Grow the window until the budget is spent (always take one token).
        last = first
        spent = 0
        while last < n and (last == first or spent + tokens[last].cost <= max_tokens):
            spent += tokens[last].cost
            last += 1

        windows.append(
            Window(
                index=len(windows),
                start=tokens[first].start,
                end=tokens[last - 1].end,
                token_start=first,
                token_end=last,
                cut_start=first > 0,
                cut_end=last < n,
            )
        )
        if last == n:
            break

        # Step back from the cut to build the overlap, but always advance by
        # at least one token relative to this window's start.
        next_first = last
        shared = 0
        while next_first - 1 > first and shared + tokens[next_first - 1].cost <= overlap:
            next_first -= 1
            shared += tokens[next_first].cost
        first = next_first
    return windows
