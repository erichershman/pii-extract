"""Token-budget arithmetic shared by the transformer backends.

Both GLiNER generations split text into words and then feed each word to a
subword tokenizer. The encoder's limit is in subwords, and PII-dense text is
exactly where the two diverge (an email address is one or a few words but many
subwords), so windows are budgeted in subwords.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def subword_costs(tokenizer: Any, words: Sequence[str]) -> list[int]:
    """Subword count of each word, tokenized on its own as the models do.

    Because the models tokenize word by word, these costs are exactly additive
    across a window. A word the tokenizer maps to nothing still occupies a
    slot in the model's word sequence, so the minimum cost is 1.
    """
    if not words:
        return []
    encoded = tokenizer(list(words), add_special_tokens=False)["input_ids"]
    return [max(1, len(ids)) for ids in encoded]


def label_prompt_cost(tokenizer: Any, labels: Sequence[str]) -> int:
    """Subwords the label prompt consumes: each label plus its marker token."""
    return sum(cost + 1 for cost in subword_costs(tokenizer, labels))
