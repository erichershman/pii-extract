"""GLiNER2 backend (the ``gliner2`` package). Default model: GLiNER2-PII."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pii_extract.backends.base import ModelSession, checked_span
from pii_extract.backends.budget import label_prompt_cost, subword_costs
from pii_extract.chunking import Token
from pii_extract.spans import Span

# GLiNER2-PII. Its model card and this project's spec call it
# "fastino/gliner2-pii-v1", but no repository exists under that name; the
# weights are published as the id below. The alias keeps the spec's name usable.
DEFAULT_GLINER2_MODEL = "fastino/gliner2-privacy-filter-PII-multi"
MODEL_ALIASES = {"fastino/gliner2-pii-v1": DEFAULT_GLINER2_MODEL}

# Subword tokens the schema prompt spends besides the labels themselves
# ([CLS], [SEP], the "( [P] entities ( ... ) )" scaffolding, the text separator).
_PROMPT_OVERHEAD = 16


class Gliner2Backend:
    name = "gliner2"

    def __init__(
        self, model_id: str = DEFAULT_GLINER2_MODEL, device: str = "auto", batch_size: int = 8
    ) -> None:
        self.model_id = MODEL_ALIASES.get(model_id, model_id)
        self._batch_size = batch_size
        self._session: ModelSession[Any] = ModelSession(self._load_model, device)

    def _load_model(self, device: str) -> Any:
        from gliner2 import GLiNER2

        model = GLiNER2.from_pretrained(self.model_id)
        model.to(device)
        model.eval()
        return model

    @property
    def device(self) -> str:
        return self._session.device

    @property
    def loaded(self) -> bool:
        return self._session.loaded

    def load(self) -> None:
        self._session.load()

    def close(self) -> None:
        self._session.close()

    def tokenize(self, text: str) -> list[Token]:
        processor = self._session.load().processor
        # lower=True matches what the processor feeds the subword tokenizer.
        # The splitter lower-cases token values only; offsets index ``text``.
        words = list(processor.word_splitter(text, True))
        costs = subword_costs(processor.tokenizer, [word for word, _, _ in words])
        return [Token(start, end, cost) for (_, start, end), cost in zip(words, costs, strict=True)]

    def window_budget(self, labels: Sequence[str]) -> int:
        model = self._session.load()
        # GLiNER2 enforces no input cap of its own, so the limit is the
        # encoder's trained position range, read from its config.
        limit = int(model.encoder.config.max_position_embeddings)
        return limit - label_prompt_cost(model.processor.tokenizer, labels) - _PROMPT_OVERHEAD

    def predict(
        self, texts: Sequence[str], labels: Sequence[str], threshold: float
    ) -> list[list[Span]]:
        results: list[list[Span]] = [[] for _ in texts]
        todo = [(i, text) for i, text in enumerate(texts) if text.strip()]
        if not todo or not labels:
            return results

        with self._session.use() as model:
            raw = model.batch_extract_entities(
                [text for _, text in todo],
                list(labels),
                batch_size=self._batch_size,
                threshold=threshold,
                include_confidence=True,
                include_spans=True,
            )

        for (index, text), result in zip(todo, raw, strict=True):
            # {"entities": {label: [{"text", "confidence", "start", "end"}, ...]}}
            for label, items in result.get("entities", {}).items():
                for item in items:
                    start, end, reported = _drop_synthetic_terminator(
                        text, item["start"], item["end"], item.get("text")
                    )
                    if start >= end:
                        continue  # the "entity" was only the appended period
                    results[index].append(
                        checked_span(text, start, end, label, item["confidence"], reported)
                    )
        return results


_SENTENCE_END = (".", "!", "?")


def _drop_synthetic_terminator(
    text: str, start: int, end: int, reported: str | None
) -> tuple[int, int, str | None]:
    """Undo the one edit gliner2 makes to its input before tokenizing.

    The library appends "." to any text that does not already end in ".", "!"
    or "?". The model therefore sees one token we never sent, at offset
    ``len(text)``, and may include it in an entity that ends there, reporting
    ``end == len(text) + 1``. Because the character is always appended and
    never inserted, every other offset is unaffected; the only correction
    needed is to cut that one position back off. Anything else out of range
    is left alone for ``checked_span`` to reject.
    """
    if end == len(text) + 1 and not text.endswith(_SENTENCE_END):
        end -= 1
        if reported is not None and reported.endswith("."):
            reported = reported[:-1]
    return start, end, reported
