"""GLiNER2 backend (the ``gliner2`` package). Default model: GLiNER2-PII."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pii_extract.backends.base import ModelSession, checked_span
from pii_extract.backends.budget import label_prompt_cost, subword_costs
from pii_extract.chunking import Token
from pii_extract.spans import Span

# GLiNER2-PII. The model card and the project spec call it
# "fastino/gliner2-pii-v1"; this is the repository it is actually published under.
DEFAULT_GLINER2_MODEL = "fastino/gliner2-privacy-filter-PII-multi"

# Subword tokens the schema prompt spends besides the labels themselves
# ([CLS], [SEP], the "( [P] entities ( ... ) )" scaffolding, the text separator).
_PROMPT_OVERHEAD = 16


class Gliner2Backend:
    name = "gliner2"

    def __init__(
        self, model_id: str = DEFAULT_GLINER2_MODEL, device: str = "auto", batch_size: int = 8
    ) -> None:
        self.model_id = model_id
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
                    results[index].append(
                        checked_span(
                            text,
                            item["start"],
                            item["end"],
                            label,
                            item["confidence"],
                            item.get("text"),
                        )
                    )
        return results
