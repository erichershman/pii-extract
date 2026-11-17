"""GLiNER (v1) backend (the ``gliner`` package), kept as the comparison model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pii_extract.backends.base import ModelSession, checked_span
from pii_extract.backends.budget import label_prompt_cost, subword_costs
from pii_extract.chunking import Token
from pii_extract.spans import Span

DEFAULT_GLINER_MODEL = "urchade/gliner_multi_pii-v1"

# [CLS], [SEP] and the <<SEP>> between the label prompt and the text.
_PROMPT_OVERHEAD = 4


class GlinerBackend:
    name = "gliner"

    def __init__(
        self, model_id: str = DEFAULT_GLINER_MODEL, device: str = "auto", batch_size: int = 8
    ) -> None:
        self.model_id = model_id
        self._batch_size = batch_size
        self._session: ModelSession[Any] = ModelSession(self._load_model, device)

    def _load_model(self, device: str) -> Any:
        from gliner import GLiNER

        model = GLiNER.from_pretrained(self.model_id)
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
        processor = self._session.load().data_processor
        words = list(processor.words_splitter(text))
        costs = subword_costs(processor.transformer_tokenizer, [word for word, _, _ in words])
        return [Token(start, end, cost) for (_, start, end), cost in zip(words, costs, strict=True)]

    def window_budget(self, labels: Sequence[str]) -> int:
        model = self._session.load()
        tokenizer = model.data_processor.transformer_tokenizer
        encoder_limit = int(model.config.encoder_config.max_position_embeddings)
        subword_budget = encoder_limit - label_prompt_cost(tokenizer, labels) - _PROMPT_OVERHEAD
        # GLiNER silently truncates input past config.max_len WORDS. Every
        # word costs at least one subword, so capping the subword budget at
        # max_len also guarantees the word cap is never hit.
        return min(subword_budget, int(model.config.max_len))

    def predict(
        self, texts: Sequence[str], labels: Sequence[str], threshold: float
    ) -> list[list[Span]]:
        results: list[list[Span]] = [[] for _ in texts]
        todo = [(i, text) for i, text in enumerate(texts) if text.strip()]
        if not todo or not labels:
            return results

        with self._session.use() as model:
            raw = model.inference(
                [text for _, text in todo],
                list(labels),
                flat_ner=True,
                threshold=threshold,
                batch_size=self._batch_size,
            )

        for (index, text), items in zip(todo, raw, strict=True):
            # [{"start", "end", "text", "label", "score"}, ...]
            for item in items:
                results[index].append(
                    checked_span(
                        text,
                        item["start"],
                        item["end"],
                        item["label"],
                        item["score"],
                        item.get("text"),
                    )
                )
        return results
