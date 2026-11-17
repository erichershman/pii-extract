"""The backend protocol and the per-instance model session.

A backend adapts one model library to the canonical span schema. It owns
three things the rest of the pipeline cannot know: how the model tokenizes,
how large a window it can take, and how to run it.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Protocol

from pii_extract.chunking import Token
from pii_extract.spans import OffsetError, Span, span_from_offsets

logger = logging.getLogger(__name__)


class Backend(Protocol):
    """What the pipeline needs from a model."""

    name: str
    model_id: str

    @property
    def device(self) -> str:
        """The resolved device (``"cuda"``, ``"cpu"``, ...)."""
        ...

    @property
    def loaded(self) -> bool: ...

    def load(self) -> None:
        """Load weights now. Idempotent. Otherwise loading happens on first use."""
        ...

    def close(self) -> None:
        """Release the model and its memory. The backend may be loaded again."""
        ...

    def tokenize(self, text: str) -> list[Token]:
        """Tokens of ``text`` with offsets into ``text`` and a window-budget cost."""
        ...

    def window_budget(self, labels: Sequence[str]) -> int:
        """Largest total token cost one window may carry for these labels."""
        ...

    def predict(
        self, texts: Sequence[str], labels: Sequence[str], threshold: float
    ) -> list[list[Span]]:
        """Detect entities in each text. Offsets are relative to that text."""
        ...


def resolve_device(requested: str = "auto") -> str:
    """``"auto"`` means CUDA when a GPU is usable, otherwise CPU."""
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class ModelSession[ModelT]:
    """Owns one loaded model: lazy load, explicit lifecycle, serialized inference.

    All state lives on the instance. There is no module-level model, no
    class-level singleton, and no global lock: two sessions are fully
    independent, and a test can create and close as many as it likes.

    Concurrency: ``_lock`` belongs to this instance. It makes the lazy load
    happen exactly once under concurrent first requests, and it serializes
    forward passes on this one model. A single GPU executes one batch at a
    time anyway, and the model libraries keep per-call state on the model
    object, so serializing per session is both safe and close to free.
    Throughput comes from batching windows inside a call, not from threads.
    """

    def __init__(self, loader: Callable[[str], ModelT], device: str = "auto") -> None:
        self._loader = loader
        self._requested_device = device
        self._device: str | None = None
        self._model: ModelT | None = None
        self._lock = threading.RLock()

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = resolve_device(self._requested_device)
        return self._device

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> ModelT:
        if self._model is None:
            with self._lock:
                if self._model is None:  # another thread may have won the race
                    logger.info("loading model on %s", self.device)
                    self._model = self._loader(self.device)
        return self._model

    @contextmanager
    def use(self) -> Iterator[ModelT]:
        """Exclusive access to the (lazily loaded) model for one inference call."""
        with self._lock:
            yield self.load()

    def close(self) -> None:
        with self._lock:
            if self._model is None:
                return
            on_cuda = self.device.startswith("cuda")
            self._model = None
            if on_cuda:
                import torch

                torch.cuda.empty_cache()

    def __enter__(self) -> ModelSession[ModelT]:
        self.load()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def checked_span(
    source: str, start: int, end: int, label: str, score: float, reported_text: str | None
) -> Span:
    """Turn a model's raw detection into a canonical span, verifying its offsets.

    The span's text is always read from ``source``. If the model also reported
    the text and it disagrees with the slice, the offsets have drifted from
    the string we gave the model; that must never be papered over, because a
    redactor acting on wrong offsets destroys the wrong characters and leaves
    the PII in place.
    """
    span = span_from_offsets(source, start, end, label, score)
    if reported_text is not None and reported_text != span.text:
        raise OffsetError(
            f"backend reported {reported_text!r} at [{start}, {end}) but the text there is "
            f"{span.text!r}"
        )
    return span
