"""Span decoding for both adapters, using stub models instead of real weights."""

import pytest

from pii_extract.backends import Gliner2Backend, GlinerBackend
from pii_extract.backends.base import ModelSession
from pii_extract.spans import OffsetError

TEXT = "Contact Jane Doe at jane.doe@example.com or 555-867-5309."
LABELS = ["person", "email"]


class StubGliner2:
    """Answers in gliner2's shape: {"entities": {label: [{text, confidence, start, end}]}}."""

    def __init__(self, shift: int = 0):
        self.shift = shift
        self.seen: list[dict] = []

    def batch_extract_entities(self, texts, labels, **kwargs):
        self.seen.append({"texts": texts, "labels": labels, **kwargs})
        out = []
        for text in texts:
            start = text.index("Jane Doe") + self.shift
            out.append(
                {
                    "entities": {
                        "person": [
                            {
                                "text": "Jane Doe",
                                "confidence": 0.97,
                                "start": start,
                                "end": start + 8,
                            }
                        ],
                        "email": [],
                    }
                }
            )
        return out


class StubGliner:
    """Answers in gliner's shape: [{start, end, text, label, score}]."""

    def __init__(self, shift: int = 0):
        self.shift = shift
        self.seen: list[dict] = []

    def inference(self, texts, labels, **kwargs):
        self.seen.append({"texts": texts, "labels": labels, **kwargs})
        out = []
        for text in texts:
            start = text.index("jane.doe@example.com") + self.shift
            out.append(
                [
                    {
                        "start": start,
                        "end": start + 20,
                        "text": "jane.doe@example.com",
                        "label": "email",
                        "score": 0.91,
                    }
                ]
            )
        return out


def with_stub(backend, stub):
    backend._session = ModelSession(lambda device: stub, device="cpu")
    return backend


def test_gliner2_results_become_canonical_spans():
    stub = StubGliner2()
    backend = with_stub(Gliner2Backend(device="cpu", batch_size=4), stub)
    (spans,) = backend.predict([TEXT], LABELS, 0.4)
    assert [(s.start, s.end, s.label, s.text) for s in spans] == [(8, 16, "person", "Jane Doe")]
    assert spans[0].score == pytest.approx(0.97)
    call = stub.seen[0]
    assert call["labels"] == LABELS and call["threshold"] == 0.4 and call["batch_size"] == 4
    assert call["include_spans"] is True and call["include_confidence"] is True


def test_gliner_results_become_canonical_spans():
    stub = StubGliner()
    backend = with_stub(GlinerBackend(device="cpu", batch_size=2), stub)
    (spans,) = backend.predict([TEXT], LABELS, 0.3)
    assert [(s.start, s.end, s.label, s.text) for s in spans] == [
        (20, 40, "email", "jane.doe@example.com")
    ]
    call = stub.seen[0]
    assert call["threshold"] == 0.3 and call["batch_size"] == 2 and call["flat_ner"] is True


@pytest.mark.parametrize(
    "backend_class,stub_class", [(Gliner2Backend, StubGliner2), (GlinerBackend, StubGliner)]
)
def test_blank_texts_are_skipped_but_keep_their_slot(backend_class, stub_class):
    stub = stub_class()
    backend = with_stub(backend_class(device="cpu"), stub)
    results = backend.predict(["", TEXT, "  \n", TEXT], LABELS, 0.5)
    assert [len(r) for r in results] == [0, 1, 0, 1]
    assert stub.seen[0]["texts"] == [TEXT, TEXT]


@pytest.mark.parametrize(
    "backend_class,stub_class", [(Gliner2Backend, StubGliner2), (GlinerBackend, StubGliner)]
)
def test_nothing_to_do_never_loads_the_model(backend_class, stub_class):
    backend = with_stub(backend_class(device="cpu"), stub_class())
    assert backend.predict([], LABELS, 0.5) == []
    assert backend.predict(["", " "], LABELS, 0.5) == [[], []]
    assert backend.predict([TEXT], [], 0.5) == [[]]
    assert not backend.loaded


@pytest.mark.parametrize(
    "backend_class,stub_class", [(Gliner2Backend, StubGliner2), (GlinerBackend, StubGliner)]
)
def test_offset_drift_from_the_model_is_an_error_not_a_wrong_span(backend_class, stub_class):
    backend = with_stub(backend_class(device="cpu"), stub_class(shift=1))
    with pytest.raises(OffsetError):
        backend.predict([TEXT], LABELS, 0.5)


def test_lifecycle_is_delegated_to_the_session():
    backend = with_stub(Gliner2Backend(device="cpu"), StubGliner2())
    assert not backend.loaded and backend.device == "cpu"
    backend.load()
    assert backend.loaded
    backend.close()
    assert not backend.loaded
