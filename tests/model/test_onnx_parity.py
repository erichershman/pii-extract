"""PyTorch vs ONNX Runtime parity for the urchade backend.

Needs the exported graph (``just export-onnx``) and the ``onnx`` extra; skips
otherwise. Both runtimes go through the same adapter and the same pipeline,
so what is compared is the output AFTER span decoding, windowing and merging.

Tolerance: labels and start/end offsets must be identical on every provider.
Scores must agree within 1e-3 on the CPU provider and within 1e-2 on the CUDA
provider (``PII_ONNX_PROVIDER=cuda``). The CUDA figure is looser because that
is what the provider does: against PyTorch on the same GPU the largest
difference observed is 4.4e-03 (median 1.6e-04), all on mid-confidence spans,
with no span gained or lost. On CPU the largest observed is 1.0e-04.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from pii_extract.backends import create_backend
from pii_extract.pipeline import Pipeline
from pii_extract.spans import verify_spans

PROVIDER = os.environ.get("PII_ONNX_PROVIDER", "cpu")
SCORE_TOLERANCE = {"cpu": 1e-3, "cuda": 1e-2}[PROVIDER]
ONNX_DIR = Path(os.environ.get("PII_ONNX_DIR", "models/gliner_multi_pii-v1-onnx"))
LABELS = ["person", "email", "phone number", "credit card number"]
FILLER = "The quarterly report was reviewed by the committee and no action was required. "

SHORT_TEXTS = [
    "Contact Jane Doe at jane.doe@example.com or 555-867-5309. Card on file: 4111 1111 1111 1111.",
    "﻿Name:  Jane   Doe\r\n\tE-MAIL: JANE.DOE@EXAMPLE.COM  — tel. (555) 867-5309\r\n",
    "John Roe paid with 4111 1111 1111 1111 and can be reached on 555-010-0199",
    "Nothing sensitive in this sentence at all.",
]
LONG_TEXT = "".join(
    FILLER * 2
    + f"Please reach Jane Doe at jane.doe{i}@example.com or call 555-{200 + i}-{1000 + i}. "
    for i in range(40)
)


@pytest.fixture(scope="module")
def onnx_backend():
    if importlib.util.find_spec("onnxruntime") is None:
        pytest.skip("the `onnx` extra (onnxruntime) is not installed")
    if not (ONNX_DIR / "model.onnx").exists():
        pytest.skip(f"no exported model at {ONNX_DIR}; run `just export-onnx`")
    backend = create_backend("onnx", model_id=str(ONNX_DIR), onnx_provider=PROVIDER)
    backend.load()
    assert backend.providers[0].startswith("CUDA" if backend.provider == "cuda" else "CPU")
    yield backend
    backend.close()


def compare(torch_spans, onnx_spans) -> float:
    """Assert identical keys; return the largest score difference."""
    assert [s.key for s in onnx_spans] == [s.key for s in torch_spans]
    deltas = [abs(a.score - b.score) for a, b in zip(torch_spans, onnx_spans, strict=True)]
    assert all(delta <= SCORE_TOLERANCE for delta in deltas), max(deltas)
    return max(deltas, default=0.0)


def test_tokenization_and_budget_match(gliner_backend, onnx_backend):
    assert onnx_backend.window_budget(LABELS) == gliner_backend.window_budget(LABELS)
    for text in [*SHORT_TEXTS, LONG_TEXT]:
        assert onnx_backend.tokenize(text) == gliner_backend.tokenize(text)


@pytest.mark.parametrize("text", SHORT_TEXTS)
def test_short_inputs_decode_to_identical_spans(gliner_backend, onnx_backend, text):
    reference = Pipeline(gliner_backend, labels=LABELS).extract(text)
    candidate = Pipeline(onnx_backend, labels=LABELS).extract(text)
    verify_spans(text, candidate)
    compare(reference, candidate)


def test_windowed_long_input_decodes_to_identical_spans(gliner_backend, onnx_backend, capsys):
    torch_pipe = Pipeline(gliner_backend, labels=LABELS)
    onnx_pipe = Pipeline(onnx_backend, labels=LABELS)
    windows = torch_pipe.plan(LONG_TEXT)
    assert len(windows) > 3 and windows == onnx_pipe.plan(LONG_TEXT)

    # Per window, before merging: the raw decoded candidates must agree too.
    texts = [w.slice(LONG_TEXT) for w in windows]
    worst = 0.0
    total = 0
    for a, b in zip(
        gliner_backend.predict(texts, LABELS, 0.5),
        onnx_backend.predict(texts, LABELS, 0.5),
        strict=True,
    ):
        worst = max(worst, compare(a, b))
        total += len(a)

    merged_torch = torch_pipe.extract(LONG_TEXT)
    merged_onnx = onnx_pipe.extract(LONG_TEXT)
    verify_spans(LONG_TEXT, merged_onnx)
    worst = max(worst, compare(merged_torch, merged_onnx))
    assert len(merged_torch) > 60
    with capsys.disabled():
        print(
            f"\nONNX parity ({PROVIDER}): {len(windows)} windows, {total} window-level spans, "
            f"{len(merged_torch)} merged spans, max score delta {worst:.2e}"
        )


def test_cuda_probe_agrees_with_what_the_backend_can_do():
    if importlib.util.find_spec("onnxruntime") is None:
        pytest.skip("the `onnx` extra (onnxruntime) is not installed")
    import onnxruntime

    from pii_extract.backends.onnx_probe import probe_cuda_provider

    result = probe_cuda_provider()
    if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
        assert not result.ok and "no CUDA provider" in result.detail
    elif PROVIDER == "cuda":
        assert result.ok, result.detail
    else:
        assert result.detail  # either outcome is legitimate on a CPU-only run; it must not crash
