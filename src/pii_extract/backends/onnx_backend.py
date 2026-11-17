"""ONNX Runtime backend for an exported GLiNER (v1) model.

Everything except loading is inherited from ``GlinerBackend``: the same word
splitter, window budget, inference call and span decoding run on the
ONNX-loaded model, which is what makes PyTorch/ONNX parity checkable.

Export a model first with ``just export-onnx``.

The execution provider is explicit configuration (``cpu`` or ``cuda``,
default ``cpu``), not something to discover by trying. A broken CUDA setup can
crash the process with a segmentation fault while the session is being
created, and a segfault cannot be caught: a "try CUDA, fall back to CPU"
policy would take the service down instead of falling back. So CUDA is only
used when the operator asks for it, and asking for it without getting it is an
error rather than a silent downgrade.

Before a CUDA session is created in this process, ``onnx_probe`` creates one
in a child process. If the child crashes or reports that CUDA is unusable,
loading fails with an error that says so, and this process never touches the
CUDA provider.
"""

from __future__ import annotations

import logging
from typing import Any

from pii_extract.backends import onnx_probe
from pii_extract.backends.gliner_backend import GlinerBackend

logger = logging.getLogger(__name__)

DEFAULT_ONNX_MODEL = "models/gliner_multi_pii-v1-onnx"
CUDA_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"
PROVIDERS = {"cpu": [CPU_PROVIDER], "cuda": [CUDA_PROVIDER, CPU_PROVIDER]}


class OnnxBackend(GlinerBackend):
    name = "onnx"

    def __init__(
        self,
        model_id: str = DEFAULT_ONNX_MODEL,
        device: str = "auto",
        batch_size: int = 8,
        provider: str = "cpu",
    ) -> None:
        if provider not in PROVIDERS:
            raise ValueError(f"unknown ONNX provider {provider!r}; choose from {sorted(PROVIDERS)}")
        # ``device`` configures torch backends; ONNX Runtime placement is
        # decided by ``provider`` alone.
        super().__init__(model_id=model_id, device=provider, batch_size=batch_size)
        self.provider = provider
        self.providers: list[str] = []  # what the session actually uses, once loaded

    def _load_model(self, device: str) -> Any:
        # Probe first, before this process imports anything that touches CUDA.
        if self.provider == "cuda":
            probe = onnx_probe.probe_cuda_provider()
            if not probe.ok:
                raise RuntimeError(
                    f"PII_ONNX_PROVIDER=cuda, but the CUDA provider is not usable: {probe.detail}. "
                    "Fix the host or set PII_ONNX_PROVIDER=cpu."
                )

        from gliner import GLiNER

        requested = PROVIDERS[self.provider]
        model = GLiNER.from_pretrained(
            self.model_id,
            load_onnx_model=True,
            load_tokenizer=True,
            runtime_options={"providers": requested},
        )
        session = getattr(model.model, "session", None)
        # ONNX Runtime drops a provider it cannot initialise without raising,
        # so ask the session what it ended up with.
        self.providers = list(session.get_providers()) if session is not None else requested
        if self.provider == "cuda" and CUDA_PROVIDER not in self.providers:
            raise RuntimeError(
                f"the CUDA provider was requested but ONNX Runtime is using {self.providers}"
            )
        logger.info("ONNX Runtime providers for %s: %s", self.model_id, self.providers)
        return model
