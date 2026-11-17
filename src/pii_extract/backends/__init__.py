"""Model backends. Importing this package does not import torch."""

from __future__ import annotations

from pii_extract.backends.base import Backend, ModelSession, resolve_device
from pii_extract.backends.gliner2_backend import DEFAULT_GLINER2_MODEL, Gliner2Backend
from pii_extract.backends.gliner_backend import DEFAULT_GLINER_MODEL, GlinerBackend

BACKENDS = {
    Gliner2Backend.name: (Gliner2Backend, DEFAULT_GLINER2_MODEL),
    GlinerBackend.name: (GlinerBackend, DEFAULT_GLINER_MODEL),
}
DEFAULT_BACKEND = Gliner2Backend.name


def create_backend(
    name: str = DEFAULT_BACKEND,
    model_id: str | None = None,
    device: str = "auto",
    batch_size: int = 8,
) -> Backend:
    """Build a backend by name. The model is not loaded until first use."""
    try:
        backend_class, default_model = BACKENDS[name]
    except KeyError:
        raise ValueError(f"unknown backend {name!r}; choose from {sorted(BACKENDS)}") from None
    return backend_class(model_id=model_id or default_model, device=device, batch_size=batch_size)


__all__ = [
    "BACKENDS",
    "DEFAULT_BACKEND",
    "DEFAULT_GLINER2_MODEL",
    "DEFAULT_GLINER_MODEL",
    "Backend",
    "Gliner2Backend",
    "GlinerBackend",
    "ModelSession",
    "create_backend",
    "resolve_device",
]
