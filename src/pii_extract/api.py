"""The FastAPI service: GET /healthz, POST /extract, POST /redact.

The app owns exactly one pipeline, created in ``create_app`` and stored on
``app.state``. Nothing is module-global, so tests build as many apps as they
like, each with its own (fake or real) backend.

Endpoints are plain ``def`` functions: FastAPI runs them in its thread pool,
which keeps the event loop free while a forward pass is running. Concurrent
requests are safe because the backend's session serializes inference on its
own instance lock.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from pii_extract import __version__
from pii_extract.config import Settings, build_pipeline, build_strategy, load_settings
from pii_extract.pipeline import Pipeline
from pii_extract.redaction import Replacement
from pii_extract.spans import OffsetError, Span

logger = logging.getLogger(__name__)

EXAMPLE_TEXT = (
    "Contact Jane Doe at jane.doe@example.com or 555-867-5309. Card on file: 4111 1111 1111 1111."
)


class ExtractRequest(BaseModel):
    text: str = Field(
        description="The text to scan. It is never modified.", examples=[EXAMPLE_TEXT]
    )
    labels: list[str] | None = Field(
        default=None,
        description="Entity types to detect (zero-shot). Omit to use the server's defaults.",
        examples=[["person", "email", "phone number", "ssn"]],
    )
    threshold: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Minimum confidence. Omit for the default."
    )


class RedactRequest(ExtractRequest):
    strategy: Literal["mask", "hash", "placeholder"] | None = Field(
        default=None, description="How to replace each entity. Omit for the server's default."
    )
    mask_char: str | None = Field(
        default=None, min_length=1, max_length=1, description="Only for the mask strategy."
    )


class ExtractResponse(BaseModel):
    entities: list[Span] = Field(
        description="Non-overlapping, in document order. Offsets are Unicode code points "
        "into the request's text; text == request.text[start:end]."
    )
    model: str


class RedactResponse(BaseModel):
    redacted_text: str
    strategy: str
    replacements: list[Replacement]
    model: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    backend: str
    model: str
    device: str
    model_loaded: bool


def create_app(settings: Settings | None = None, pipeline: Pipeline | None = None) -> FastAPI:
    """Build the app. Pass ``pipeline`` to inject a backend (used by the tests)."""
    settings = settings or load_settings()
    pipeline = pipeline or build_pipeline(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Model residency: with preload on, the weights are on the device
        # before the first request and stay there until shutdown.
        if settings.preload:
            pipeline.backend.load()
        yield
        pipeline.backend.close()

    app = FastAPI(
        title="pii-extract",
        version=__version__,
        summary="PII detection and redaction on GLiNER-family models.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.pipeline = pipeline

    def check_size(text: str) -> None:
        if len(text) > settings.max_text_chars:
            raise HTTPException(
                status_code=413,
                detail=f"text has {len(text)} characters; the limit is {settings.max_text_chars}",
            )

    @app.exception_handler(OffsetError)
    def offset_error(request: Request, error: OffsetError) -> JSONResponse:
        # A backend produced offsets that do not match the input. Fail the
        # request rather than answer with spans we could not verify: a caller
        # redacting by those offsets would leave the PII in place.
        logger.error("offset verification failed: %s", error)
        return JSONResponse(status_code=500, content={"detail": "span offset verification failed"})

    @app.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        backend = pipeline.backend
        return HealthResponse(
            status="ok",
            version=__version__,
            backend=backend.name,
            model=backend.model_id,
            device=backend.device,
            model_loaded=backend.loaded,
        )

    @app.post("/extract", response_model=ExtractResponse)
    def extract(request: ExtractRequest) -> ExtractResponse:
        check_size(request.text)
        try:
            entities = pipeline.extract(request.text, request.labels, request.threshold)
        except OffsetError:
            raise
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return ExtractResponse(entities=entities, model=pipeline.backend.model_id)

    @app.post("/redact", response_model=RedactResponse)
    def redact(request: RedactRequest) -> RedactResponse:
        check_size(request.text)
        try:
            strategy = build_strategy(settings, request.strategy, request.mask_char)
            result = pipeline.redact(request.text, strategy, request.labels, request.threshold)
        except OffsetError:
            raise
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return RedactResponse(
            redacted_text=result.text,
            strategy=result.strategy,
            replacements=result.replacements,
            model=pipeline.backend.model_id,
        )

    return app
