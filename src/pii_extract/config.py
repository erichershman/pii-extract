"""Settings and their precedence: CLI > environment > JSON file > defaults.

One ``Settings`` object configures the library, the CLI, and the service, so
all three resolve configuration the same way.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pii_extract.backends import DEFAULT_BACKEND, create_backend
from pii_extract.dedup import DEFAULT_IOU_THRESHOLD
from pii_extract.pipeline import (
    DEFAULT_LABELS,
    DEFAULT_OVERLAP_TOKENS,
    DEFAULT_THRESHOLD,
    Pipeline,
)
from pii_extract.redaction import STRATEGIES, RedactionStrategy, make_strategy

CONFIG_FILE_ENV = "PII_CONFIG"

# Environment variable for each setting. GLINER_MODEL is the name the serving
# scaffold has always used; everything else is namespaced.
ENV_VARS: dict[str, str] = {
    "backend": "PII_BACKEND",
    "model": "GLINER_MODEL",
    "device": "PII_DEVICE",
    "labels": "PII_LABELS",
    "threshold": "PII_THRESHOLD",
    "window_tokens": "PII_WINDOW_TOKENS",
    "overlap_tokens": "PII_OVERLAP_TOKENS",
    "iou_threshold": "PII_IOU_THRESHOLD",
    "batch_size": "PII_BATCH_SIZE",
    "strategy": "PII_REDACTION_STRATEGY",
    "hash_salt": "PII_HASH_SALT",
    "preload": "PII_PRELOAD",
    "max_text_chars": "PII_MAX_TEXT_CHARS",
}


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: Literal["gliner2", "gliner"] = DEFAULT_BACKEND
    model: str | None = Field(default=None, description="Model id; default depends on backend.")
    device: str = Field(default="auto", description="auto, cpu, cuda, cuda:0, ...")
    labels: tuple[str, ...] = DEFAULT_LABELS
    threshold: float = Field(default=DEFAULT_THRESHOLD, ge=0.0, le=1.0)
    window_tokens: int | None = Field(
        default=None, gt=0, description="Cap on window size; the model's own limit still applies."
    )
    overlap_tokens: int = Field(default=DEFAULT_OVERLAP_TOKENS, ge=0)
    iou_threshold: float = Field(default=DEFAULT_IOU_THRESHOLD, gt=0.0, le=1.0)
    batch_size: int = Field(default=8, gt=0, description="Windows per forward pass.")
    strategy: str = "placeholder"
    hash_salt: str | None = Field(default=None, repr=False)
    preload: bool = Field(default=False, description="Service: load the model at startup.")
    max_text_chars: int = Field(default=1_000_000, gt=0)

    @field_validator("labels", mode="before")
    @classmethod
    def _split_labels(cls, value: Any) -> Any:
        """Accept ``"person, email"`` (the form an env var or CLI flag takes)."""
        if isinstance(value, str):
            value = [part.strip() for part in value.split(",")]
        if isinstance(value, (list, tuple)):
            value = [label for label in value if label]
            if not value:
                raise ValueError("at least one label is required")
        return value

    @field_validator("strategy")
    @classmethod
    def _known_strategy(cls, value: str) -> str:
        if value not in STRATEGIES:
            raise ValueError(f"unknown strategy {value!r}; choose from {sorted(STRATEGIES)}")
        return value


def _read_config_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"config file {path} must contain a JSON object")
    return data


def load_settings(
    cli: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    config_file: str | os.PathLike[str] | None = None,
) -> Settings:
    """Resolve settings from all four layers.

    ``cli`` holds explicitly passed options only; ``None`` values mean "not
    given" and fall through to the next layer. ``env`` defaults to the process
    environment. The config file is ``config_file`` if given, otherwise the
    path in ``$PII_CONFIG``, otherwise none.
    """
    env = os.environ if env is None else env
    merged: dict[str, Any] = {}

    path = config_file or env.get(CONFIG_FILE_ENV)
    if path:
        merged.update(_read_config_file(path))

    for field, variable in ENV_VARS.items():
        value = env.get(variable)
        if value is not None and value != "":
            merged[field] = value

    for field, value in (cli or {}).items():
        if value is not None:
            merged[field] = value

    return Settings(**merged)


def build_pipeline(settings: Settings) -> Pipeline:
    """Construct a pipeline (and its backend) from settings. Loads nothing yet."""
    backend = create_backend(
        settings.backend,
        model_id=settings.model,
        device=settings.device,
        batch_size=settings.batch_size,
    )
    return Pipeline(
        backend,
        labels=settings.labels,
        threshold=settings.threshold,
        window_tokens=settings.window_tokens,
        overlap_tokens=settings.overlap_tokens,
        iou_threshold=settings.iou_threshold,
    )


def build_strategy(
    settings: Settings, name: str | None = None, mask_char: str | None = None
) -> RedactionStrategy:
    """A fresh strategy instance (strategies keep per-document state)."""
    chosen = name or settings.strategy
    if chosen == "hash":
        return make_strategy("hash", salt=settings.hash_salt)
    if chosen == "mask" and mask_char is not None:
        return make_strategy("mask", mask_char=mask_char)
    return make_strategy(chosen)
