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

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pii_extract.backends import DEFAULT_BACKEND, create_backend
from pii_extract.dedup import DEFAULT_IOU_THRESHOLD
from pii_extract.pipeline import DEFAULT_OVERLAP_TOKENS, DEFAULT_THRESHOLD, Pipeline
from pii_extract.redaction import STRATEGIES, RedactionStrategy, make_strategy
from pii_extract.schema import DEFAULT_PACK, get_pack

CONFIG_FILE_ENV = "PII_CONFIG"

# Environment variable for each setting. GLINER_MODEL is the name the serving
# scaffold has always used; everything else is namespaced.
ENV_VARS: dict[str, str] = {
    "backend": "PII_BACKEND",
    "model": "GLINER_MODEL",
    "device": "PII_DEVICE",
    "onnx_provider": "PII_ONNX_PROVIDER",
    "schema_pack": "PII_SCHEMA_PACK",
    "labels": "PII_LABELS",
    "validators": "PII_VALIDATORS",
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

    backend: Literal["gliner2", "gliner", "onnx"] = DEFAULT_BACKEND
    model: str | None = Field(default=None, description="Model id; default depends on backend.")
    device: str = Field(default="auto", description="auto, cpu, cuda, cuda:0, ...")
    onnx_provider: Literal["cpu", "cuda"] = Field(
        default="cpu", description="ONNX Runtime execution provider (onnx backend only)."
    )
    schema_pack: str = Field(default=DEFAULT_PACK, description="Label set and validators.")
    labels: tuple[str, ...] | None = Field(
        default=None, description="Entity types to detect; default: the schema pack's labels."
    )
    validators: tuple[str, ...] = Field(
        default=(),
        description="Validators to run: off (default), on/all, or a list of labels.",
    )
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

    @field_validator("validators", mode="before")
    @classmethod
    def _parse_validators(cls, value: Any) -> Any:
        """``on``/``all``/``true`` -> every validator; ``off``/``none``/``false`` -> none;
        anything else is a comma-separated list of labels to validate."""
        if isinstance(value, bool):
            return ("all",) if value else ()
        if isinstance(value, str):
            switch = value.strip().casefold()
            if switch in {"on", "all", "true", "yes", "1"}:
                return ("all",)
            if switch in {"off", "none", "false", "no", "0", ""}:
                return ()
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @model_validator(mode="after")
    def _check_schema(self) -> Settings:
        get_pack(self.schema_pack).select(self.validators)  # raises on unknown names
        return self

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
        onnx_provider=settings.onnx_provider,
        batch_size=settings.batch_size,
    )
    return Pipeline(
        backend,
        labels=settings.labels,
        schema=get_pack(settings.schema_pack),
        validators=settings.validators,
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
