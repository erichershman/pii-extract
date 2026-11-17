import json

import pytest
from pydantic import ValidationError

from pii_extract.config import Settings, build_pipeline, build_strategy, load_settings
from pii_extract.pipeline import DEFAULT_LABELS


def test_defaults():
    settings = load_settings(env={})
    assert settings == Settings()
    assert settings.backend == "gliner2" and settings.model is None
    assert settings.labels == DEFAULT_LABELS
    assert (settings.threshold, settings.overlap_tokens, settings.iou_threshold) == (0.5, 64, 0.5)


def test_precedence_cli_over_env_over_file_over_defaults(tmp_path):
    config = tmp_path / "pii.json"
    config.write_text(
        json.dumps({"threshold": 0.1, "overlap_tokens": 11, "batch_size": 2, "device": "cpu"})
    )
    env = {"PII_THRESHOLD": "0.2", "PII_OVERLAP_TOKENS": "22"}
    cli = {"threshold": 0.3, "overlap_tokens": None}

    settings = load_settings(cli=cli, env=env, config_file=config)

    assert settings.threshold == 0.3  # CLI wins
    assert settings.overlap_tokens == 22  # env beats file; a None CLI value is "not given"
    assert settings.batch_size == 2  # file beats default
    assert settings.device == "cpu"
    assert settings.iou_threshold == 0.5  # default


def test_config_file_path_can_come_from_the_environment(tmp_path):
    config = tmp_path / "pii.json"
    config.write_text(json.dumps({"backend": "gliner", "labels": ["email"]}))
    settings = load_settings(env={"PII_CONFIG": str(config)})
    assert settings.backend == "gliner" and settings.labels == ("email",)

    other = tmp_path / "other.json"
    other.write_text(json.dumps({"backend": "gliner2"}))
    explicit = load_settings(env={"PII_CONFIG": str(config)}, config_file=other)
    assert explicit.backend == "gliner2"


def test_environment_values_are_parsed():
    settings = load_settings(
        env={
            "GLINER_MODEL": "fastino/gliner2-pii-v1",
            "PII_BACKEND": "gliner2",
            "PII_LABELS": "person, email ,phone number,",
            "PII_WINDOW_TOKENS": "256",
            "PII_PRELOAD": "true",
            "PII_HASH_SALT": "pepper",
            "PII_DEVICE": "",  # empty means unset
        }
    )
    assert settings.model == "fastino/gliner2-pii-v1"
    assert settings.labels == ("person", "email", "phone number")
    assert settings.window_tokens == 256 and settings.preload is True
    assert settings.device == "auto"
    assert "pepper" not in repr(settings)


@pytest.mark.parametrize(
    "env",
    [
        {"PII_THRESHOLD": "1.5"},
        {"PII_BACKEND": "spacy"},
        {"PII_LABELS": " , "},
        {"PII_REDACTION_STRATEGY": "rot13"},
        {"PII_BATCH_SIZE": "0"},
    ],
)
def test_invalid_values_are_rejected(env):
    with pytest.raises(ValidationError):
        load_settings(env=env)


def test_unknown_keys_in_the_config_file_are_rejected(tmp_path):
    config = tmp_path / "pii.json"
    config.write_text(json.dumps({"treshold": 0.4}))
    with pytest.raises(ValidationError):
        load_settings(env={}, config_file=config)
    config.write_text("[1, 2]")
    with pytest.raises(ValueError, match="JSON object"):
        load_settings(env={}, config_file=config)


def test_build_pipeline_is_lazy_and_resolves_the_spec_model_name():
    pipeline = build_pipeline(
        load_settings(env={"GLINER_MODEL": "fastino/gliner2-pii-v1", "PII_DEVICE": "cpu"})
    )
    assert not pipeline.backend.loaded
    assert pipeline.backend.model_id == "fastino/gliner2-privacy-filter-PII-multi"
    assert build_pipeline(Settings(backend="gliner", device="cpu")).backend.name == "gliner"


def test_build_strategy():
    settings = Settings(strategy="mask", hash_salt="pepper")
    assert build_strategy(settings).name == "mask"
    assert build_strategy(settings, "placeholder").name == "placeholder"
    assert build_strategy(settings, "hash").name == "hash"
