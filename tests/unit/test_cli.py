import io
import json

import pytest

from fakes import ALL_LABELS, FakeBackend
from pii_extract.cli import run
from pii_extract.pipeline import Pipeline

TEXT = "Contact Jane Doe at jane.doe@example.com or 555-867-5309."


class Harness:
    """Runs the CLI with a fake backend and records the settings it resolved."""

    def __init__(self):
        self.settings = None

    def factory(self, settings):
        self.settings = settings
        return Pipeline(
            FakeBackend(budget=settings.window_tokens or 64),
            labels=settings.labels,
            threshold=settings.threshold,
            overlap_tokens=min(settings.overlap_tokens, 4),
        )

    def __call__(self, *argv, stdin: bytes = b"", env=None):
        out = io.StringIO()
        stream = io.TextIOWrapper(io.BytesIO(stdin), encoding="utf-8")
        code = run(
            list(argv),
            stdin=stream,
            stdout=out,
            env=env or {"PII_LABELS": ",".join(ALL_LABELS)},
            pipeline_factory=self.factory,
        )
        assert code == 0
        return out.getvalue()


@pytest.fixture
def cli():
    return Harness()


def test_extract_from_argument(cli):
    payload = json.loads(cli("extract", TEXT))
    assert payload["model"] == "fake/regex"
    assert [(e["label"], e["text"]) for e in payload["entities"]] == [
        ("person", "Jane Doe"),
        ("email", "jane.doe@example.com"),
        ("phone number", "555-867-5309"),
    ]
    for entity in payload["entities"]:
        assert TEXT[entity["start"] : entity["end"]] == entity["text"]


def test_extract_from_stdin_when_no_text_is_given(cli):
    assert json.loads(cli("extract", stdin=TEXT.encode()))["entities"][0]["text"] == "Jane Doe"
    assert json.loads(cli("extract", "-", stdin=TEXT.encode()))["entities"][0]["text"] == "Jane Doe"


def test_extract_from_file_preserves_bom_and_crlf_offsets(cli, tmp_path):
    raw = "﻿line one\r\ncafé jane.doe@example.com\r\n555-867-5309\r\n"
    path = tmp_path / "input.txt"
    path.write_bytes(raw.encode("utf-8"))

    entities = json.loads(cli("extract", "--file", str(path)))["entities"]

    assert [(e["start"], e["text"]) for e in entities] == [
        (raw.index("jane.doe@example.com"), "jane.doe@example.com"),
        (raw.index("555-867-5309"), "555-867-5309"),
    ]
    assert raw[0] == "﻿" and entities[0]["start"] == 16


def test_stdin_bytes_are_not_newline_translated(cli):
    raw = "a\r\nb\r\n555-867-5309"
    (entity,) = json.loads(cli("extract", stdin=raw.encode()))["entities"]
    assert entity["start"] == raw.index("555")  # would be 2 lower if CRLF became LF


def test_redact_json_and_text_only(cli):
    payload = json.loads(cli("redact", TEXT))
    assert payload["redacted_text"] == "Contact [PERSON_1] at [EMAIL_1] or [PHONE_1]."
    assert payload["strategy"] == "placeholder"
    assert len(payload["replacements"]) == 3

    assert cli("redact", TEXT, "--text-only") == "Contact [PERSON_1] at [EMAIL_1] or [PHONE_1]."


def test_redact_strategies(cli):
    masked = cli("redact", TEXT, "--strategy", "mask", "--mask-char", "#", "--text-only")
    assert masked == "Contact ######## at #################### or ############."
    one = cli("redact", TEXT, "--strategy", "hash", "--hash-salt", "pepper", "--text-only")
    two = cli("redact", TEXT, "--strategy", "hash", "--hash-salt", "pepper", "--text-only")
    assert one == two and one.startswith("Contact [PERSON:")


def test_compact_output_is_one_line_and_keeps_unicode(cli):
    out = cli("extract", "café jane.doe@example.com", "--compact")
    assert out.count("\n") == 1 and '"start":5' in out


def test_precedence_cli_over_env_over_file_over_default(cli, tmp_path):
    config = tmp_path / "pii.json"
    config.write_text(json.dumps({"threshold": 0.1, "overlap_tokens": 11, "batch_size": 2}))
    env = {"PII_THRESHOLD": "0.2", "PII_OVERLAP_TOKENS": "22", "PII_LABELS": "email"}

    cli("extract", TEXT, "--config", str(config), "--threshold", "0.3", env=env)

    assert cli.settings.threshold == 0.3  # CLI
    assert cli.settings.overlap_tokens == 22  # env
    assert cli.settings.batch_size == 2  # file
    assert cli.settings.iou_threshold == 0.5  # default
    assert cli.settings.labels == ("email",)


def test_labels_flag_overrides_environment(cli):
    payload = json.loads(cli("extract", TEXT, "--labels", "email, phone number"))
    assert [e["label"] for e in payload["entities"]] == ["email", "phone number"]


def test_model_is_released_after_the_command(cli):
    backends = []
    original = cli.factory

    def tracking(settings):
        pipeline = original(settings)
        backends.append(pipeline.backend)
        return pipeline

    cli.factory = tracking
    cli("extract", TEXT)
    assert backends[0].loaded is False


@pytest.mark.parametrize(
    "argv",
    [
        ["extract", "text", "--file", "also-a-file.txt"],
        ["extract", "--file", "/nonexistent/input.txt"],
        ["extract", "text", "--threshold", "7"],
        ["extract", "text", "--config", "/nonexistent/config.json"],
        ["redact", "text", "--strategy", "mask", "--mask-char", "ab"],
    ],
)
def test_usage_errors_exit_with_a_message(cli, argv):
    with pytest.raises(SystemExit) as raised:
        cli(*argv)
    assert isinstance(raised.value.code, str) and raised.value.code.startswith("pii-extract:")


def test_invalid_utf8_input_is_reported(cli):
    with pytest.raises(SystemExit, match="not valid UTF-8"):
        cli("extract", stdin=b"\xff\xfe broken")
