"""Command line interface: text in (argument, file, or stdin), JSON out.

    pii-extract extract "Call Jane Doe on 555-867-5309"
    pii-extract redact --file notes.txt --strategy mask
    cat notes.txt | pii-extract redact --labels "person,email"

Configuration precedence is CLI flag > environment variable > JSON config
file (``--config`` or ``$PII_CONFIG``) > built-in default.

Exit codes: 0 success, 2 usage or configuration error.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from typing import Any, TextIO

from pydantic import ValidationError

from pii_extract import __version__
from pii_extract.config import Settings, build_pipeline, build_strategy, load_settings
from pii_extract.pipeline import Pipeline
from pii_extract.redaction import STRATEGIES

PipelineFactory = Callable[[Settings], Pipeline]


def _read_bytes_as_text(data: bytes, source: str) -> str:
    """Decode input WITHOUT touching it: a BOM stays a character, CRLF stays CRLF.

    Text-mode reads would translate newlines and ``utf-8-sig`` would drop the
    BOM; either would make reported offsets disagree with the bytes the
    caller actually has.
    """
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SystemExit(f"pii-extract: {source} is not valid UTF-8: {error}") from error


def _read_input(args: argparse.Namespace, stdin: TextIO) -> str:
    if args.text is not None and args.file is not None:
        raise SystemExit("pii-extract: give either TEXT or --file, not both")
    if args.text is not None and args.text != "-":
        return args.text
    if args.file is not None and args.file != "-":
        try:
            with open(args.file, "rb") as handle:
                return _read_bytes_as_text(handle.read(), args.file)
        except OSError as error:
            raise SystemExit(f"pii-extract: cannot read {args.file}: {error.strerror}") from error
    buffer = getattr(stdin, "buffer", None)
    if buffer is not None:
        return _read_bytes_as_text(buffer.read(), "stdin")
    return stdin.read()


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    source = parser.add_argument_group("input (default: stdin)")
    source.add_argument("text", nargs="?", help="text to scan; '-' or omitted reads stdin")
    source.add_argument("-f", "--file", help="read text from this UTF-8 file ('-' for stdin)")

    config = parser.add_argument_group("configuration (each overrides env, config file, defaults)")
    config.add_argument("--config", help="JSON config file (env: PII_CONFIG)")
    config.add_argument("--backend", choices=["gliner2", "gliner"], help="env: PII_BACKEND")
    config.add_argument("--model", help="model id (env: GLINER_MODEL)")
    config.add_argument("--device", help="auto, cpu, cuda, cuda:0 (env: PII_DEVICE)")
    config.add_argument("--labels", help="comma-separated entity types (env: PII_LABELS)")
    config.add_argument("--threshold", type=float, help="minimum confidence (env: PII_THRESHOLD)")
    config.add_argument(
        "--window-tokens", type=int, help="window size cap (env: PII_WINDOW_TOKENS)"
    )
    config.add_argument(
        "--overlap-tokens", type=int, help="window overlap (env: PII_OVERLAP_TOKENS)"
    )
    config.add_argument(
        "--iou-threshold", type=float, help="duplicate IoU cutoff (env: PII_IOU_THRESHOLD)"
    )
    config.add_argument("--batch-size", type=int, help="windows per batch (env: PII_BATCH_SIZE)")

    output = parser.add_argument_group("output")
    output.add_argument("--compact", action="store_true", help="single-line JSON")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pii-extract",
        description="Detect and redact PII with GLiNER-family models. Prints JSON.",
    )
    parser.add_argument("--version", action="version", version=f"pii-extract {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    extract = commands.add_parser("extract", help="detect entities and print their spans")
    _add_common_options(extract)

    redact = commands.add_parser("redact", help="replace detected entities")
    _add_common_options(redact)
    redact.add_argument(
        "--strategy", choices=sorted(STRATEGIES), help="env: PII_REDACTION_STRATEGY"
    )
    redact.add_argument("--hash-salt", help="secret for the hash strategy (env: PII_HASH_SALT)")
    redact.add_argument("--mask-char", help="replacement character for the mask strategy")
    redact.add_argument(
        "--text-only", action="store_true", help="print only the redacted text, not JSON"
    )
    return parser


_SETTING_FLAGS = (
    "backend",
    "model",
    "device",
    "labels",
    "threshold",
    "window_tokens",
    "overlap_tokens",
    "iou_threshold",
    "batch_size",
    "strategy",
    "hash_salt",
)


def run(
    argv: Sequence[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    env: dict[str, str] | None = None,
    pipeline_factory: PipelineFactory = build_pipeline,
) -> int:
    """Run the CLI. Streams, environment, and pipeline are injectable for tests."""
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    args = build_parser().parse_args(argv)

    cli_values = {name: getattr(args, name, None) for name in _SETTING_FLAGS}
    try:
        settings = load_settings(cli=cli_values, env=env, config_file=args.config)
    except (ValidationError, ValueError, OSError) as error:
        raise SystemExit(f"pii-extract: invalid configuration: {error}") from error

    text = _read_input(args, stdin)
    pipeline = pipeline_factory(settings)
    try:
        payload: dict[str, Any]
        if args.command == "extract":
            spans = pipeline.extract(text)
            payload = {
                "entities": [span.model_dump() for span in spans],
                "model": pipeline.backend.model_id,
            }
        else:
            try:
                strategy = build_strategy(settings, mask_char=args.mask_char)
            except ValueError as error:
                raise SystemExit(f"pii-extract: {error}") from error
            result = pipeline.redact(text, strategy)
            if args.text_only:
                stdout.write(result.text)
                return 0
            payload = {
                "redacted_text": result.text,
                "strategy": result.strategy,
                "replacements": [item.model_dump() for item in result.replacements],
                "model": pipeline.backend.model_id,
            }
    finally:
        pipeline.backend.close()

    # ensure_ascii=False keeps non-ASCII PII readable; offsets are code points either way.
    if args.compact:
        stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    else:
        stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return 0


def main() -> None:
    try:
        code = run()
    except SystemExit as error:
        if isinstance(error.code, str):
            print(error.code, file=sys.stderr)
            sys.exit(2)
        raise
    sys.exit(code)


if __name__ == "__main__":
    main()
