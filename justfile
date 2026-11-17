# pii-extract task runner. `just` with no arguments lists the recipes.
#
# The Python environment is managed by uv and pinned by uv.lock:
#   just setup      -> everything, including torch (CUDA 12.8) and both model runtimes
#   just setup-ci   -> only what the pure-logic tests and ruff need (no torch)

set shell := ["bash", "-euo", "pipefail", "-c"]

port := env_var_or_default("PII_PORT", "8000")
base := "http://localhost:" + port

default:
    @just --list

# --- environment -------------------------------------------------------------

# Create .venv from the lockfile with all extras (models, bench) and dev tools.
setup:
    uv sync --frozen --all-extras

# Minimal environment: no torch, no model runtimes.
setup-ci:
    uv sync --frozen

# Re-resolve dependencies and rewrite uv.lock (do this deliberately).
lock:
    uv lock --upgrade

# --- quality -----------------------------------------------------------------

# Pure-logic tests: chunking, dedup, redaction, offsets, API, CLI. No model needed.
unit:
    uv run pytest -m "not model"

# Tests that load real model weights (downloads them on first run; uses the GPU if present).
model-tests:
    uv run pytest -m model

# Every test.
pytest:
    uv run pytest

lint:
    uv run ruff check .
    uv run ruff format --check .

fmt:
    uv run ruff check --fix .
    uv run ruff format .

# What CI runs.
ci: lint unit

# --- local service -----------------------------------------------------------

# Run the API on the host (no Docker), with the model loaded at startup.
serve:
    PII_PRELOAD=1 uv run uvicorn main:app --app-dir src --host 0.0.0.0 --port {{port}}

# --- container ---------------------------------------------------------------

build:
    docker compose build

up:
    docker compose up -d --wait

down:
    docker compose down

logs:
    docker compose logs -f pii-extract

# Smoke-test a running service (container or `just serve`).
test:
    curl -s {{base}}/healthz | jq
    curl -s -X POST {{base}}/extract \
        -H 'Content-Type: application/json' \
        -d '{"text": "Contact Jane Doe at jane.doe@example.com or 555-867-5309. Card on file: 4111 1111 1111 1111.", "labels": ["person", "email", "phone number", "credit card number"]}' | jq

# Same input through /redact.
test-redact strategy="placeholder":
    curl -s -X POST {{base}}/redact \
        -H 'Content-Type: application/json' \
        -d '{"text": "Contact Jane Doe at jane.doe@example.com or 555-867-5309. Card on file: 4111 1111 1111 1111.", "labels": ["person", "email", "phone number", "credit card number"], "strategy": "{{strategy}}"}' | jq

# --- benchmark ---------------------------------------------------------------

# Compare both backends on a public dataset sample; prints a Markdown table.
bench *args:
    uv run python -m bench.run {{args}}
