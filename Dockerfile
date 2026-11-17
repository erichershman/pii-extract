# syntax=docker/dockerfile:1

# CUDA 12.8 is the first CUDA 12 release with RTX 50-series (Blackwell) support.
# The PyTorch cu128 wheels bundle their own CUDA libraries, so the slim
# "base" image (driver hooks only) is enough; no cudnn/devel layers needed.
FROM nvidia/cuda:12.8.1-base-ubuntu24.04

COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, from the lockfile, so this layer is cached until
# pyproject.toml or uv.lock changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra models --no-install-project --python 3.12

COPY README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra models --python 3.12

# Run unprivileged. Model weights are downloaded on first start into
# HF_HOME, which compose mounts as a named volume so they survive rebuilds.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /cache/huggingface \
    && chown -R app:app /cache
USER app

ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/cache/huggingface \
    GLINER_MODEL=fastino/gliner2-pii-v1 \
    PII_BACKEND=gliner2 \
    PII_DEVICE=auto \
    PII_PRELOAD=1

EXPOSE 8000

# start-period covers the first-run weight download and the load onto the GPU.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

# One worker on purpose: each worker would load its own copy of the model
# into VRAM. Scale with replicas (one per GPU), not with workers.
CMD ["uvicorn", "main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
