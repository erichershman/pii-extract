# pii-extract: Project Spec

## What it is

A PII detection and redaction framework, delivered as a library, a CLI, and a containerized GPU inference service. At its core is GLiNER-family zero-shot NER. The default model is `fastino/gliner2-pii-v1` (GLiNER2-PII, Apache-2.0), behind a swappable backend. It supports pluggable schema packs and several redaction modes, and is benchmarked against a public dataset.

## Model choice

- **Default:** `fastino/gliner2-pii-v1`. This is a GLiNER2-based PII model with about 0.3B parameters, released May 2026 under Apache-2.0. It uses the `gliner2` Python package, which has a different API from `gliner`.
- **Comparison backend:** `urchade/gliner_multi_pii-v1` (Apache-2.0), which uses the original `gliner` package.
- Put both behind one small backend protocol that returns the canonical span schema. Two real implementations exist, so this abstraction is justified.
- Later benchmark candidates: `knowledgator/gliner-pii-base-v1.0` (Apache-2.0) and `nvidia/gliner-PII` (NVIDIA Open Model License, not Apache; check the terms before redistributing).
- Generative LLM redactors such as OpenPipe PII-Redact (Llama 3.2 1B) are out of scope as the core. They are slower, they rewrite text instead of returning spans, and exact offsets would have to be recovered afterward.

**Target summary:** an open-source PII detection and redaction framework built on GLiNER and ONNX Runtime. It runs as a GPU-accelerated FastAPI service on Docker with the NVIDIA Container Toolkit, is benchmarked on public datasets, and runs CI on every push.

## Provenance rules

1. Use only public models and public datasets, and credit both in the README.
2. Demo and test data must be synthetic and in the PII domain: 555 phone numbers, 4111 test card numbers, fake names and emails. Do not use street-address examples in demo or test data.
3. Build from scratch. Do not port or adapt code from other codebases.
4. Make small, well-described commits from day one.

## Design principles

- **Separate the stages.** Tokenization, inference, windowing, dedup, and redaction each get their own module. No single module mixes them.
- **Use one canonical span schema** throughout. Use one span-key convention, and implement dedup in exactly one place.
- **Manage sessions per instance.** No class-level singleton and no global lock. The service must be safe under concurrent requests.
- **Write tests first** for tokenization alignment, span decoding, windowing, and dedup.
- **Avoid speculative abstraction.** Don't add an ABC or interface until a second implementation exists.
- **Don't hardcode machine-specific values** such as RAM or VRAM limits. Probe them or make them configurable.
- **Map character offsets exactly.** Every span must map back to exact character offsets in the original input. Watch for drift from normalization, BOMs, and whitespace.
- **Run CI from the start.** GitHub Actions runs pytest and ruff on every push.

## Roadmap

### Stage 1: Core inference
- Run inference through the backend protocol, starting with the GLiNER2-PII backend. Add the urchade backend if time allows.
- Implement instance-based session management with lazy loading, an explicit lifecycle, and device selection (CUDA if available, otherwise CPU).
- Use one canonical span schema.
- Write tests first for tokenization alignment and span decoding.
- Add an ONNX export script. This can be deferred.

### Stage 2: Framework layer
- **Schema packs:** label sets plus optional post-processors for each label, such as regex validators for email, SSN, and phone, and a Luhn checksum for card numbers.
- **Sliding window and IoU dedup** as a reusable, tested stage. Windows are sized to the model's token limit and the overlap is configurable. Define a rule for overlapping spans that have different labels.
- **Redaction modes:** mask, salted hash, and typed placeholders such as `[PHONE_1]`, numbered consistently for each unique value.
- **Serving layer:** the existing scaffold (FastAPI, Dockerfile, justfile, and compose, with GPU support). See below.

### Stage 3: Polish
- **CLI:** accepts a single string, a file, or stdin and outputs JSON. Config precedence is CLI, then environment variables, then a JSON file, then defaults.
- **Benchmark** against a public dataset such as AI4Privacy or Gretel synthetic PII. Put a results table and a latency figure in the README (for example, "~X ms/request on RTX 5080").
- **GitHub Actions:** run tests and ruff on every push. Stretch goal: build the Docker image in CI.
- **README:** a short demo GIF and curl examples.

## Serving layer (existing scaffold)

The scaffold provides FastAPI, a Dockerfile (CUDA 12.8 base for the RTX 5080), a justfile, and compose.yaml. It includes model residency, a cache volume, health checks, and a compose GPU reservation. All of these carry over unchanged.

Required configuration:

**Model:** the PII model is the default everywhere.
```dockerfile
ENV GLINER_MODEL=fastino/gliner2-pii-v1
```

**justfile `test` recipe:**
```make
test:
    curl -s http://localhost:8000/healthz | jq
    curl -s -X POST http://localhost:8000/extract \
        -H 'Content-Type: application/json' \
        -d '{"text": "Contact Jane Doe at jane.doe@example.com or 555-867-5309. Card on file: 4111 1111 1111 1111.", "labels": ["person", "email", "phone number", "credit card number"]}' | jq
```

**`src/main.py`:** the example for the `labels` field is `["person", "email", "phone number", "ssn"]`.

**Endpoints:** keep `GET /healthz` and `POST /extract` for detection. Add `POST /redact`.

**README provenance line:** "Built on public GLiNER-family models (`fastino/gliner2-pii-v1` and `urchade/gliner_multi_pii-v1`, both Apache-2.0). No proprietary code, data, or benchmarks."

## Environment

Development happens in WSL2 Ubuntu 24.04 with Docker and the NVIDIA Container Toolkit. The repo lives at `~/dev/pii-extract`, not under `/mnt/c/`. Commands run through the justfile.
