# pii-extract

PII detection and redaction built on GLiNER-family zero-shot NER. It ships as a
Python library, a CLI, and a containerized GPU inference service.

Built on public GLiNER-family models (`fastino/gliner2-pii-v1` and `urchade/gliner_multi_pii-v1`, both Apache-2.0). No proprietary code, data, or benchmarks.

> `fastino/gliner2-pii-v1` is the name GLiNER2-PII's model card gives the model.
> On the Hugging Face Hub the weights are published as
> [`fastino/gliner2-privacy-filter-PII-multi`](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi);
> pii-extract accepts either name and loads that repository.

What it does:

- **Detects** entities for any label set you give it (the models are zero-shot)
  and returns spans with **exact character offsets into your original text**.
- **Handles long documents** with token-budgeted sliding windows and a
  documented merge rule for detections that repeat or disagree across windows.
- **Redacts** with a pluggable strategy: `mask`, salted `hash`, or typed
  `placeholder` (`[PHONE_1]`, numbered consistently per unique value).
- **Swaps models** behind a small backend protocol: GLiNER2-PII by default,
  `urchade/gliner_multi_pii-v1` as the comparison backend.

All examples and test data are synthetic.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/).
The environment is pinned by `uv.lock`.

```bash
just setup          # .venv with torch (CUDA 12.8 wheels), gliner2, gliner, dev tools
just unit           # 165 pure-logic tests, no model needed (under a second)
just model-tests    # 19 tests on real weights (downloads ~1.5 GB on first run)
```

### Service

```bash
just serve          # on the host, or:
just build && just up     # in Docker with the NVIDIA Container Toolkit
just test           # smoke test: /healthz and /extract
```

```bash
curl -s http://localhost:8000/healthz
```
```json
{"status":"ok","version":"0.1.0","backend":"gliner2",
 "model":"fastino/gliner2-privacy-filter-PII-multi","device":"cuda","model_loaded":true}
```

```bash
curl -s -X POST http://localhost:8000/extract \
  -H 'Content-Type: application/json' \
  -d '{"text": "Contact Jane Doe at jane.doe@example.com or 555-867-5309. Card on file: 4111 1111 1111 1111.",
       "labels": ["person", "email", "phone number", "credit card number"]}'
```
```json
{"entities": [
   {"start": 8,  "end": 16, "label": "person",             "score": 0.995, "text": "Jane Doe"},
   {"start": 20, "end": 40, "label": "email",              "score": 1.000, "text": "jane.doe@example.com"},
   {"start": 44, "end": 56, "label": "phone number",       "score": 1.000, "text": "555-867-5309"},
   {"start": 72, "end": 91, "label": "credit card number", "score": 0.993, "text": "4111 1111 1111 1111"}],
 "model": "fastino/gliner2-privacy-filter-PII-multi"}
```

```bash
curl -s -X POST http://localhost:8000/redact \
  -H 'Content-Type: application/json' \
  -d '{"text": "Contact Jane Doe at jane.doe@example.com or 555-867-5309. Call 555-867-5309 again.",
       "strategy": "placeholder"}'
```
```json
{"redacted_text": "Contact [PERSON_1] at [EMAIL_1] or [PHONE_1]. Call [PHONE_1] again.",
 "strategy": "placeholder",
 "replacements": [
   {"span": {"start": 8, "end": 16, "label": "person", "score": 0.995, "text": "Jane Doe"},
    "replacement": "[PERSON_1]", "redacted_start": 8, "redacted_end": 18}, "..."],
 "model": "fastino/gliner2-privacy-filter-PII-multi"}
```

The other strategies, on the first sentence:

| `strategy` | Result |
|---|---|
| `placeholder` | `Contact [PERSON_1] at [EMAIL_1] or [PHONE_1].` |
| `mask` | `Contact ******** at ******************** or ************.` |
| `hash` | `Contact [PERSON:478703faadf6] at [EMAIL:904f913ef55b] or [PHONE:c947c54d2985].` |

Request fields: `text` (required), `labels` and `threshold` (optional, fall back
to server defaults), and for `/redact` also `strategy` and `mask_char`.
Interactive docs are at `/docs`.

Offsets are Unicode code points (Python `str` indices), half-open, and satisfy
`text[start:end] == entity.text` for the exact string you sent.

### CLI

Text comes from an argument, `--file`, or stdin; output is JSON.

```bash
uv run pii-extract extract "Contact Jane Doe at jane.doe@example.com" --labels "person,email"
uv run pii-extract redact --file notes.txt --strategy mask
cat notes.txt | uv run pii-extract redact --text-only
```

### Library

```python
from pii_extract import Pipeline, make_strategy
from pii_extract.backends import create_backend

pipeline = Pipeline(create_backend("gliner2"), labels=["person", "email", "phone number"])
spans = pipeline.extract(text)  # list[Span]
result = pipeline.redact(text, make_strategy("placeholder"))  # .text, .replacements
```

## Configuration

Precedence: **CLI flag > environment variable > JSON config file > default.**
The config file is `--config PATH` or `$PII_CONFIG`; its keys are the setting
names below.

| Setting | Env var | CLI flag | Default |
|---|---|---|---|
| `backend` | `PII_BACKEND` | `--backend` | `gliner2` (or `gliner`) |
| `model` | `GLINER_MODEL` | `--model` | the backend's default model |
| `device` | `PII_DEVICE` | `--device` | `auto` (CUDA if available, else CPU) |
| `labels` | `PII_LABELS` (comma-separated) | `--labels` | `person, email, phone number, credit card number, ssn` |
| `threshold` | `PII_THRESHOLD` | `--threshold` | `0.5` |
| `window_tokens` | `PII_WINDOW_TOKENS` | `--window-tokens` | the model's limit |
| `overlap_tokens` | `PII_OVERLAP_TOKENS` | `--overlap-tokens` | `64` |
| `iou_threshold` | `PII_IOU_THRESHOLD` | `--iou-threshold` | `0.5` |
| `batch_size` | `PII_BATCH_SIZE` | `--batch-size` | `8` |
| `strategy` | `PII_REDACTION_STRATEGY` | `--strategy` | `placeholder` |
| `hash_salt` | `PII_HASH_SALT` | `--hash-salt` | random per request |
| `preload` | `PII_PRELOAD` | | `false` (the container sets it to `1`) |
| `max_text_chars` | `PII_MAX_TEXT_CHARS` | | `1000000` |

Set `PII_HASH_SALT` to a secret if `hash` redactions must be stable across
requests. Without it each request gets a fresh random salt, so digests are
consistent within a document but cannot be linked across documents.

## Benchmark

`just bench` scores both backends on a seeded 500-document sample of the
public, synthetic [`gretelai/gretel-pii-masking-en-v1`](https://huggingface.co/datasets/gretelai/gretel-pii-masking-en-v1)
test split (Apache-2.0). Both models get the same seven natural-language
labels. Numbers below are from an **NVIDIA RTX 5080** (CUDA 12.8,
torch 2.11.0) and are precision / recall / F1 for exact span matches
(start, end, and label must all agree).

| Label | Support | `fastino/gliner2-privacy-filter-PII-multi` | `urchade/gliner_multi_pii-v1` |
|---|---:|---|---|
| person | 230 | 0.611 / 0.922 / 0.735 | 0.558 / 0.943 / 0.701 |
| email | 97 | 0.932 / 0.990 / 0.960 | 0.913 / 0.969 / 0.940 |
| phone number | 75 | 0.841 / 0.920 / 0.879 | 0.918 / 0.893 / 0.905 |
| ssn | 161 | 0.442 / 0.919 / 0.597 | 0.959 / 0.876 / 0.916 |
| credit card number | 64 | 0.775 / 0.859 / 0.815 | 0.883 / 0.828 / 0.855 |
| date of birth | 240 | 0.704 / 0.950 / 0.809 | 0.836 / 0.871 / 0.853 |
| ip address | 97 | 0.863 / 0.907 / 0.884 | 0.915 / 0.670 / 0.774 |
| **micro average** | 964 | **0.657 / 0.929 / 0.770** | **0.774 / 0.878 / 0.823** |
| micro average, overlap match | | 0.674 / 0.953 / 0.790 | 0.788 / 0.898 / 0.839 |

Latency per request through the full pipeline, one request at a time, after warm-up:

| Model | Input | Mean | p50 | p95 | Windows |
|---|---|---:|---:|---:|---:|
| `fastino/gliner2-privacy-filter-PII-multi` | single documents (median 188 chars) | 12.8 ms | 12.5 ms | 15.3 ms | 1.0 |
| `urchade/gliner_multi_pii-v1` | single documents (median 188 chars) | 15.5 ms | 15.1 ms | 19.5 ms | 1.0 |
| `fastino/gliner2-privacy-filter-PII-multi` | 25 documents joined (median 5,544 chars) | 74.3 ms | 71.6 ms | 84.8 ms | 5.3 |
| `urchade/gliner_multi_pii-v1` | 25 documents joined (median 5,544 chars) | 84.4 ms | 84.1 ms | 100.5 ms | 6.5 |

So: **~13 ms per request on an RTX 5080** for a short document with the default model.

How to read this:

- **GLiNER2-PII has the higher recall (0.93 vs 0.88); the urchade model has the
  higher precision and the higher F1 on this dataset.** For redaction, recall
  is the number that corresponds to leaked PII.
- Most of GLiNER2-PII's precision gap is one label. Of its 187 false-positive
  `ssn` predictions, 105 are strings the dataset annotates as
  `medical_record_number` and another 39 are customer or employee IDs: real
  identifiers, but types that were not among the seven labels offered. The
  benchmark scores only those seven, so it penalizes a model for tagging
  out-of-scope PII with the nearest label it was given.
- Every dataset document fits in one window, so the first table does not
  exercise windowing. The joined-document run does (5 to 7 windows per text)
  on the same 964 gold entities: recall is 0.926 for GLiNER2-PII and 0.831 for
  urchade, against 0.929 and 0.878 unjoined. The urchade drop is the model
  responding to longer context, not entities lost in merging: shrinking its
  windows to 64 tokens brings recall back to 0.878, while GLiNER2-PII stays
  between 0.915 and 0.926 at every window size tried. Smaller windows cost
  both models precision. Full tables are in [`bench/results/`](bench/results/).
- Caveats: one dataset, one seed, 500 documents, one threshold (0.5), and
  label prompts that were not tuned for either model. Gold offsets are
  reconstructed by locating each annotated string in the text (the dataset has
  no offsets); see [DESIGN.md](DESIGN.md#benchmark-methodology).

Reproduce with `just bench` and `just bench --concat 25`.

## How it works

```
text ─► backend.tokenize ─► plan_windows ─► backend.predict ─► shift offsets ─► merge_spans ─► spans ─► redact
        (model's own        (subword-        (batched, per     (+ window.start)  (IoU dedup,
         word splitter)      budgeted)        window)                             label conflicts)
```

Each stage is its own module under `src/pii_extract/`:

| Module | Responsibility |
|---|---|
| `spans.py` | The one canonical `Span` type and the offset invariant |
| `chunking.py` | Sliding windows over token offsets (pure arithmetic) |
| `backends/` | Backend protocol, per-instance model session, GLiNER2 and GLiNER adapters |
| `dedup.py` | The only place spans are deduplicated or overlap is resolved |
| `redaction.py` | `mask`, `hash`, `placeholder` strategies and the splice |
| `pipeline.py` | Wires the stages together |
| `config.py`, `api.py`, `cli.py` | Settings precedence, FastAPI service, CLI |

[DESIGN.md](DESIGN.md) explains each decision, including the offset-mapping
approach and the cross-window dedup rule. [NEXT_STEPS.md](NEXT_STEPS.md) lists
what is left.

## Development

```bash
just            # list recipes
just ci         # what GitHub Actions runs: ruff + pure-logic tests
just fmt        # ruff --fix and format
```

Tests under `tests/unit` need no model and run in CI on every push. Tests under
`tests/model` carry the `model` marker and skip themselves when torch or the
weights are unavailable.

## Credits

- **GLiNER2-PII**, Fastino: [`fastino/gliner2-privacy-filter-PII-multi`](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi), Apache-2.0, via the [`gliner2`](https://github.com/fastino-ai/GLiNER2) package.
- **GLiNER multi PII v1**, Urchade Zaratiana et al.: [`urchade/gliner_multi_pii-v1`](https://huggingface.co/urchade/gliner_multi_pii-v1), Apache-2.0, via the [`gliner`](https://github.com/urchade/GLiNER) package.
- **Gretel PII masking dataset**: [`gretelai/gretel-pii-masking-en-v1`](https://huggingface.co/datasets/gretelai/gretel-pii-masking-en-v1), Apache-2.0, synthetic. Used only by the benchmark; not redistributed here.

## License

Apache-2.0. See [LICENSE](LICENSE).
