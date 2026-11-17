# pii-extract

[![CI](https://github.com/erichershman/pii-extract/actions/workflows/ci.yml/badge.svg)](https://github.com/erichershman/pii-extract/actions/workflows/ci.yml)

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
- **Fails closed when redacting**: every character any window flagged is
  replaced, even where detections disagree about boundaries or labels.
- **Schema packs**: named label sets with optional validators (regex shape
  checks for email, SSN and phone; a Luhn check for card numbers) that can be
  switched on per deployment or per request.
- **Swaps models** behind a small backend protocol: GLiNER2-PII by default,
  `urchade/gliner_multi_pii-v1` as the comparison backend.

All examples and test data are synthetic.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/).
The environment is pinned by `uv.lock`.

```bash
just setup          # .venv with torch (CUDA 12.8 wheels), gliner2, gliner, dev tools
just unit           # 258 pure-logic tests, no model needed (under a second)
just model-tests    # 21 tests on real weights (downloads ~1.5 GB on first run)
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

Request fields: `text` (required); `labels`, `threshold` and `validate`
(optional, fall back to server settings); and for `/redact` also `strategy`
and `mask_char`.
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
| `schema_pack` | `PII_SCHEMA_PACK` | `--schema-pack` | `pii` (or `pii-extended`) |
| `labels` | `PII_LABELS` (comma-separated) | `--labels` | the schema pack's labels |
| `validators` | `PII_VALIDATORS` | `--validators` | `off` (or `on`, or a comma-separated list of labels) |
| `threshold` | `PII_THRESHOLD` | `--threshold` | `0.5` |
| `window_tokens` | `PII_WINDOW_TOKENS` | `--window-tokens` | the model's limit |
| `overlap_tokens` | `PII_OVERLAP_TOKENS` | `--overlap-tokens` | `64` |
| `iou_threshold` (extraction only) | `PII_IOU_THRESHOLD` | `--iou-threshold` | `0.5` |
| `batch_size` | `PII_BATCH_SIZE` | `--batch-size` | `8` |
| `strategy` | `PII_REDACTION_STRATEGY` | `--strategy` | `placeholder` |
| `hash_salt` | `PII_HASH_SALT` | `--hash-salt` | random per request |
| `preload` | `PII_PRELOAD` | | `false` (the container sets it to `1`) |
| `max_text_chars` | `PII_MAX_TEXT_CHARS` | | `1000000` |

Set `PII_HASH_SALT` to a secret if `hash` redactions must be stable across
requests. Without it each request gets a fresh random salt, so digests are
consistent within a document but cannot be linked across documents.

## Schema packs and validators

A schema pack is a label set plus optional per-label validators.

| Pack | Labels | Validators |
|---|---|---|
| `pii` (default) | person, email, phone number, credit card number, ssn | email, phone number, credit card number, ssn |
| `pii-extended` | the above plus date of birth, ip address, medical record number | the above plus ip address |

| Validator | Passes when |
|---|---|
| email | the text is one well-formed address |
| phone number | only phone characters, 7 to 15 digits, optional extension |
| ssn | nine digits grouped 3-2-4 with consistent separators (US shape; allocation rules are not enforced) |
| credit card number | 12 to 19 digits, optionally grouped, passing the Luhn checksum |
| ip address | a valid IPv4 or IPv6 address |

A span whose label has an enabled validator is dropped if its text fails the
check. Labels are matched loosely (`phone_number`, `Phone Number` and
`telephone number` all reach the phone validator), and labels without a
validator are never filtered. Validators run after windows are merged, so
they see whole entities.

**Validators are off by default.** Switch them on per deployment or per request:

```bash
PII_VALIDATORS=on just serve                        # all validators
PII_VALIDATORS="ssn,credit card number" just serve  # only these labels
uv run pii-extract extract --validators on "Card 4111 1111 1111 1112"   # fails Luhn: dropped
curl -s -X POST http://localhost:8000/extract -H 'Content-Type: application/json' \
  -d '{"text": "Card 4111 1111 1111 1112", "validate": true}'
```

`"validate": true` runs every validator in the pack, `false` none; omitting it
uses the server setting. `/healthz` reports the pack and the validators in force.

### Why the defaults favor recall

The two kinds of error are not equally bad for a redactor. **A missed span is
a leak: the PII leaves in the output. An extra span is over-redaction: some
harmless text is replaced.** The first cannot be undone downstream; the second
costs a little readability. So every default leans toward flagging more:

| Default | Why |
|---|---|
| GLiNER2-PII as the model | Highest recall of the two (0.93 vs 0.88 below), at lower precision |
| Validators off | A validator can only remove detections, and a malformed value is often still sensitive (a card number with one wrong digit fails Luhn). On the benchmark, validators cut `ssn` recall from 0.92 to 0.53 |
| Redaction covers the union | Where detections disagree on boundaries or labels, `/redact` replaces every flagged character |
| Overlapping windows | An entity cut by one window is seen whole by the next |

If you are extracting entities for analytics rather than redacting, the
trade-off reverses: turn validators on, and consider the comparison backend.

### `/extract` and `/redact` merge differently

When two detections of the same entity overlap, `/extract` keeps the more
confident one's boundaries, while `/redact` covers the union. For
`Jane Doe` (0.95) and `Jane Doe Smith` (0.60), `/extract` returns `Jane Doe`
and `/redact` replaces `Jane Doe Smith`. The spans in a `/redact` response can
therefore be wider than `/extract`'s for the same text.

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
| `fastino/gliner2-privacy-filter-PII-multi` | single documents (median 188 chars) | 13.9 ms | 13.6 ms | 16.8 ms | 1.0 |
| `urchade/gliner_multi_pii-v1` | single documents (median 188 chars) | 16.0 ms | 15.5 ms | 20.8 ms | 1.0 |
| `fastino/gliner2-privacy-filter-PII-multi` | 25 documents joined (median 5,544 chars) | 85.3 ms | 72.7 ms | 179.2 ms | 5.3 |
| `urchade/gliner_multi_pii-v1` | 25 documents joined (median 5,544 chars) | 105.6 ms | 86.7 ms | 202.1 ms | 6.5 |

So: **~14 ms per request on an RTX 5080** for a short document with the
default model. The joined run has only 20 texts, so its p95 is close to its
maximum and varies between runs.

### Validators and an extra label

`just bench --ablation` reruns the same 500 documents four ways. `ssn` and the
micro average are scored on the same seven labels in every row, precision /
recall / F1:

| Model | Variant | ssn | Micro average (7 labels) |
|---|---|---|---|
| GLiNER2-PII | baseline | 0.442 / 0.919 / 0.597 | 0.657 / 0.929 / 0.770 |
| GLiNER2-PII | validators on | 0.955 / 0.528 / 0.680 | 0.759 / 0.844 / 0.800 |
| GLiNER2-PII | + medical record number | 0.738 / 0.894 / 0.809 | 0.739 / 0.920 / **0.819** |
| GLiNER2-PII | validators on + medical record number | 0.964 / 0.503 / 0.661 | 0.769 / 0.835 / 0.801 |
| urchade | baseline | 0.959 / 0.876 / 0.916 | 0.774 / 0.878 / **0.823** |
| urchade | validators on | 0.976 / 0.503 / 0.664 | 0.772 / 0.799 / 0.785 |
| urchade | + medical record number | 0.953 / 0.876 / 0.913 | 0.774 / 0.880 / 0.823 |
| urchade | validators on + medical record number | 0.976 / 0.503 / 0.664 | 0.773 / 0.801 / 0.787 |

- **Giving GLiNER2-PII the right label fixes most of its `ssn` problem without
  costing recall.** With "medical record number" offered, `ssn` precision goes
  from 0.44 to 0.74, recall from 0.92 to 0.89, and overall F1 from 0.770 to
  0.819, level with urchade's 0.823. The new label itself scores
  0.869 / 0.863 / 0.866 for GLiNER2-PII and 0.930 / 0.750 / 0.830 for urchade
  (284 gold spans). urchade is unchanged: it was not confusing the two.
- **Validators buy precision with recall.** `ssn` precision reaches 0.96 for
  GLiNER2-PII, but `ssn` recall falls from 0.92 to 0.53 for both models.
  The cause is the data, not a bug: only 94 of the 162 `ssn` annotations in
  the sample have the US 3-2-4 shape. The rest are nine digits in groups of
  three or UK National Insurance style (`AA 99 99 99 A`), which a US-shape
  check rejects. Overall F1 rises for GLiNER2-PII (0.770 to 0.800) because it
  had many false positives to remove, and falls for urchade (0.823 to 0.785)
  because it had few.
- For redaction, the extra-label route is the better one: it is the only
  variant that raises precision while holding recall at 0.92.

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

Reproduce with `just bench`, `just bench --concat 25` and `just bench --ablation`.
Other switches: `--validators on` and `--extra-labels "medical record number"`.

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
| `schema.py` | Schema packs: label sets and per-label validators |
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
