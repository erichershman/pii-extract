# Next steps

Remaining work, in priority order. Each item says where the change goes and
how to know it is done, so a later session can start without rediscovery.
Background for every decision referenced here is in [DESIGN.md](DESIGN.md).

State: the core pipeline, both backends, schema packs with validators, the
API, CLI, container, CI workflow, and a benchmark with an ablation mode are
working and tested (258 pure-logic tests, 21 model tests). The repository is
on GitHub and the Actions workflow passes there.

Done since the first version of this list: cross-label conflicts are
fail-closed (the winner absorbs the loser), redaction covers the union of
everything flagged, and schema packs with validators exist.

## 1. Detection quality for redaction

**1a. Offer the models more identifier labels by default.**
The ablation shows that adding "medical record number" raises GLiNER2-PII's
`ssn` precision from 0.44 to 0.74 and overall F1 from 0.770 to 0.819 without
costing recall. `pii-extended` already carries that label. Decide whether it
should be the default pack, and test `customer id` and `employee id` the same
way (`just bench --extra-labels "customer id" "employee id"`).

**1b. Native label vocabularies.**
GLiNER2-PII was trained on 42 specific labels (`phone_number`, `card_number`,
`national_id_number`, ...; listed on its model card) and the pipeline sends
free-text labels. Add `native_labels: dict[backend_name, dict[label, str]]` to
`SchemaPack`, map on the way into `backend.predict` and back on the way out,
and measure with `just bench`.

**1c. Regex backstop.**
Patterns that are cheap and near-certain (emails, Luhn-valid card numbers)
can be injected as candidates with score 1.0 before the merge, so the merge
rules decide how they combine with model output. Benchmark recall for
`credit card number` is 0.86 for the default model. The validators in
`src/pii_extract/schema.py` are the starting point.

**1d. A "relabel" validation policy.**
Today a span that fails validation is dropped. An alternative that keeps
coverage: relabel it to a generic type instead. It needs a decision on the
label name and on how placeholders render it.

**1e. International formats.**
The `ssn` validator is US-shape only, and 68 of 162 gold SSNs in the
benchmark sample are other national formats. Either add per-country
validators or rename the label so the scope is honest.

**1f. Tighten the "clipped" test.**
A fragment that stops short of the window edge is not marked clipped
(DESIGN.md, section 4). Consider treating a span as clipped when it ends
within one token of a cut edge. `_touches_cut_edge` in
`src/pii_extract/pipeline.py` is the only place to change; it would need the
window's token offsets passed in.

## 2. ONNX export and an ONNX backend

Spec Stage 1 defers this and the target summary names ONNX Runtime.

- `gliner` 0.2.30 already contains an export path and an onnxruntime loading
  path (`GLiNER.from_pretrained(..., load_onnx_model=True)`; see
  `_run_torch_onnx_export` in `gliner/model.py`). Start with the urchade
  model: write `scripts/export_onnx.py` around it and confirm the exported
  model reproduces the torch spans.
- `gliner2` 2.1.0 has **no** ONNX export. Exporting GLiNER2-PII means tracing
  the encoder plus span head by hand, including the schema prompt and the
  count-prediction layer. Treat it as a separate, larger task. Community ONNX
  conversions exist on the Hub, but the provenance rule says to build the
  export rather than adopt one.
- Add `OnnxBackend` in `src/pii_extract/backends/onnx_backend.py` implementing
  the existing `Backend` protocol. `tokenize` and `window_budget` can reuse
  `backends/budget.py`; only `predict` and the span decoding are new. Register
  it in `BACKENDS` and add `onnxruntime-gpu` as a new optional extra.
- Done when: a parity test in `tests/model` asserts the ONNX backend returns
  the same span keys as the torch backend on the existing fixtures (scores
  within a tolerance), and the benchmark table has an ONNX row with latency.
- `.gitignore` already excludes `*.onnx` and `models/`.
- Quantization and GPU offload tiers stay out of scope until this lands.

## 3. Performance

Current numbers: about 14 ms per short request and about 18 ms per window on
long documents (RTX 5080, default model, batch size 8).

1. **Measure under load first.** Nothing has been measured with concurrent
   clients beyond a 64-request smoke test. Add a load script (for example
   `oha` or `hey` against `just serve`) reporting requests per second and
   p50/p95/p99 at concurrency 1, 8, and 32. Record cold-start time too.
2. **Cross-request micro-batching.** Inference is serialized per session, so
   concurrent requests queue. A small asyncio queue that gathers windows from
   several requests into one batch (max batch size, max wait of a few
   milliseconds) should raise throughput. It belongs in a new module between
   `api.py` and the backend, not inside `ModelSession`.
3. **Stop tokenizing twice.** `Pipeline.extract` tokenizes the whole text to
   plan windows, then the model library tokenizes each window again. On the
   339k-character test the first pass is 0.73 s of 3.7 s. Either pass
   pre-split words to the model or cache per-word subword counts.
4. **Mixed precision.** Try `torch.autocast` with fp16 on CUDA in the two
   `predict` methods and compare spans against fp32 on the benchmark before
   enabling it by default.
5. **Batching details.** Sort windows by length before batching to cut
   padding, and sweep `batch_size` (8 is a guess, not a measurement).
6. **Image size.** The image is about 12 GB, nearly all torch and CUDA
   libraries. An ONNX Runtime image would be much smaller.

## 4. Broader benchmarks

The current result is one dataset, one seed, 500 documents, one threshold.

1. **More models.** Add a `--model` override per backend to `bench/run.py`
   and run the spec's other candidates, both of which load through the
   existing `gliner` backend and are already in the local Hugging Face cache:
   - `knowledgator/gliner-pii-base-v1.0` (Apache-2.0).
   - `nvidia/gliner-PII` (NVIDIA Open Model License, **not** Apache). Fine to
     benchmark; read the terms before redistributing weights or shipping it
     as a default, and keep it out of the README's provenance line.
2. **Native labels.** Rerun with each model's own label vocabulary (item 1b). The current table uses identical free-text
   prompts for both models, which likely understates GLiNER2-PII.
3. **Threshold sweep.** `--threshold` exists; add a sweep that prints
   precision and recall per label at several thresholds, and choose defaults
   from it. GLiNER2-PII's precision (0.66) suggests its default threshold
   could be higher than 0.5.
4. **A second dataset.** AI4Privacy `pii-masking-300k` ships character
   offsets, which would remove the gold-reconstruction step that the Gretel
   loader needs. Its license is listed on the Hub as "other", so read it
   first. Add `bench/ai4privacy.py` beside `bench/gretel.py`; the scoring in
   `bench/metrics.py` is dataset-independent.
5. **Error analysis as a feature.** The false-positive breakdown quoted in
   the README (GLiNER2-PII's `ssn` errors are mostly medical record numbers)
   came from a throwaway script. Fold it into `bench/run.py` as
   `--confusions` so it is reproducible.
6. **Statistical care.** Several seeds, a larger sample, and confidence
   intervals. Add CPU latency for comparison.

## 5. README and repository polish

1. Done: pushed to GitHub, workflow passing, CI badge in the README.
2. **Demo GIF** (spec Stage 3): record `just test` and a `pii-extract redact`
   call with `vhs` or asciinema, using only the synthetic example text.
3. **Build the Docker image in CI** (the spec's stretch goal). The build needs
   no GPU; use layer caching, since the image is large.
4. A short "choosing a model" section once the broader benchmark exists.
5. `pre-commit` config running ruff, mirroring `just lint`.

## 6. API and library extensions

Smaller items, none blocking.

- **`offset_unit`** request field (`codepoint`, `utf16`, `bytes`), so
  JavaScript clients can slice with the offsets they get back
  (DESIGN.md, section 1). Convert at the API boundary only.
- **Batch endpoint** accepting `texts: [...]`, which pairs naturally with
  micro-batching.
- **Separate `/readyz` from `/healthz`** if preload is ever made optional in
  the container.
- **Request body limit at the ASGI layer.** `max_text_chars` is checked after
  the body is parsed.
- **Structured logs and metrics** (request count, latency, windows per
  request). Never log request text.
- **Exact prompt overhead.** The 16 and 4 token constants in the adapters are
  conservative guesses. Compute the real prompt length by running each
  library's own prompt builder once per label set.
- **Languages without spaces.** `gliner2` has a character-level splitter for
  Chinese and similar scripts; the adapter always uses the model's configured
  splitter, but this path has no test.
- **Label descriptions.** `gliner2` accepts a description per label, which may
  help with ambiguous types such as `ssn` versus other identifiers.
