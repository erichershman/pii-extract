# Next steps

Remaining work, in priority order. Each item says where the change goes and
how to know it is done, so a later session can start without rediscovery.
Background for every decision referenced here is in [DESIGN.md](DESIGN.md).

State: the core pipeline, both backends, schema packs with validators, the
API, CLI, container, CI workflow, and a benchmark with an ablation mode are
working and tested (266 pure-logic tests, 27 model tests). The repository is
on GitHub and the Actions workflow passes there.

Done since the first version of this list: cross-label conflicts are
fail-closed (the winner absorbs the loser), redaction covers the union of
everything flagged, and schema packs with validators exist.

## 1. Detection quality for redaction

**1a. Hybrid regex + NER for structured types.**
`ssn`, `email`, `credit card number` and `phone number` have shapes a regex
can find without a model. Run regex detectors alongside the model and union
the two, so each covers the other's misses.
- New module `src/pii_extract/detectors.py`: one finder per structured type
  returning spans with offsets into the original text. The validators in
  `src/pii_extract/schema.py` are the starting point (they check a span;
  a finder searches for one), including the Luhn check to keep card matches
  precise.
- Feed regex matches into `merge_spans` as extra `Candidate`s rather than
  appending them afterwards, so the existing rules decide how they combine
  with model spans. On the redaction path the merge already unions everything
  flagged. Decide the score regex candidates carry (1.0 makes them win label
  conflicts) and make the feature a setting, off or on per label, like
  validators.
- Benchmark three ways on the same sample, per label and overall: **model
  only**, **regex only**, and **union**. Add a `--detectors model|regex|union`
  switch to `bench/run.py`; regex-only needs no backend. The number to watch
  is recall on the structured labels. Today GLiNER2-PII's is 0.86 for card
  numbers and 0.92 for phones and SSNs.
- Expect regex-only precision to be poor for bare digit runs (any nine digits
  look like an SSN), which is the argument for union over replacement, and
  for requiring context or a checksum where one exists.

**1b. More identifier labels.**
"medical record number" is now in the default `pii` pack. Test `customer id`
and `employee id` the same way
(`just bench --extra-labels "customer id" "employee id"`), and rerun the
extra-label and validator combination, which predates the `ssn` validator fix.

**1c. Native label vocabularies.**
GLiNER2-PII was trained on 42 specific labels (`phone_number`, `card_number`,
`national_id_number`, ...; listed on its model card) and the pipeline sends
free-text labels. Add `native_labels: dict[backend_name, dict[label, str]]` to
`SchemaPack`, map on the way into `backend.predict` and back on the way out,
and measure with `just bench`.

**1d. A "relabel" validation policy.**
Today a span that fails validation is dropped. An alternative that keeps
coverage: relabel it to a generic type instead. It needs a decision on the
label name and on how placeholders render it.

**1e. Card numbers and Luhn.**
With validators on, `credit card number` recall falls from 0.86 to 0.56 on
the benchmark because many synthetic card numbers fail Luhn. Decide whether
the card validator should be shape-only by default with Luhn as an opt-in.

**1f. Tighten the "clipped" test.**
A fragment that stops short of the window edge is not marked clipped
(DESIGN.md, section 4). Consider treating a span as clipped when it ends
within one token of a cut edge. `_touches_cut_edge` in
`src/pii_extract/pipeline.py` is the only place to change; it would need the
window's token offsets passed in.

## 2. ONNX: from export to a backend

Done: `just export-onnx` exports `urchade/gliner_multi_pii-v1` (fp32) and
`just onnx-parity` shows identical spans against PyTorch with a max score
delta of 1.04e-04 (DESIGN.md, section 12). Remaining, in order:

1. **`OnnxBackend` behind the protocol.** Add
   `src/pii_extract/backends/onnx_backend.py` and register it in `BACKENDS`.
   The parity test already shows `GlinerBackend`'s `tokenize`,
   `window_budget` and span decoding work unchanged on the ONNX-loaded model,
   so the new backend is mostly a loader
   (`GLiNER.from_pretrained(dir, load_onnx_model=True, load_tokenizer=True)`)
   plus a setting for the model directory. Then move the parity test onto it.
2. **Execution-provider fallback, CUDA to CPU.** The `onnx` extra installs the
   CPU build of ONNX Runtime. Add `onnxruntime-gpu`, request
   `CUDAExecutionProvider` first and fall back to `CPUExecutionProvider`,
   report the provider actually in use on `/healthz`, and rerun parity on the
   CUDA provider. Check that the CUDA 12.8 libraries torch bundles satisfy it.
3. **Latency.** Add the ONNX backend to `bench/run.py` and report ms per
   request beside the PyTorch rows, on GPU and CPU.
4. **fp16 and int8 comparison.** Export fp16 and a quantized graph
   (`export_to_onnx(quantize=True)`), and report size, latency, parity
   against fp32 (span mismatches, max score delta) and benchmark F1 for each.
5. **GLiNER2 export.** `gliner2` 2.1.0 has no export path. Evaluate the
   `gliner2-onnx` package for exporting GLiNER2-PII, with the same parity
   test as the acceptance bar. Mind the provenance rule: build the export
   from the public weights rather than adopting a third-party converted model.

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
2. **Native labels.** Rerun with each model's own label vocabulary (item 1c). The current table uses identical free-text
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
