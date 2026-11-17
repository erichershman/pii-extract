# Design

This document records each non-obvious decision in pii-extract: what was
chosen, what else was considered, and why. Numbers quoted here were measured
on the development machine (RTX 5080, torch 2.11.0+cu128, `gliner2` 2.1.0,
`gliner` 0.2.30) unless stated otherwise.

## The pipeline

```
text ─► tokenize ─► plan windows ─► predict (batched) ─► shift offsets ─► merge ─► validate ─► spans ─► redact
        backend      chunking.py     backend              pipeline.py      dedup.py  schema.py           redaction.py
```

Each stage lives in one module and does one thing. `pipeline.py` is the only
file that knows the order, and it is short. The stages
with no model in them (windowing, merging, redaction, offsets) are pure
functions over plain data, which is why 270 of the 298 tests run without torch.

## 1. One canonical span schema

**Decision.** A single frozen Pydantic model, `Span(start, end, label, score, text)`,
is used by backends, the merge stage, redaction, the API responses, and the
CLI output. Identity is `span.key == (start, end, label)` and nothing else
defines span equality.

- Offsets are half-open `[start, end)` Python `str` indices, which are Unicode
  code points.
- `span.text` is always read from the source string (`span_from_offsets`). A
  model's own `text` field is never copied into a span, only compared.
- The invariant `source[start:end] == span.text` is checked by `verify_spans`
  at the pipeline's exit and again inside `redact`.

**Alternatives.** A dataclass internally plus a separate Pydantic model at the
API boundary. That is two schemas to keep in sync, and Pydantic is already a
core dependency, so one type everywhere was simpler. Validation cost per span
is negligible next to a forward pass.

**Trade-off to know.** Code-point offsets match Python, but JavaScript indexes
strings in UTF-16 code units, so a JS client slicing text that contains emoji
or other astral characters would be off. Offering `offset_unit=utf16|bytes` is
listed in NEXT_STEPS.

## 2. Offset mapping

This was the top priority, so it is designed to be correct by construction
rather than by correction.

**Decision.** The input string is never modified. Not stripped, not
NFC-normalized, not BOM-stripped, not newline-translated. A window is an
exact slice `text[w.start:w.end]` whose bounds are token edges. A span the
model reports at `[s, e)` in that slice is therefore at
`[w.start + s, w.start + e)` in the original. That one addition
(`Span.shifted`) is the entire mapping.

Because a slice contains whatever characters were there, a BOM, CRLF, NBSP,
tab runs, emoji, and combining marks need no special handling. They are not
"handled"; they are simply never touched.

**Alternative rejected: normalize, then map back.** Cleaning the text first
(collapse whitespace, strip the BOM, NFC) and translating offsets back through
an alignment table is the common approach and the common source of drift
bugs. It would add a table that must be exactly right for every Unicode edge
case, to solve a problem the models do not have: both tokenize with regex
splitters that already skip whitespace.

**Where drift can still enter, and the guard for each:**

| Risk | Guard |
|---|---|
| The model library edits its input | `checked_span` in the adapter compares the model's reported text to the slice at its reported offsets and raises `OffsetError` on any mismatch |
| The CLI reads a file in text mode | Input is read as bytes and decoded with plain `utf-8` (not `utf-8-sig`), so the BOM stays a character and CRLF stays two characters |
| A stage hands on stale offsets | `verify_spans` at the end of `Pipeline.extract` and at the start of `redact` |
| JSON transport | None needed: JSON strings decode to the same code points |

The policy on a mismatch is to **fail closed**. The service answers 500 with
no spans rather than return offsets it could not verify, because a caller that
redacts by wrong offsets destroys the wrong characters and leaves the PII in
place. That is worse than an error.

**This guard caught a real bug.** Running the benchmark raised
`OffsetError: span [142, 156) is outside text of length 155`. The cause:
`gliner2` appends a `"."` to any input that does not end in `.`, `!` or `?`
(`processor._normalize_text`), and the model can include that synthetic token
in an entity, reporting `end == len(text) + 1`. Since most windows end
mid-sentence, this affects windowed inference in general, not a corner case.
The fix is exact rather than heuristic, because the edit is known precisely:
the character is always appended, never inserted, so every other offset is
unaffected, and `_drop_synthetic_terminator` cuts back exactly that one
position, and only when the text lacks a terminator. Any other out-of-range
offset is still rejected. Covered by four unit tests against a stub that
mimics the behaviour and one test on the real models.

**Tests.** Round-trips through a fixture containing a BOM, NBSP, em space,
CRLF, tabs, vertical tab, form feed, emoji, combining marks, a zero-width
space, Hebrew and CJK, at five window sizes; a seeded generator that builds
150 random documents and asserts the returned spans equal the planted ones
exactly; and BOM/CRLF checks through the API and CLI.

## 3. Windowing

**What the models actually do with long input** (probed, not assumed):

| | `gliner` (urchade) | `gliner2` (GLiNER2-PII) |
|---|---|---|
| Enforced cap | `config.max_len` = 384 **words**; input past it is **silently dropped** (a `warnings.warn`, no error) | none |
| Encoder | mdeberta-v3-base, 512 positions | mdeberta-v3-base, 512 positions |
| Longest entity | 12 words | 8 words |

For `gliner`, an unwindowed 756-word text found 2 of 4 emails: everything
after word 384 was never seen. For `gliner2`, nothing is truncated, but one
unwindowed pass over a 41,589-character text (9,304 subwords) took 376 seconds
and found all 40 emails but 0 of 40 phone numbers (one measurement; the time
is probably VRAM exhaustion). Windowed, a 339,390-character text takes
3.7 seconds over 206 windows and returns 1,500 of 1,500 emails and phones.

**Decision: budget windows in subword tokens, cut on word boundaries.**
The backend supplies `Token(start, end, cost)` from the model's own word
splitter, where `cost` is that word's subword count. `plan_windows` fills each
window up to a budget of total cost and never splits a token.

- The encoder's limit is in subwords, and PII-dense text is exactly where
  words and subwords diverge: `jane.doe@example.com` is one word to GLiNER2's
  splitter and 8 subwords. A word-count budget would overrun on the inputs
  this tool exists for.
- Both libraries tokenize word by word, so per-word costs are exactly
  additive. The budget is not an estimate.
- Using the model's own splitter means a window boundary is always a boundary
  the model would also see.

**The budget is read from the model, not hardcoded:**
`encoder.max_position_embeddings − label-prompt cost − fixed prompt overhead`.
The label prompt shares the encoder's input with the text, so more labels
leave less room: 480 tokens for GLiNER2-PII with five short labels, 336
with forty. For `gliner` the result is also capped at `config.max_len`; since
every word costs at least one subword, a subword budget of at most 384 can
never exceed 384 words, so one number enforces both limits. The fixed overhead
(16 and 4 tokens) is a conservative constant, not derived from the prompt
template; see NEXT_STEPS.

`window_tokens` in settings can only shrink the window below the model's
limit, never raise it.

**Overlap** defaults to 64 tokens and is configurable. Its job: an entity
that straddles one window's edge and is no longer than the overlap is seen
whole by the next window. If the configured overlap is at least half the
window it is capped at half, so a misconfiguration slows nothing to a crawl.

**Edge cases.** A single token that costs more than the whole budget gets a
window to itself (it cannot be split). Empty or whitespace-only text produces
no windows and never reaches the model. Planning always advances by at least
one token, so it terminates.

**Alternatives.**
- *Sentence-aware windows.* Better context at boundaries, but needs a sentence
  splitter (a dependency and another source of offset bugs) and does not
  remove the need for a hard budget. Overlap gets most of the benefit.
- *Character-count windows.* Simple, but would cut tokens in half and has no
  relation to the model's limit.
- *The library's built-in long-text helpers* (`gliner2` has `extract_entities_long`).
  They are word-budgeted, exist for one backend only, and merge with their
  own rule. The spec asks for windowing and dedup as one tested, reusable
  stage shared by all backends.

**Measured trade-off.** On joined benchmark documents (overlap 32), shrinking
windows from the model limit to 64 tokens raised urchade's recall from 0.825 to 0.878 and
lowered its precision from 0.891 to 0.699; GLiNER2-PII's recall stayed between
0.915 and 0.926 while its precision fell from 0.756 to 0.562. Larger windows
give the models more context and are also faster, so the default is the
model's limit.

## 4. Merging across windows (the dedup rule)

Overlapping windows report the same entity twice, sometimes with different
boundaries or different labels, and a window that cuts an entity in half
reports a fragment. `dedup.merge_spans` is the only place this is resolved.
Its output is sorted and **flat** (no two spans overlap), which is what
redaction requires.

Each candidate carries one extra bit: **`clipped`**, set when the span touches
an edge of its window where the document continues. A clipped span may be a
truncated fragment.

Rules, in order:

1. **Same label, IoU ≥ threshold (default 0.5): duplicates.** They are the
   same entity seen twice. Keep the single best candidate *with its
   boundaries* and drop the rest.
2. **Same label, overlapping, IoU < threshold: boundary disagreement.** The
   windows agree these characters are PII of this type but not where the
   entity ends. Merge into the **union**, score = max.
3. **Different labels, any overlap: conflict.** The best candidate wins the
   *label*. The loser is not discarded: any of its characters that no winner
   covers are absorbed into the winner beside them, so the winner grows to
   the union of the two spans.

"Best" is a total order: **not clipped**, then higher score, then longer span,
then earlier start, then label alphabetically. Being total makes the result
independent of input order, which a seeded shuffle test asserts.

**Why IoU does real work here.** Rules 1 and 2 behave differently, and the
threshold is what chooses between them. `"Jane Doe"` (0.95) and
`"Jane Doe Smith"` (0.6) have IoU 0.57: at threshold 0.5 they are duplicates
and the confident, tighter span wins; at 0.6 they are a disagreement and
become the union. High overlap means "trust the better detection's
boundaries", so a sloppy duplicate that drags in a trailing comma does not
widen the result. Low overlap means "do not guess", so no flagged character
is left exposed. For a redactor, rule 2 is the fail-closed choice.

**Why `clipped` outranks score.** Models are often *more* confident about a
clean-looking fragment than about the whole entity. If window A ends inside a
card number it may report `4111 1111 1111` at 0.99 while window B reports the
full number at 0.90. IoU is 0.74, so they are duplicates, and score alone
would keep the fragment and leak the last four digits. Putting "seen whole"
ahead of score fixes this. The fake model in the tests deliberately scores
fragments higher (0.99 vs 0.90) to prove the rule is what saves the result,
and `test_entity_split_across_a_window_boundary` checks the same thing on the
real models.

**The cross-label rule: one label, union of characters.** A span cannot carry
two labels, so something must win, and confidence is the only evidence
available. But losing the label is not a reason to lose coverage. The winner
keeps its label and score and is extended over whatever the loser flagged
that no winner already covers:

| Winner | Loser | Result |
|---|---|---|
| `jane.doe@example.com` email 0.99 | `jane.doe` person 0.6 (nested) | email, unchanged |
| `Jane Doe` person 0.9 | `Doe Smith` organization 0.6 (partial) | person `Jane Doe Smith` |
| `Doe` person 0.95 | `Jane Doe Smith` organization 0.5 (contains the winner) | person `Jane Doe Smith` |

The first version dropped the loser whole, which exposed `Smith` in the second
row and both outer words in the third. That was the one place the merge was
not fail-closed, and it is now covered by regression tests.

Two details make this well defined:

- *A loser between two winners.* If B loses to A on its left and C on its
  right, the part of B that neither covers goes to the higher-priority winner.
  The other winner is left exactly as detected.
- *Several losers.* Losers are absorbed best-first, each measured against the
  winners **as already grown**, so two losers can never push two winners into
  each other. The output stays flat; a test asserts it.

The cost is honest to state: in a conflict, the winner's span can be wider
than what the model reported under that label. Extraction consumers get a
span that says "person" over characters another candidate called
"organization". For a redactor that is the right way to be wrong.

### Extraction and redaction merge differently

Rules 2 and 3 never drop a flagged character. Rule 1 does, deliberately: when
two same-label detections nearly coincide, it keeps the better one's
boundaries and treats the weaker one's extra characters as noise. Whether
that is right depends on what the spans are for, so the two entry points use
the same merge code with different thresholds:

| | `extract` (`/extract`, `pii-extract extract`) | `redact` (`/redact`, `pii-extract redact`) |
|---|---|---|
| Goal | The best estimate of each entity | Nothing flagged survives |
| IoU threshold | configured `iou_threshold`, default 0.5 | fixed at 1.0 (`COVERAGE_IOU_THRESHOLD`) |
| Near-duplicates, same label | best detection's boundaries | union |
| `Jane Doe` 0.95 + `Jane Doe Smith` 0.60 | `Jane Doe` | `Jane Doe Smith` |
| Guarantee | flat, tight spans | flat spans covering **every** character any window flagged |

At threshold 1.0 only exact duplicates count as duplicates; everything else
overlapping is unioned (same label) or absorbed (different labels). So the
redaction path has a property that is easy to state and to test: the set of
replaced characters equals the set of characters flagged by any window.
Three tests assert it (a randomized merge test, a randomized pipeline test
over real windows, and API and CLI tests showing the two paths disagree on the
same input).

Consequences worth knowing:

- The spans in a `/redact` response can be wider than the spans `/extract`
  returns for the same text. That is intended; they answer different questions.
- `iou_threshold` configures extraction only. Redaction ignores it, so no
  setting can make redaction leak.
- The wider span keeps the best candidate's label and the highest score.
- Validators, when switched on, run after the merge on either path. On the
  redaction path that means a union span is what gets validated.

**Alternative considered:** one merge for both with `iou_threshold` defaulting
to 1.0. Simpler, but it would make every extraction span absorb its sloppiest
duplicate (trailing commas, neighbouring words), costing exact-match accuracy
for callers who want entities, not coverage.

**Residual risk.** A fragment that does not reach the window edge is not
marked clipped. A model that reports only the first two groups of a card
number three tokens before the cut would compete on score alone.

**Cost.** Only overlapping spans can interact, so candidates are first grouped
into clusters of transitively overlapping spans (one sort, one sweep) and the
pairwise rules run inside each cluster. Total cost is O(n log n); 30,000
candidates merge in about 0.1 s. The first version compared every pair, which
would have been about a billion comparisons for the same input.

**Alternatives.**
- *Exact-match dedup only* (`set` of keys). Fails as soon as two windows
  disagree by one character.
- *Keep the higher score, always.* Leaks the tail of split entities, as above.
- *Window ownership*: each window is authoritative only for its central
  region. Simple and needs no IoU, but it discards a detection made near an
  edge even when the neighbouring window missed the entity, which costs recall.
- *Always union same-label overlaps.* Fail-closed, but every sloppy duplicate
  widens the span; rule 1 exists to prevent that.

## 5. Backends and sessions

**Decision.** `Backend` is a `typing.Protocol` with two real implementations,
which is the point at which the spec allows an abstraction. It covers what the
pipeline cannot know: `tokenize`, `window_budget`, `predict`, plus lifecycle
(`load`, `close`, `loaded`, `device`). It is structural, so the test fake
implements it without inheriting anything.

The two libraries really do differ, which is why the adapter layer exists:

| | `gliner2` | `gliner` |
|---|---|---|
| Call | `batch_extract_entities(texts, labels, include_spans=True, include_confidence=True)` | `inference(texts, labels, flat_ner=True)` |
| Result | `{"entities": {label: [{text, confidence, start, end}]}}` | `[{start, end, text, label, score}]` |
| Offsets without flags | none (strings only) | always |
| Input edits | appends `"."` | none found |

**Sessions.** `ModelSession` owns one loaded model. All state is on the
instance: there is no module-level model, no class-level singleton, and no
global lock. Loading is lazy and idempotent, `close()` releases the weights
and empties the CUDA cache, the session can be reloaded afterwards, and it
works as a context manager. Device is `auto` (CUDA if `torch.cuda.is_available()`,
else CPU) or whatever string is configured.

**Concurrency.** Each session has its own `RLock`, used for two things:
making the lazy load happen exactly once when several first requests race
(double-checked), and serializing forward passes on that one model. The lock
is per instance, so two sessions never wait on each other; a test runs two
sessions on twelve threads and asserts both that each session's peak
concurrency is 1 and that the two sessions did overlap.

Why serialize at all: one GPU executes one batch at a time, and both libraries
keep per-call state on the model object (`gliner2` switches a processor mode
and caches a collator on `self`). Concurrent forward passes would buy almost
no throughput and risk interleaving that state. Throughput comes from batching
the windows of one request into one call, which the pipeline does.

**Alternatives.**
- *A global lock or module singleton.* Simplest, and what the spec rules out:
  it couples every model in the process and makes tests order-dependent.
- *Lock-free concurrent inference.* Would need a proof that neither library
  mutates shared state during `predict`; they do.
- *A cross-request micro-batching queue.* The right way to raise throughput
  under load (one batch serves many requests). It is real work with its own
  latency trade-off and is in NEXT_STEPS.
- *Several Uvicorn workers.* Each would load its own copy of the weights into
  VRAM. The container runs one worker; scale with replicas.

**Torch is optional.** `torch`, `gliner`, and `gliner2` are in a `models`
extra and imported inside the loaders. Importing `pii_extract`, running the
pure-logic tests, and CI need none of them.

## 6. Redaction

**Decision.** `redact(text, spans, strategy)` does the splicing once. A
strategy is a `Protocol` with a single method, `replacement(span) -> str`, and
is registered by name. Three ship, so the abstraction has three users.

- **`mask`**: every character becomes `*` (configurable). Length-preserving,
  so offsets in the redacted text equal offsets in the original.
- **`hash`**: `[EMAIL:9f2c1a7b3d4e]`, HMAC-SHA256 keyed with the salt,
  truncated to 12 hex characters. HMAC rather than `sha256(salt + value)`
  because it is the standard keyed construction. The label is part of the
  message, so the same digits hash differently as a phone and as an account
  number. With no salt configured, a random one is generated per strategy
  instance: digests are consistent within a document and unlinkable across
  documents, which is the safe default. PII values are low-entropy (there are
  only 10^10 ten-digit phone numbers), so a digest is exactly as private as
  the salt; it is treated as a secret and kept out of `repr`.
- **`placeholder`**: `[PHONE_1]`. Numbered per label, per unique value, in
  document order. The redactor sorts spans first, so numbering does not depend
  on the order spans were passed in.

**"Same value"** means equal after collapsing whitespace and case folding, so
`Jane  Doe` and `jane doe` share a placeholder. Format-level equivalence
(`555-867-5309` vs `(555) 867-5309`) is deliberately not attempted; that needs
per-label normalizers, which belong to the schema-pack work.

**Safety checks.** `redact` refuses overlapping spans and spans whose text
does not match the source at their offsets, because splicing with either would
silently corrupt the output. Every replacement is reported with its position
in both the original and the redacted text.

**Scope of consistency.** A strategy instance holds the numbering, and the API
builds a fresh one per request, so `[PHONE_1]` in two responses are unrelated.
Reusing one instance across documents (library use) keeps numbering consistent
across them; a test covers both.

## 7. Schema packs and validators

**Decision.** A `SchemaPack` is a name, a label tuple, a map from label to
validator, and an alias table. Two ship: `pii` (person, email, phone number, credit card number, ssn,
medical record number) and `pii-extended` (adds date of birth and ip address).
"medical record number" has no validator; it is in the default set so the
model has somewhere to put record identifiers instead of calling them SSNs
(on the benchmark that alone lifts GLiNER2-PII's `ssn` precision from 0.44 to
0.74). The
pipeline takes its default labels from the pack, and explicit `labels`
override them.

**A validator is a predicate on the detected text** (`Callable[[str], bool]`).
The spec says "post-processors"; validators are the only kind that exists, so
that is the type. When a transforming post-processor appears (trimming a
trailing comma, say), the type widens then.

| Validator | Rule | Deliberately not checked |
|---|---|---|
| email | regex for one well-formed address | deliverability |
| phone number | phone characters only, 7 to 15 digits (15 is the E.164 limit), optional extension | country-specific plans |
| ssn | nine digits in any grouping (dashed, spaced, undelimited), or the UK National Insurance shape | SSA allocation rules: synthetic fixtures use the reserved ranges (000, 9xx) on purpose, and 9xx numbers are ITINs, which are just as sensitive |
| credit card number | 12 to 19 digits, spaces or hyphens allowed, Luhn checksum | issuer prefixes |
| ip address | parses with the standard library | |

Validators tolerate a trailing comma or full stop, so a sloppy span boundary
is not mistaken for a wrong detection.

**Labels are matched by normalized form plus aliases.** The models are
zero-shot, so callers write `phone_number`, `Phone Number` or
`telephone number`. Labels are case-folded with separators collapsed, then
looked up in an alias table, so all three reach the phone validator. A label
with no validator is never filtered.

**Validators run after the merge**, on final spans. Before the merge they
would see window fragments: `4111 1111` fails Luhn, would be dropped, and
could no longer mark the full number's rival as a fragment or be unioned into
it. A test covers a card cut by a window boundary with validators on.

**Off by default, and why.** A validator can only remove a detection. A value
that fails its check is often still sensitive (one mistyped digit breaks
Luhn), and real data does not follow one country's formats. The benchmark
shows both halves of that. With validators on, GLiNER2-PII's overall recall
falls from 0.929 to 0.910, nearly all of it on card numbers (0.86 to 0.56),
because many of the dataset's synthetic cards fail Luhn. And the first `ssn`
validator, which required the US 3-2-4 grouping, cut `ssn` recall from 0.92
to 0.53: 68 of 162 gold SSNs were 3-3-3 groupings or National Insurance
numbers. It was widened, and `ssn` recall with validators is now unchanged
at 0.92 with precision at 0.97. The lesson generalizes: a validator runs on
spans the model has already typed, so it should reject what cannot be that
type, not enforce one layout. For a redactor a missed span is a leak and an
extra span is over-redaction, so dropping detections has to be something the operator chooses.

**Switches.** `validators` in settings is `off`, `on`/`all`, or a list of
labels (`PII_VALIDATORS`, `--validators`), resolved against the pack at
startup so a typo is a configuration error. A request can override with
`"validate": true|false`. `/healthz` reports what is in force.

**Alternatives.**
- *Relabel instead of drop* (a failed `ssn` becomes a generic `identifier`).
  Keeps coverage, but invents a label the caller did not ask for. Worth
  revisiting; see NEXT_STEPS.
- *Validators as extra detectors* (regex finds emails the model missed). A
  different feature (a recall backstop), also in NEXT_STEPS.
- *Packs loaded from JSON.* Label sets could be, validators cannot; `labels`
  in settings already covers custom label sets.

## 8. Service, configuration, CLI

- **One pipeline per app.** `create_app(settings, pipeline)` stores it on
  `app.state`. Tests build many apps, each with its own fake backend, and one
  test asserts they do not share state.
- **Sync endpoints.** Handlers are plain `def`, so FastAPI runs them in its
  thread pool and the event loop stays free during a forward pass.
- **Model residency.** With `PII_PRELOAD=1` (set in the image) the weights are
  loaded in the lifespan hook before the server accepts requests and released
  at shutdown. `/healthz` therefore doubles as readiness, and the compose
  healthcheck has a `start_period` to cover the first-run download. Off by
  default so the library and CLI stay lazy.
- **The hash salt is server-side only.** It is configuration, not a request
  field: a secret does not belong in request bodies and logs.
- **Errors.** Bad input is 422, text over `max_text_chars` is 413, and an
  offset verification failure is 500 with no spans in the body.
- **`src/main.py` is a shim.** The spec names `src/main.py` as the app module.
  The routes and models (including the `labels` example
  `["person", "email", "phone number", "ssn"]`) live in `pii_extract/api.py`
  so they ship inside the installed package; `src/main.py` instantiates the
  app, and Uvicorn is started with `--app-dir src`.
- **Configuration.** One `Settings` model. `load_settings` merges four layers
  in order (defaults, JSON file, environment, CLI) into a dict and validates
  once, so every layer gets the same type checks and unknown keys in the file
  are an error rather than a silent typo. A CLI value of `None` means "not
  given" and falls through. `GLINER_MODEL` keeps the name the spec uses;
  everything else is `PII_*`.
- **CLI.** `argparse`, no extra dependency. Input from an argument, `--file`,
  or stdin; JSON out. The model is closed in a `finally`.

## 9. The default model's name

The spec names `fastino/gliner2-pii-v1`. No repository with that id exists on
the Hugging Face Hub (the API returns 401 and `from_pretrained` raises
`RepositoryNotFoundError`, with the local token present). The model card for
[`fastino/gliner2-privacy-filter-PII-multi`](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi)
calls the model "GLiNER2-PII", uses the string `fastino/gliner2-pii-v1` in its
own benchmark table, was created on 2026-05-10, has 307M parameters, and is
Apache-2.0. All of that matches the spec's description, so it is the same
model under its published id.

**Decision.** The default is the real repository id, and
`fastino/gliner2-pii-v1` is accepted as an alias, so the spec's
`ENV GLINER_MODEL=fastino/gliner2-pii-v1` works unchanged in the Dockerfile.
`/healthz` reports the resolved id. The README keeps the spec's provenance
line verbatim and adds one sentence explaining the name.

Note the model card says "205M parameters" in prose while the weights file has
307M; the spec's "about 0.3B" matches the weights.

## 10. Environment and packaging

- **uv with a lockfile.** `uv.lock` pins every package; `just setup` and the
  Docker build both use `--frozen`. Chosen over `venv` plus `requirements.txt`
  because one lock covers the extras and the dev group, and CI installs the
  torch-free subset from the same file.
- **CUDA 12.8 wheels.** `[tool.uv.sources]` points `torch` at the
  `cu128` index on Linux, since the RTX 5080 needs CUDA 12.8.
- **`protobuf` is an explicit dependency** of the `models` extra. `gliner`
  loads mdeberta's SentencePiece vocabulary, whose conversion to a fast
  tokenizer needs protobuf, and neither `gliner` nor `transformers` declares
  it; without it the urchade model fails to load.
- **Image.** `nvidia/cuda:12.8.1-base-ubuntu24.04`. The `base` flavour is
  enough because the torch wheels bundle their own CUDA libraries. The image
  runs as a non-root user and keeps weights in a named volume at `HF_HOME`.
  It is about 12 GB, almost all of it torch and the CUDA libraries.

## 11. Testing strategy

| Layer | Count | Needs | What it covers |
|---|---:|---|---|
| `tests/unit` | 270 | nothing | spans, windowing, merging, redaction, offsets end to end, schema packs and validators, adapters, sessions, config, API, CLI, benchmark scoring |
| `tests/model` | 28 | weights | both real models: detection, token alignment, window budget, offset round trip, an entity cut by a window boundary, concurrency, close and reload |

- **`FakeBackend`** is a regex "model" with a whitespace tokenizer and a
  configurable budget. It runs the real tokenize → window → shift → merge
  path with only the forward pass replaced. Its card and name patterns match
  *prefixes*, so it reports fragments at window cuts the way a real model
  does.
- **Stub models** in the adapter tests answer in each library's exact result
  shape, so span decoding is tested without weights.
- **Model tests** carry the `model` marker (added automatically to everything
  under `tests/model`) and skip themselves when torch or the weights are
  missing. CI runs `-m "not model"`. Verified in a clean environment without
  torch (at the end of the first session: 165 passed, 19 skipped), and the
  same workflow now passes on GitHub Actions.
- Model tests assert **pipeline properties**, and assert exact model output
  only where the model is reliable. For example, urchade tags the local part
  of `agent10@example.com` as a person in some contexts; that is model
  quality, so the long-document test asserts emails for the default model only
  and phones for both.

## 12. ONNX backend (urchade model)

**Export.** `scripts/export_onnx.py` (`just export-onnx`) calls gliner's own
`export_to_onnx` (fp32, opset 19, no quantization) and writes a
self-contained directory under `models/`: config, tokenizer and a 1.16 GB
`model.onnx`. `models/` and `*.onnx` are gitignored; nothing exported is
committed. Only the urchade model is exported: `gliner2` 2.1.0 has no export
path.

**Backend.** `OnnxBackend` (`PII_BACKEND=onnx`, model directory in
`GLINER_MODEL`, default `models/gliner_multi_pii-v1-onnx`) subclasses
`GlinerBackend` and overrides only loading. The word splitter, window budget,
inference call and span decoder are the same code on both runtimes, which is
what makes parity a meaningful check and keeps the backend small.

**The execution provider is explicit configuration:** `PII_ONNX_PROVIDER`
(`--onnx-provider`), `cpu` or `cuda`, default `cpu`. The `device` setting does
not affect it. An earlier draft tried CUDA first and fell back to CPU. That
is the wrong design: the failure this project actually met was a segmentation
fault during session creation, and a segfault cannot be caught in-process, so
the "fallback" would never run and the service would die at startup. CUDA is
used only when asked for, and asking for it without getting it raises.

**Out-of-process probe.** With `provider=cuda`, before this process imports
anything that touches CUDA, `backends/onnx_probe.py` starts a child process
that creates a CUDA session for an embedded one-node model and runs it. The
parent reads the exit status:

| Child outcome | Result |
|---|---|
| exit 0 | proceed to load the real model on CUDA |
| killed by a signal | `RuntimeError` naming the signal (for example SIGSEGV) and pointing at the host's driver or CUDA libraries |
| build has no CUDA provider, or the session dropped it | `RuntimeError` with the child's message |
| wrong result, timeout (120 s), or cannot start | `RuntimeError` saying which |

The probe uses a tiny model on purpose: it exercises provider loading, the
CUDA runtime, cuDNN and the driver in about a second, without loading 1.2 GB.
It costs one extra process start per model load and nothing per request.
After the real session is created the backend also checks the providers the
session reports, because ONNX Runtime drops a provider it cannot initialise
without raising. Unit tests drive the probe with children that segfault,
exit non-zero, hang, and succeed.

**Parity.** `tests/model/test_onnx_parity.py` (`just onnx-parity`, provider
from `PII_ONNX_PROVIDER`) runs both runtimes through the same adapter and
`Pipeline`, so the comparison is made after span decoding, windowing and
merging. Fixtures: four short texts (including BOM, NBSP, CRLF, and one with
no PII) and a long text split into 8 windows. Reference: PyTorch on CUDA.

| | ONNX, CPU provider | ONNX, CUDA provider |
|---|---|---|
| Tokenization and window budget | identical | identical |
| Span mismatches (146 window-level spans, 120 merged) | 0 | 0 |
| Max score delta | 1.04e-04 | 4.41e-03 |
| Median score delta | not recorded | 1.6e-04 |
| Spans differing by more than 1e-03 | 0 | 26 |
| Tolerance asserted by the test | 1e-3 | 1e-2 |

The CUDA provider's scores differ from PyTorch's more than the CPU
provider's do. The 26 larger differences are all `phone number` spans with
mid-range scores (0.67 to 0.82), for example `555-233-1033` at 0.6753 against
0.6797; no span was gained or lost. Batch size is not the cause (batch of 1
gives the same picture), and PyTorch against itself across batch sizes
differs by at most 6.8e-05. The cause inside the CUDA provider was not
investigated. The practical consequence: a span whose score sits within
about 0.005 of the threshold could be kept by one runtime and dropped by the
other. None did on these fixtures. The test's CUDA tolerance is set to 1e-2
to describe this observed behaviour, not because 1e-2 was the goal.

For reference, ONNX on the CPU provider against PyTorch on CPU (three short
fixtures) agreed to 4.23e-06, so the export itself is faithful.

**Latency on the RTX 5080**, model only (`backend.predict`), mean and p95
over 40 to 150 timed calls after warm-up, with `torch.cuda.synchronize()`
around each call:

| | PyTorch, CUDA | ONNX Runtime, CUDA |
|---|---:|---:|
| Batch 1, short text | 11.5 ms / 11.5 ms | 7.6 ms / 8.1 ms |
| Batch 1, one 384-token window | 22.8 ms / 23.8 ms | 16.1 ms / 16.7 ms |
| Batch 8, eight 384-token windows | 85.7 ms / 85.1 ms | 76.8 ms / 76.4 ms |

On CPU, batch 1 on short fixtures: 51.4 ms / 59.5 ms for PyTorch against
24.3 ms / 30.6 ms for ONNX Runtime. These are ad hoc measurements on
synthetic fixtures, not part of `bench/`. ONNX Runtime is about a third
faster at batch 1 on the GPU and about 10% faster at batch 8.

**The host problem that shaped this design, now fixed.** The first attempts
to use the CUDA provider segfaulted (exit 139). It was a host fault, not an
ONNX Runtime or project bug:

- The crash reproduced on a one-node model, without torch in the process, and
  in a clean environment with matched CUDA 13 and cuDNN libraries, which
  ruled out a conflict with torch's bundled CUDA 12.8 and a CUDA/cuDNN
  mismatch.
- `LD_DEBUG=libs` showed the last library loaded before the crash:
  `/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1`, version 580.178.04,
  from the Ubuntu package `libnvidia-compute-580`. That is a native Linux
  driver package. Under WSL2 the driver is the Windows one (610.88), and its
  own copy of that library sits in the driver store
  (`/usr/lib/wsl/drivers/nv_dispig.inf_amd64_*/`), which is not on the
  default library path, so the loader found the stale 580 copy and the
  610.88 `libcuda` crashed calling into it.
- PyTorch was unaffected because its cu128 wheels ship kernels precompiled
  for the RTX 5080, so the driver never needed that library. That ONNX
  Runtime needs it because its wheel lacks such kernels is likely but was not
  verified.
- **Fix: the native packages (`libnvidia-compute-535`, `libnvidia-compute-580`
  and Ubuntu's CUDA 12.0 toolkit) were removed from the WSL distribution.**
  With no code change, no preload and no library-path change, the CUDA
  provider now loads and runs, and the results above were measured that way.
  No workaround was ever added to the backend: papering over a broken driver
  install in application code would have hidden it from every other CUDA
  program on the machine.

**Packaging facts.** `onnxruntime-gpu` 1.32 is a CUDA 13 build (it links
`libcublas.so.13`). It includes the CPU provider, so the `onnx` extra uses it
for both. For the CUDA provider it needs a CUDA 13 runtime and cuDNN 9 from
the host; on the development machine those come from a system CUDA 13.2
toolkit on `LD_LIBRARY_PATH`. It cannot use the CUDA 12.8 libraries torch
bundles. The Docker image installs only the `models` extra, so it has no ONNX
backend yet; its CUDA 12.8 base could not serve the CUDA provider anyway.
`onnxruntime` and `onnxruntime-gpu` install into the same directory, and so
do the cu12 and cu13 NVIDIA packages; switching between them in one
environment can leave files missing, and the cure is a reinstall from the
lockfile.

## Benchmark methodology

`bench/` scores both backends through `Pipeline.extract`, the same path the
API uses, so latency includes tokenizing, windowing, and merging.

- **Data.** A seeded sample of the `gretelai/gretel-pii-masking-en-v1` test
  split: Apache-2.0, synthetic, English. AI4Privacy, the spec's other
  suggestion, was not evaluated in this session.
- **Gold offsets are reconstructed.** The dataset lists entity *strings* and
  types, not offsets. Each string is located in the text at whole-token
  occurrences (so the first name `Al` does not match inside `Also`).
  Annotations that cannot be found are counted and reported.
- **Two normalizations** make gold comparable with span predictors: a gold
  span nested inside a longer gold span of the same label is dropped, and
  same-label gold spans separated by a single space are merged, so a first
  name followed by a last name is one `person`, as the models report it.
- **Labels.** Seven labels, with the same natural-language prompt for both
  models. Address-like types are out of scope.
- **Scoring.** Exact match on `(start, end, label)` per label, micro-averaged,
  plus an overlap-match micro average that forgives boundary conventions.
- **Latency.** One request at a time after five warm-up calls, with
  `torch.cuda.synchronize()` around each timed call.
- **`--concat N`** joins N documents into one text and shifts the gold
  offsets, because every dataset document fits in a single window and the
  windowing path would otherwise go unmeasured.

- **Ablation.** `--ablation` runs four variants per backend on one loaded
  model: baseline, validators on, an extra "medical record number" label, and
  both. The label table is built by `label_table()`, so an extra label adds
  both a prompt and the matching gold spans. The summary scores `ssn` and the
  micro average on the base seven labels in every variant, so rows are
  comparable; predictions under the extra label are scored separately.
  Result: the extra label lifts GLiNER2-PII's `ssn` precision from 0.44 to
  0.74 and its overall F1 from 0.770 to 0.819 with recall at 0.92; validators
  on their own (rerun after the `ssn` validator was widened; the ablation
  file's validator rows predate that) lift GLiNER2-PII's overall F1 to 0.836
  with recall at 0.91, and leave urchade's flat (0.823 to 0.821).

- **Held-out sample.** `--offset 500` skips the first 500 documents of the
  shuffled split. The `ssn` validator was widened after inspecting its
  rejections on documents 0-499, so validator figures on those are tuned;
  the headline validator figures come from documents 500-999. There,
  GLiNER2-PII's `ssn` goes from 0.482 / 0.849 / 0.615 to 0.969 / 0.849 / 0.905
  and the micro average from 0.677 / 0.915 / 0.778 to 0.777 / 0.892 / 0.831.

Limits of the result: one dataset, one seed, 500 documents, one threshold,
untuned prompts, and reconstructed gold. The dataset contains many identifier
types outside the seven scored labels, which counts against a model that tags
them with the nearest available label (this accounts for most of GLiNER2-PII's
`ssn` false positives). It is a comparison of two models under one setup, not
a general ranking.

## Deviations from the spec, in one place

| Spec | What was done | Why |
|---|---|---|
| Default model `fastino/gliner2-pii-v1` | Default is `fastino/gliner2-privacy-filter-PII-multi`; the spec's name is an alias | The spec's id does not exist on the Hub (section 9) |
| App in `src/main.py` | `src/main.py` is a shim over `pii_extract/api.py` | So the API ships in the installed package (section 8) |
| "Existing scaffold" carries over | Scaffold created from scratch to the spec's description | The repository held only the spec and `.gitignore` |
| Schema packs, validators, Luhn | Built (section 7); validators are off by default | A validator can only remove detections, which is the wrong default for a redactor |
| ONNX export | Export script, `OnnxBackend` with CPU and CUDA providers, and a parity test for the urchade model (section 12) | GLiNER2 has no export path in `gliner2` 2.1.0 |
