"""Benchmark both backends on a sample of a public synthetic-PII dataset.

    just bench                                    # 500 documents, both backends
    just bench --limit 100 --backends gliner2
    just bench --concat 25                        # join documents to force windowing
    just bench --validators on                    # run the schema pack's validators
    just bench --extra-labels "medical record number"
    just bench --ablation                         # the four combinations, side by side

Prints a Markdown report and writes it (plus raw JSON) to bench/results/.
Every document goes through ``Pipeline.extract``, the same code path the API
and CLI use, so latency includes tokenizing, windowing and dedup.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from bench.gretel import (
    BASE_LABELS,
    DATASET_ID,
    EXTRA_LABELS,
    LabelTable,
    gold_spans,
    label_table,
    parse_entities,
    prompt_to_label,
    prompts,
)
from bench.metrics import Counts, micro_average, percentile, score_exact, score_overlap
from pii_extract.backends import BACKENDS, Backend, create_backend
from pii_extract.pipeline import Pipeline
from pii_extract.schema import PII_EXTENDED

RESULTS_DIR = Path(__file__).parent / "results"
SEPARATOR = "\n\n"
MRN = "medical record number"

# A text plus the dataset documents it was built from: (offset, text, annotations).
# Gold spans are located inside each source document, never across them.
RawDocument = tuple[str, list[tuple[int, str, list]]]


@dataclass(frozen=True)
class Variant:
    """One benchmark configuration."""

    validators: bool = False
    extra_labels: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        parts = ["validators on" if self.validators else "baseline"]
        if self.extra_labels:
            parts.append("+ " + ", ".join(self.extra_labels))
        return " ".join(parts) if self.validators or not self.extra_labels else parts[1]

    @property
    def slug(self) -> str:
        parts = [f"plus-{label.replace(' ', '-')}" for label in self.extra_labels]
        if self.validators:
            parts.append("validators")
        return "-".join(parts)


ABLATION = (
    Variant(),
    Variant(validators=True),
    Variant(extra_labels=(MRN,)),
    Variant(validators=True, extra_labels=(MRN,)),
)


def load_sample(limit: int, seed: int, concat: int = 1, offset: int = 0) -> list[RawDocument]:
    """A seeded sample of the test split.

    With ``concat > 1``, consecutive documents are joined into one long text
    (each keeps its own annotations and offset). The dataset's documents all fit in a single
    window, so this is what makes the windowing and dedup stages do work.
    """
    from datasets import load_dataset

    rows = load_dataset(DATASET_ID, split="test")
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    documents = [
        (rows[i]["text"], parse_entities(rows[i]["entities"]))
        for i in order[offset : offset + limit]
    ]
    step = max(concat, 1)
    sample: list[RawDocument] = []
    for first in range(0, len(documents), step):
        parts: list[tuple[int, str, list]] = []
        offset = 0
        for text, entities in documents[first : first + step]:
            parts.append((offset, text, entities))
            offset += len(text) + len(SEPARATOR)
        sample.append((SEPARATOR.join(text for _, text, _ in parts), parts))
    return sample


def hardware() -> str:
    import torch

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        return f"{gpu} (CUDA {torch.version.cuda}, torch {torch.__version__})"
    return f"CPU: {platform.processor() or platform.machine()} (torch {torch.__version__})"


def evaluate(
    backend: Backend, sample: list[RawDocument], variant: Variant, threshold: float
) -> dict:
    """Score one loaded backend under one variant."""
    import torch

    table = label_table(variant.extra_labels)
    to_label = prompt_to_label(table)
    pipeline = Pipeline(
        backend,
        labels=prompts(table),
        schema=PII_EXTENDED,
        validators=["all"] if variant.validators else [],
        threshold=threshold,
    )
    for text, _ in sample[:5]:  # warm-up: CUDA kernels, tokenizer caches
        pipeline.extract(text)

    exact: dict[str, Counts] = {label: Counts() for label in table}
    relaxed: dict[str, Counts] = {label: Counts() for label in table}
    latencies_ms: list[float] = []
    windows = 0
    unlocated = 0
    on_gpu = backend.device.startswith("cuda")
    for text, parts in sample:
        gold = []
        for offset, part_text, entities in parts:
            part_gold, missing = gold_spans(part_text, entities, table)
            gold.extend((start + offset, end + offset, label) for start, end, label in part_gold)
            unlocated += missing
        windows += len(pipeline.plan(text))

        if on_gpu:
            torch.cuda.synchronize()
        tick = time.perf_counter()
        spans = pipeline.extract(text)
        if on_gpu:
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - tick) * 1000)

        predicted = [(s.start, s.end, to_label[s.label]) for s in spans]
        for label, counts in score_exact(gold, predicted).items():
            exact[label].add(counts)
        for label, counts in score_overlap(gold, predicted).items():
            relaxed[label].add(counts)

    return {
        "backend": backend.name,
        "model": backend.model_id,
        "device": backend.device,
        "variant": variant.name,
        "validators": variant.validators,
        "extra_labels": list(variant.extra_labels),
        "documents": len(sample),
        "windows": windows,
        "unlocated_gold_annotations": unlocated,
        "latency_ms": {
            "mean": round(statistics.fmean(latencies_ms), 2),
            "p50": round(percentile(latencies_ms, 50), 2),
            "p95": round(percentile(latencies_ms, 95), 2),
            "max": round(max(latencies_ms), 2),
        },
        "exact": {label: vars(c) for label, c in exact.items()},
        "overlap": {label: vars(c) for label, c in relaxed.items()},
    }


def _counts(result: dict, kind: str, labels) -> Counts:
    return micro_average({k: Counts(**v) for k, v in result[kind].items() if k in labels})


def _prf(counts: Counts) -> str:
    return f"{counts.precision:.3f} / {counts.recall:.3f} / {counts.f1:.3f}"


def describe(args: argparse.Namespace, sample: list[RawDocument], hw: str) -> str:
    chars = [len(text) for text, _ in sample]
    shape = (
        f"{len(sample)} documents"
        if args.concat <= 1
        else f"{args.limit} documents joined {args.concat} at a time into {len(sample)} long texts"
    )
    return (
        f"Dataset: `{DATASET_ID}` test split, {shape} "
        f"(seed {args.seed}, offset {args.offset}; "
        f"median {int(statistics.median(chars))} characters, "
        f"max {max(chars)}). Threshold {args.threshold}. Hardware: {hw}."
    )


def render_variant(results: list[dict], table: LabelTable) -> str:
    """Per-label table and latency for one variant, one column per backend."""
    lines = [
        "Exact span match (start, end and label must all agree), precision / recall / F1:",
        "",
        "| Label | Support | " + " | ".join(f"`{r['model']}`" for r in results) + " |",
        "|---|---:|" + "---|" * len(results),
    ]
    for label in table:
        support = Counts(**results[0]["exact"][label]).support
        cells = [_prf(Counts(**r["exact"][label])) for r in results]
        lines.append(f"| {label} | {support} | " + " | ".join(cells) + " |")
    micro = [_counts(r, "exact", table) for r in results]
    lines.append(
        f"| **micro average** | {micro[0].support} | "
        + " | ".join(f"**{_prf(m)}**" for m in micro)
        + " |"
    )
    if len(table) > len(BASE_LABELS):
        base = [_counts(r, "exact", BASE_LABELS) for r in results]
        lines.append(
            f"| micro average, base {len(BASE_LABELS)} labels | {base[0].support} | "
            + " | ".join(_prf(m) for m in base)
            + " |"
        )
    relaxed = [_counts(r, "overlap", table) for r in results]
    lines.append(
        "| micro average, overlap match | | " + " | ".join(_prf(m) for m in relaxed) + " |"
    )
    lines += [
        "",
        "Latency per text through `Pipeline.extract` (one request at a time, after warm-up):",
        "",
        "| Model | Device | Mean | p50 | p95 | Max | Windows/text |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        ms = r["latency_ms"]
        per_doc = r["windows"] / r["documents"]
        lines.append(
            f"| `{r['model']}` | {r['device']} | {ms['mean']:.1f} ms | {ms['p50']:.1f} ms "
            f"| {ms['p95']:.1f} ms | {ms['max']:.1f} ms | {per_doc:.2f} |"
        )
    return "\n".join(lines) + "\n"


def render_summary(results: list[dict]) -> str:
    """One row per backend and variant: the ssn label and the base-label micro average."""
    lines = [
        f"`ssn` and the micro average over the base {len(BASE_LABELS)} labels "
        "(so every row is scored on the same gold spans), precision / recall / F1:",
        "",
        "| Model | Variant | ssn | micro average |",
        "|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| `{r['model']}` | {r['variant']} | {_prf(Counts(**r['exact']['ssn']))} "
            f"| {_prf(_counts(r, 'exact', BASE_LABELS))} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--limit", type=int, default=500, help="documents to evaluate")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed")
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="skip this many documents of the shuffled split first; --offset 500 gives a "
        "sample disjoint from the default one (use it as a held-out set)",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backends", nargs="+", default=list(BACKENDS), choices=list(BACKENDS))
    parser.add_argument(
        "--concat", type=int, default=1, help="join this many documents per text (forces windows)"
    )
    parser.add_argument(
        "--validators",
        choices=["off", "on"],
        default="off",
        help="run the schema pack's validators and drop entities that fail",
    )
    parser.add_argument(
        "--extra-labels",
        nargs="+",
        default=[],
        choices=sorted(EXTRA_LABELS),
        metavar="LABEL",
        help=f"labels to offer the models beyond the base set: {sorted(EXTRA_LABELS)}",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="run baseline, validators on, + medical record number, and both; print a summary",
    )
    parser.add_argument("--no-save", action="store_true", help="print only")
    args = parser.parse_args()

    if args.ablation:
        variants = ABLATION
    else:
        variants = (
            Variant(validators=args.validators == "on", extra_labels=tuple(args.extra_labels)),
        )

    sample = load_sample(args.limit, args.seed, args.concat, args.offset)
    hw = hardware()
    results: list[dict] = []
    for name in args.backends:  # load each model once, run every variant on it
        backend = create_backend(name, device=args.device)
        backend.load()
        results.extend(evaluate(backend, sample, variant, args.threshold) for variant in variants)
        backend.close()

    sections = [describe(args, sample, hw), ""]
    if args.ablation:
        sections += [render_summary(results)]
    for variant in variants:
        chosen = [r for r in results if r["variant"] == variant.name]
        if args.ablation:
            sections.append(f"### {variant.name}\n")
        sections.append(render_variant(chosen, label_table(variant.extra_labels)))
    report = "\n".join(sections)
    print(report)

    if not args.no_save:
        RESULTS_DIR.mkdir(exist_ok=True)
        parts = ["gretel"]
        if args.offset:
            parts.append(f"heldout{args.offset}")
        if args.concat > 1:
            parts.append(f"concat{args.concat}")
        parts.append("ablation" if args.ablation else variants[0].slug)
        stem = "-".join(part for part in parts if part)
        (RESULTS_DIR / f"{stem}.md").write_text(report, encoding="utf-8")
        (RESULTS_DIR / f"{stem}.json").write_text(
            json.dumps(
                {"dataset": DATASET_ID, "args": vars(args), "hardware": hw, "results": results},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
