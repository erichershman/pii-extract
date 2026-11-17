"""Benchmark both backends on a sample of a public synthetic-PII dataset.

    just bench                      # 500 documents, both backends
    just bench --limit 100 --backends gliner2

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
from pathlib import Path

from bench.gretel import DATASET_ID, LABELS, PROMPT_TO_LABEL, gold_spans, parse_entities
from bench.metrics import Counts, micro_average, percentile, score_exact, score_overlap
from pii_extract.backends import BACKENDS, create_backend
from pii_extract.pipeline import Pipeline

RESULTS_DIR = Path(__file__).parent / "results"


def load_sample(limit: int, seed: int) -> list[tuple[str, list]]:
    from datasets import load_dataset

    rows = load_dataset(DATASET_ID, split="test")
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    sample = []
    for index in order[:limit]:
        row = rows[index]
        sample.append((row["text"], parse_entities(row["entities"])))
    return sample


def hardware() -> str:
    import torch

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        return f"{gpu} (CUDA {torch.version.cuda}, torch {torch.__version__})"
    return f"CPU: {platform.processor() or platform.machine()} (torch {torch.__version__})"


def evaluate(backend_name: str, sample: list, threshold: float, device: str) -> dict:
    import torch

    backend = create_backend(backend_name, device=device)
    pipeline = Pipeline(backend, labels=[prompt for prompt, _ in LABELS.values()])
    pipeline.threshold = threshold

    started = time.perf_counter()
    backend.load()
    load_seconds = time.perf_counter() - started
    for text, _ in sample[:5]:  # warm-up: CUDA kernels, tokenizer caches
        pipeline.extract(text)

    exact: dict[str, Counts] = {label: Counts() for label in LABELS}
    relaxed: dict[str, Counts] = {label: Counts() for label in LABELS}
    latencies_ms: list[float] = []
    windows = 0
    unlocated = 0
    for text, entities in sample:
        gold, missing = gold_spans(text, entities)
        unlocated += missing
        windows += len(pipeline.plan(text))

        if backend.device.startswith("cuda"):
            torch.cuda.synchronize()
        tick = time.perf_counter()
        spans = pipeline.extract(text)
        if backend.device.startswith("cuda"):
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - tick) * 1000)

        predicted = [(s.start, s.end, PROMPT_TO_LABEL[s.label]) for s in spans]
        for label, counts in score_exact(gold, predicted).items():
            exact[label].add(counts)
        for label, counts in score_overlap(gold, predicted).items():
            relaxed[label].add(counts)

    device_name = backend.device
    model_id = backend.model_id
    backend.close()
    return {
        "backend": backend_name,
        "model": model_id,
        "device": device_name,
        "load_seconds": round(load_seconds, 2),
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


def _prf(counts: Counts) -> str:
    return f"{counts.precision:.3f} / {counts.recall:.3f} / {counts.f1:.3f}"


def render(results: list[dict], args: argparse.Namespace, sample: list, hw: str) -> str:
    chars = [len(text) for text, _ in sample]
    lines = [
        f"Dataset: `{DATASET_ID}` test split, {len(sample)} documents "
        f"(seed {args.seed}; median {int(statistics.median(chars))} characters, "
        f"max {max(chars)}). Threshold {args.threshold}. Hardware: {hw}.",
        "",
        "Exact span match (start, end and label must all agree), precision / recall / F1:",
        "",
        "| Label | Support | " + " | ".join(f"`{r['model']}`" for r in results) + " |",
        "|---|---:|" + "---|" * len(results),
    ]
    for label in LABELS:
        support = Counts(**results[0]["exact"][label]).support
        cells = [_prf(Counts(**r["exact"][label])) for r in results]
        lines.append(f"| {label} | {support} | " + " | ".join(cells) + " |")
    micro = [micro_average({k: Counts(**v) for k, v in r["exact"].items()}) for r in results]
    lines.append(
        f"| **micro average** | {micro[0].support} | "
        + " | ".join(f"**{_prf(m)}**" for m in micro)
        + " |"
    )
    relaxed = [micro_average({k: Counts(**v) for k, v in r["overlap"].items()}) for r in results]
    lines.append(
        "| micro average, overlap match | | " + " | ".join(_prf(m) for m in relaxed) + " |"
    )
    lines += [
        "",
        "Latency per document through `Pipeline.extract` (one request at a time, after warm-up):",
        "",
        "| Model | Device | Mean | p50 | p95 | Max | Windows/doc |",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--limit", type=int, default=500, help="documents to evaluate")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backends", nargs="+", default=list(BACKENDS), choices=list(BACKENDS))
    parser.add_argument("--no-save", action="store_true", help="print only")
    args = parser.parse_args()

    sample = load_sample(args.limit, args.seed)
    hw = hardware()
    results = [evaluate(name, sample, args.threshold, args.device) for name in args.backends]
    report = render(results, args, sample, hw)
    print(report)

    if not args.no_save:
        RESULTS_DIR.mkdir(exist_ok=True)
        (RESULTS_DIR / "gretel.md").write_text(report, encoding="utf-8")
        (RESULTS_DIR / "gretel.json").write_text(
            json.dumps(
                {"dataset": DATASET_ID, "args": vars(args), "hardware": hw, "results": results},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
