"""Export the urchade GLiNER model to ONNX (fp32, no quantization).

    just export-onnx
    uv run python scripts/export_onnx.py --model MODEL_ID --out models/some-dir

Writes a self-contained directory (config, tokenizer, model.onnx) that
``GLiNER.from_pretrained(out, load_onnx_model=True, load_tokenizer=True)`` can
load. ``models/`` and ``*.onnx`` are gitignored: exported weights are build
artifacts and are never committed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_MODEL = "urchade/gliner_multi_pii-v1"
DEFAULT_OUT = Path("models") / "gliner_multi_pii-v1-onnx"


def export(model_id: str, out_dir: Path, opset: int = 19) -> Path:
    from gliner import GLiNER

    model = GLiNER.from_pretrained(model_id)
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Config and tokenizer first, so the directory loads on its own.
    model.save_pretrained(str(out_dir))
    paths = model.export_to_onnx(
        str(out_dir), onnx_filename="model.onnx", quantize=False, opset=opset
    )
    # The torch weights are not needed next to the ONNX graph.
    for leftover in ("pytorch_model.bin", "model.safetensors"):
        (out_dir / leftover).unlink(missing_ok=True)
    return Path(paths["onnx_path"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--opset", type=int, default=19)
    args = parser.parse_args()
    path = export(args.model, args.out, args.opset)
    print(f"exported {args.model} -> {path} ({path.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
