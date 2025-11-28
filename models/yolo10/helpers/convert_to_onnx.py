#!/usr/bin/env python
"""
Convert a YOLOv10 PyTorch checkpoint to ONNX.

Usage:
    pixi run python models/yolo10/helpers/convert_to_onnx.py yolov10n
    pixi run python models/yolo10/helpers/convert_to_onnx.py yolov10n.pt

This will read models/yolo10/checkpoints/<model-name>.pt and write
models/yolo10/onnx/<model-name>.onnx.
"""

import sys
from pathlib import Path
import os
import torch


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(
            "Usage: convert_to_onnx.py <model-name | model-name.pt>\n"
            "Example: convert_to_onnx.py yolov10n",
            file=sys.stderr,
        )
        return 1

    # Work around PyTorch 2.6+ default weights_only=True when loading YOLOv10
    # checkpoints, which rely on custom classes stored in the checkpoint.
    original_torch_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load  # type: ignore[assignment]

    model_arg = argv[1]
    model_stem = model_arg[:-3] if model_arg.endswith(".pt") else model_arg

    helpers_dir = Path(__file__).resolve().parent
    yolo_root = helpers_dir.parent
    checkpoints_dir = yolo_root / "checkpoints"
    onnx_dir = yolo_root / "onnx"
    src_dir = yolo_root / "src"

    ckpt_path = checkpoints_dir / f"{model_stem}.pt"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / f"{model_stem}.onnx"

    if not ckpt_path.is_file():
        print(
            f"Error: checkpoint not found at {ckpt_path}. "
            "Run models/yolo10/bootstrap.sh first or check the model name.",
            file=sys.stderr,
        )
        return 1

    # Prefer local cloned YOLOv10 ultralytics source if available.
    if src_dir.is_dir():
        sys.path.insert(0, str(src_dir))

    try:
        from ultralytics import YOLOv10  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - environment dependent
        print(
            "Error: could not import 'YOLOv10' from 'ultralytics'. "
            "Ensure the YOLOv10 repo under models/yolo10/src is present "
            "or that a compatible package is installed.\n"
            f"Details: {exc}",
            file=sys.stderr,
        )
        return 1

    print(f"[YOLO10] Loading checkpoint from {ckpt_path}...")
    model = YOLOv10(str(ckpt_path))

    print("[YOLO10] Exporting ONNX model...")
    try:
        exported_path_str = model.export(format="onnx")
    except Exception as exc:
        print(f"Error during ONNX export: {exc}", file=sys.stderr)
        return 1

    exported_path = Path(exported_path_str).resolve()
    if not exported_path.is_file():
        print(
            f"Error: YOLOv10 export reported '{exported_path}' "
            "but the file does not exist.",
            file=sys.stderr,
        )
        return 1

    if exported_path != onnx_path:
        try:
            exported_path.replace(onnx_path)
        except OSError:
            onnx_path.write_bytes(exported_path.read_bytes())

    print(f"[YOLO10] ONNX model written to {onnx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
