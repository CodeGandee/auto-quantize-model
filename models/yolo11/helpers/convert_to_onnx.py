#!/usr/bin/env python
"""
Convert a YOLO11 PyTorch checkpoint to ONNX.

Usage:
    python models/yolo11/helpers/convert_to_onnx.py yolo11n
    python models/yolo11/helpers/convert_to_onnx.py yolo11n.pt

This will read models/yolo11/checkpoints/<model-name>.pt and write
models/yolo11/onnx/<model-name>.onnx.
"""

import os
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(
            "Usage: convert_to_onnx.py <model-name | model-name.pt>\n"
            "Example: convert_to_onnx.py yolo11n",
            file=sys.stderr,
        )
        return 1

    model_arg = argv[1]
    model_stem = model_arg[:-3] if model_arg.endswith(".pt") else model_arg

    # Resolve important paths relative to this file
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
            "Run models/yolo11/bootstrap.sh first or check the model name.",
            file=sys.stderr,
        )
        return 1

    # Prefer local cloned ultralytics source if available.
    if src_dir.is_dir():
        sys.path.insert(0, str(src_dir))

    try:
        from ultralytics import YOLO  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - import environment dependent
        print(
            "Error: could not import 'ultralytics'. "
            "Ensure the Ultralytics YOLO package is installed or that "
            "models/yolo11/src is a valid clone of the ultralytics repo.\n"
            f"Details: {exc}",
            file=sys.stderr,
        )
        return 1

    print(f"[YOLO11] Loading checkpoint from {ckpt_path}...")
    model = YOLO(str(ckpt_path))

    # Change working directory so the exported ONNX lands in the desired folder
    # as <model-stem>.onnx, then verify and rename/move if needed.
    prev_cwd = Path.cwd()
    os.chdir(onnx_dir)
    try:
        print(f"[YOLO11] Exporting ONNX model to {onnx_path}...")
        # This creates '<basename>.onnx' in the current working directory.
        model.export(format="onnx")
    finally:
        os.chdir(prev_cwd)

    if not onnx_path.is_file():
        print(
            f"Warning: expected ONNX file not found at {onnx_path}. "
            "Check Ultralytics export behavior.",
            file=sys.stderr,
        )
        return 1

    print(f"[YOLO11] ONNX model written to {onnx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

