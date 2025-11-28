#!/usr/bin/env python
"""
Run YOLOv10 inference on an image and save an annotated version using supervision.

Usage:
    pixi run python models/yolo10/helpers/infer_and_annotate.py yolov10s tmp/yolo10-infer/bus.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import supervision as sv
import cv2


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(
            "Usage: infer_and_annotate.py <model-name | model-name.pt> <image-path>\n"
            "Example: infer_and_annotate.py yolov10s tmp/yolo10-infer/bus.jpg",
            file=sys.stderr,
        )
        return 1

    model_arg = argv[1]
    model_stem = model_arg[:-3] if model_arg.endswith(".pt") else model_arg
    image_path = Path(argv[2]).expanduser().resolve()

    if not image_path.is_file():
        print(f"Error: image not found at {image_path}", file=sys.stderr)
        return 1

    # Allow YOLOv10 checkpoint loading under torch>=2.6 (weights_only default).
    original_torch_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load  # type: ignore[assignment]

    # Prefer local YOLOv10 repo's ultralytics over PyPI ultralytics.
    helpers_dir = Path(__file__).resolve().parent
    yolo_root = helpers_dir.parent
    src_dir = yolo_root / "src"
    ckpt_path = (yolo_root / "checkpoints" / f"{model_stem}.pt").resolve()

    if not ckpt_path.is_file():
        print(
            f"Error: checkpoint not found at {ckpt_path}. "
            "Run models/yolo10/bootstrap.sh first or check the model name.",
            file=sys.stderr,
        )
        return 1

    if src_dir.is_dir():
        sys.path.insert(0, str(src_dir))

    from ultralytics import YOLOv10  # type: ignore[import-not-found]

    print(f"[YOLO10] Loading checkpoint from {ckpt_path}...")
    model = YOLOv10(str(ckpt_path))

    print(f"[YOLO10] Running inference on {image_path}...")
    results = model(str(image_path))
    result = results[0]

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: failed to read image at {image_path}", file=sys.stderr)
        return 1
    detections = sv.Detections.from_ultralytics(result)

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    labels = [
        f"{model.names[class_id]} {confidence:.2f}"
        for _, _, confidence, class_id, *_ in detections
    ]

    annotated = box_annotator.annotate(scene=image.copy(), detections=detections)
    annotated = label_annotator.annotate(
        scene=annotated, detections=detections, labels=labels
    )

    out_path = image_path.with_name(f"{image_path.stem}_{model_stem}_annotated{image_path.suffix}")
    cv2.imwrite(str(out_path), annotated)
    print(f"[YOLO10] Annotated image written to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
