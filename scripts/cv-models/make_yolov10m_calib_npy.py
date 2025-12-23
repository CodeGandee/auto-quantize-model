#!/usr/bin/env python
"""Build a YOLOv10m-style calibration tensor from an image list.

Usage:
    pixi run python scripts/cv-models/make_yolov10m_calib_npy.py \\
        --list datasets/quantize-calib/quant100.txt \\
        --out tmp/yolov10m_lowbit/<run-id>/calib/calib_yolov10m_640.npy

This reads image paths (one per line), applies YOLO-style preprocessing
(letterbox to 640x640, BGR→RGB, [0,1] scaling), and saves a float32 tensor
with shape [N, 3, 640, 640] to a `.npy` file suitable for ModelOpt ONNX PTQ.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

from auto_quantize_model.cv_models.yolo_preprocess import find_repo_root, preprocess_image_path, read_image_list


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build YOLOv10m calibration tensor from an image list.")
    parser.add_argument(
        "--list",
        type=Path,
        required=True,
        help="Path to a text file listing image paths (one per line).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output .npy path for the calibration tensor.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Target square image size for letterbox preprocessing (default: 640).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap on the number of images to load from the list.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root for resolving relative paths (auto-detected when omitted).",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)

    repo_root = args.repo_root or find_repo_root(Path(__file__))
    image_paths = read_image_list(Path(args.list), repo_root=repo_root)
    if args.max_images is not None:
        image_paths = image_paths[: int(args.max_images)]

    print(f"Loaded {len(image_paths)} image paths from {args.list}")
    print(f"Building calibration tensor at size {args.imgsz}x{args.imgsz} (float32)...")

    tensors: list[np.ndarray] = []
    for idx, image_path in enumerate(image_paths, start=1):
        chw, _ = preprocess_image_path(image_path, img_size=int(args.imgsz), add_batch_dim=False)
        tensors.append(chw)
        if idx % 10 == 0 or idx == len(image_paths):
            print(f"  processed {idx}/{len(image_paths)}")

    calib = np.stack(tensors, axis=0).astype(np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, calib)
    print(f"Saved calibration tensor {calib.shape} to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

