#!/usr/bin/env python
"""
Build a YOLO11-style calibration tensor from an image list.

Usage:
    pixi run python scripts/yolo11/make_yolo11_calib_npy.py \\
        --list datasets/quantize-calib/quant100.txt \\
        --out datasets/quantize-calib/calib_yolo11_640.npy

This reads image paths (one per line) from the list file, applies a simple
YOLO-style preprocessing pipeline (letterbox to 640x640, BGR→RGB, [0,1] scaling),
and saves a float32 tensor with shape [N, 3, 640, 640] to a `.npy` file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv :
        Optional list of argument strings; if None, `sys.argv` is used.

    Returns
    -------
    argparse.Namespace
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Build YOLO11 calibration tensor from an image list."
    )
    parser.add_argument(
        "--list",
        type=str,
        required=True,
        help="Path to a text file listing image paths (one per line).",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output .npy path for the calibration tensor.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Target square image size for letterbox preprocessing (default: 640).",
    )
    return parser.parse_args(argv)


def read_image_list(list_path: Path) -> List[Path]:
    """Read image paths from a text file.

    Parameters
    ----------
    list_path :
        Path to a text file containing one image path per line.

    Returns
    -------
    list of Path
        List of resolved image paths.

    Raises
    ------
    FileNotFoundError
        If the list file does not exist.
    ValueError
        If the list file is empty after filtering blank lines.
    """
    if not list_path.is_file():
        raise FileNotFoundError(f"Image list not found at {list_path}")
    image_paths: List[Path] = []
    with list_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            image_paths.append(Path(stripped).expanduser())
    if not image_paths:
        raise ValueError(f"No image paths found in {list_path}")
    return image_paths


def letterbox(
    image: np.ndarray,
    new_shape: Tuple[int, int] | int = 640,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> np.ndarray:
    """Resize and pad image to a square shape, preserving aspect ratio.

    This follows the standard YOLO-style letterbox approach.

    Parameters
    ----------
    image :
        Input image as an HWC uint8 array in RGB order.
    new_shape :
        Target shape as an integer (square) or (height, width) tuple.
    color :
        Padding color in RGB.

    Returns
    -------
    np.ndarray
        Letterboxed image with shape `(new_height, new_width, 3)`.
    """
    height, width = image.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    new_height, new_width = new_shape
    scale = min(new_width / width, new_height / height)

    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))

    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )

    pad_width = new_width - resized_width
    pad_height = new_height - resized_height

    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=color,
    )
    return padded


def build_calibration_tensor(
    image_paths: List[Path],
    img_size: int,
) -> np.ndarray:
    """Build a stacked calibration tensor from image paths.

    Parameters
    ----------
    image_paths :
        List of paths to input images.
    img_size :
        Target square image size for letterbox preprocessing.

    Returns
    -------
    np.ndarray
        Calibration tensor with shape `(N, 3, img_size, img_size)` and
        dtype `float32`, values scaled to `[0, 1]`.

    Raises
    ------
    FileNotFoundError
        If any image cannot be read from disk.
    """
    tensors: List[np.ndarray] = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Failed to read image at {image_path}")

        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Letterbox to target size
        image = letterbox(image, new_shape=img_size)

        # HWC -> CHW, scale to [0, 1]
        image_tensor = image.transpose(2, 0, 1).astype("float32") / 255.0
        tensors.append(image_tensor)

    calibration_array = np.stack(tensors, axis=0)
    return calibration_array


def main(argv: list[str] | None = None) -> int:
    """Entry point for building a YOLO11 calibration tensor.

    Parameters
    ----------
    argv :
        Optional list of argument strings; if None, `sys.argv` is used.

    Returns
    -------
    int
        Exit code, 0 on success and non-zero on error.
    """
    args = parse_args(argv)

    list_path = Path(args.list)
    out_path = Path(args.out)

    image_paths = read_image_list(list_path)

    print(f"Loaded {len(image_paths)} image paths from {list_path}")
    print(f"Building calibration tensor at size {args.imgsz}x{args.imgsz}...")

    calibration_array = build_calibration_tensor(image_paths, img_size=args.imgsz)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, calibration_array)

    print(
        f"Saved calibration tensor with shape {calibration_array.shape} "
        f"to {out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
