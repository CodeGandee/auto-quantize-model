"""Shared YOLO-style image preprocessing utilities.

This module intentionally contains only the pieces needed to keep
calibration preprocessing and evaluation preprocessing consistent:

- Letterbox resize/pad to a fixed square (default 640x640).
- BGR (OpenCV) -> RGB conversion.
- CHW float32 tensor conversion and [0, 1] scaling.
- Reverse letterbox transform for predicted boxes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxMetadata:
    """Metadata describing a letterbox transform."""

    orig_height: int
    orig_width: int
    img_size: int
    scale: float
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int


def find_repo_root(start: Path) -> Path:
    """Locate the repository root by walking upwards until `pyproject.toml` exists."""

    current = start.resolve()
    for _ in range(20):
        if (current / "pyproject.toml").is_file():
            return current
        if current.parent == current:
            break
        current = current.parent
    raise RuntimeError(f"Failed to locate repo root from {start}")


def read_image_list(list_path: Path, *, repo_root: Path | None = None) -> List[Path]:
    """Read image paths (one per line) from a text file and resolve them."""

    if not list_path.is_file():
        raise FileNotFoundError(f"Image list not found at {list_path}")

    resolved: List[Path] = []
    with list_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            path = Path(raw).expanduser()
            if not path.is_absolute():
                if repo_root is None:
                    raise ValueError(
                        f"Found relative image path {raw!r} but repo_root is None (list={list_path})."
                    )
                path = (repo_root / path).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Image not found: {path}")
            resolved.append(path)

    if not resolved:
        raise ValueError(f"No image paths found in {list_path}")
    return resolved


def letterbox(
    image_rgb: np.ndarray,
    *,
    img_size: int,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, LetterboxMetadata]:
    """Resize and pad an RGB image to a square, preserving aspect ratio."""

    orig_h, orig_w = image_rgb.shape[:2]
    if orig_h <= 0 or orig_w <= 0:
        raise ValueError(f"Invalid image shape {image_rgb.shape}")

    scale = min(img_size / orig_w, img_size / orig_h)
    resized_w = int(round(orig_w * scale))
    resized_h = int(round(orig_h * scale))
    resized = cv2.resize(image_rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    pad_w = img_size - resized_w
    pad_h = img_size - resized_h

    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=color,
    )

    meta = LetterboxMetadata(
        orig_height=orig_h,
        orig_width=orig_w,
        img_size=img_size,
        scale=float(scale),
        pad_left=int(pad_left),
        pad_top=int(pad_top),
        pad_right=int(pad_right),
        pad_bottom=int(pad_bottom),
    )
    return padded, meta


def preprocess_image_path(
    image_path: Path,
    *,
    img_size: int,
    color: Tuple[int, int, int] = (114, 114, 114),
    add_batch_dim: bool = True,
) -> Tuple[np.ndarray, LetterboxMetadata]:
    """Load and preprocess an image for YOLO-style ONNX inference.

    Returns an RGB float32 tensor scaled to [0, 1] with shape:
      - (1, 3, img_size, img_size) if add_batch_dim=True
      - (3, img_size, img_size) if add_batch_dim=False
    """

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image at {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    letterboxed, meta = letterbox(image_rgb, img_size=img_size, color=color)

    tensor = letterboxed.transpose(2, 0, 1).astype(np.float32) / 255.0
    tensor = np.ascontiguousarray(tensor)
    if add_batch_dim:
        tensor = tensor[None, ...]
    return tensor, meta


def unletterbox_boxes_xyxy(
    boxes_xyxy: np.ndarray,
    *,
    meta: LetterboxMetadata,
    clip: bool = True,
) -> np.ndarray:
    """Map xyxy boxes from letterboxed img_size space back to original image."""

    if boxes_xyxy.size == 0:
        return boxes_xyxy.astype(np.float32)

    boxes = boxes_xyxy.astype(np.float32, copy=True)
    boxes[:, [0, 2]] -= float(meta.pad_left)
    boxes[:, [1, 3]] -= float(meta.pad_top)
    boxes /= float(meta.scale)

    if clip:
        boxes[:, 0] = boxes[:, 0].clip(0.0, float(meta.orig_width))
        boxes[:, 2] = boxes[:, 2].clip(0.0, float(meta.orig_width))
        boxes[:, 1] = boxes[:, 1].clip(0.0, float(meta.orig_height))
        boxes[:, 3] = boxes[:, 3].clip(0.0, float(meta.orig_height))
    return boxes


def write_command_notes(out_path: Path, *, commands: Sequence[str]) -> None:
    """Write a short Markdown file capturing commands for reproducibility."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Notes", "", "Commands:", ""]
    for cmd in commands:
        lines.append(f"- `{cmd}`")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def batched(iterable: Sequence[Path], batch_size: int) -> Iterable[List[Path]]:
    """Yield list batches from a sequence."""

    safe = max(int(batch_size), 1)
    for start in range(0, len(iterable), safe):
        yield list(iterable[start : start + safe])

