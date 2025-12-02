#!/usr/bin/env python
"""
Evaluate a quantized YOLO11 ONNX model on COCO 2017 val using ONNX Runtime.

This script mirrors the PyTorch-based evaluation in
``scripts/yolo11/eval_yolo11_torch_coco.py`` but runs inference through
ONNX Runtime on a (potentially quantized) YOLO11 ONNX model. It computes
COCO bbox mAP metrics via ``pycocotools`` and reports simple latency
statistics.

Example
-------
Evaluate an INT8 Q/DQ ONNX export of YOLO11n on 100 COCO val images:

>>> pixi run python scripts/yolo11/eval_yolo11_onnx_coco.py \\
...     --onnx_path models/yolo11/onnx/yolo11n-int8-qdq-proto.onnx \\
...     --data-root datasets/coco2017/source-data \\
...     --max-images 100 \\
...     --providers CUDAExecutionProvider CPUExecutionProvider \\
...     --imgsz 640 \\
...     --out datasets/quantize-calib/yolo11n_int8_onnx_coco100.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import yaml


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
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
        description="Evaluate a YOLO11 ONNX model (e.g., INT8 Q/DQ) on COCO 2017 val.",
    )
    parser.add_argument(
        "--onnx_path",
        type=str,
        required=True,
        help="Path to the YOLO11 ONNX model (quantized or baseline).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="datasets/coco2017/source-data",
        help="COCO2017 root directory with val2017/ and annotations/instances_val2017.json.",
    )
    parser.add_argument(
        "--instances",
        type=str,
        default=None,
        help="Path to instances_val2017.json (defaults to <data-root>/annotations/instances_val2017.json).",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Directory with COCO val2017 images (defaults to <data-root>/val2017).",
    )
    parser.add_argument(
        "--coco-yaml",
        type=str,
        default="models/yolo11/src/ultralytics/cfg/datasets/coco.yaml",
        help="Path to the Ultralytics COCO dataset YAML defining class names.",
    )
    parser.add_argument(
        "--providers",
        type=str,
        nargs="+",
        default=["CUDAExecutionProvider", "CPUExecutionProvider"],
        help="ONNX Runtime execution providers, in priority order.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (square, default: 640).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=500,
        help="Maximum number of COCO val images to evaluate (default: 500).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Confidence threshold before NMS (default: 0.001).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="IoU threshold for NMS (default: 0.7).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output JSON path for metrics and latency stats.",
    )
    return parser.parse_args(argv)


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
        Input image as an HWC uint8 array in BGR or RGB order (color is applied in same order).
    new_shape :
        Target shape as an integer (square) or (height, width) tuple.
    color :
        Padding color.

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


def preprocess_image(
    image_path: Path,
    img_size: int,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int, int, int]]:
    """Load and preprocess an image for YOLO11 ONNX inference.

    Parameters
    ----------
    image_path :
        Path to the input image.
    img_size :
        Target square size for letterbox.

    Returns
    -------
    tuple
        A tuple ``(input_tensor, original_shape, pads)`` where:

        * ``input_tensor`` is a float32 array with shape (1, 3, img_size, img_size)
          in RGB order, values scaled to [0, 1].
        * ``original_shape`` is (height, width) of the original image.
        * ``pads`` is the letterbox padding (pad_left, pad_top, pad_right, pad_bottom).
    """
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image at {image_path}")

    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    letterboxed = letterbox(image_rgb, new_shape=img_size)

    # Compute padding info
    scale = min(img_size / w, img_size / h)
    resized_w = int(round(w * scale))
    resized_h = int(round(h * scale))
    pad_w = img_size - resized_w
    pad_h = img_size - resized_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    # HWC -> CHW and normalize to [0, 1]
    tensor = letterboxed.transpose(2, 0, 1).astype("float32") / 255.0
    tensor = np.expand_dims(tensor, axis=0)
    return tensor, (h, w), (pad_left, pad_top, pad_right, pad_bottom)


def xywh_to_xyxy(
    xywh: np.ndarray,
    pads: Tuple[int, int, int, int],
    orig_shape: Tuple[int, int],
    img_size: int,
) -> np.ndarray:
    """Convert xywh coords in letterboxed space back to original-image xyxy.

    Parameters
    ----------
    xywh :
        Array of shape (N, 4) with [x_center, y_center, width, height]
        in letterboxed image coordinates.
    pads :
        Tuple (pad_left, pad_top, pad_right, pad_bottom).
    orig_shape :
        Original image shape (height, width).
    img_size :
        Letterboxed image size (square).

    Returns
    -------
    np.ndarray
        Array of shape (N, 4) with [x1, y1, x2, y2] in original-image pixels.
    """
    pad_left, pad_top, _, _ = pads
    orig_h, orig_w = orig_shape

    # Undo letterbox padding
    x, y, w, h = np.split(xywh, 4, axis=1)
    x = x - pad_left
    y = y - pad_top

    # Compute scaling from letterboxed dimensions to original
    scale = min(img_size / orig_w, img_size / orig_h)
    x /= scale
    y /= scale
    w /= scale
    h /= scale

    # Convert center-xywh to xyxy
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0

    return np.concatenate([x1, y1, x2, y2], axis=1)


def nms_xyxy(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_thres: float,
) -> np.ndarray:
    """Perform class-agnostic NMS on xyxy boxes.

    Parameters
    ----------
    boxes :
        Array of shape (N, 4) with [x1, y1, x2, y2].
    scores :
        Array of shape (N,) with detection scores.
    iou_thres :
        IoU threshold for suppression.

    Returns
    -------
    np.ndarray
        Indices of boxes kept after NMS.
    """
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
    order = scores.argsort()[::-1]

    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = (xx2 - xx1).clip(min=0)
        h = (yy2 - yy1).clip(min=0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)

        remaining = np.where(iou <= iou_thres)[0]
        order = order[remaining + 1]

    return np.array(keep, dtype=np.int64)


def load_coco_names(coco_yaml_path: Path) -> List[str]:
    """Load COCO class names from an Ultralytics coco.yaml file."""
    data = yaml.safe_load(coco_yaml_path.read_text(encoding="utf-8"))
    names = data.get("names", {})
    if isinstance(names, dict):
        # names: {0: "person", 1: "bicycle", ...}
        return [names[i] for i in sorted(names.keys())]
    if isinstance(names, list):
        return names
    raise ValueError(f"Unexpected names format in {coco_yaml_path}")


def build_name_to_cat_id(coco: COCO) -> Dict[str, int]:
    """Build a mapping from category name to COCO category id."""
    cats = coco.loadCats(coco.getCatIds())
    return {c["name"]: int(c["id"]) for c in cats}


def evaluate_onnx(
    session: ort.InferenceSession,
    coco: COCO,
    class_names: List[str],
    name_to_cat_id: Dict[str, int],
    images_dir: Path,
    img_size: int,
    max_images: int,
    conf_thres: float,
    iou_thres: float,
) -> Tuple[Dict[str, Any], List[float]]:
    """Run ONNX Runtime inference and compute COCO bbox mAP and latency.

    Parameters
    ----------
    session :
        ONNX Runtime inference session for the YOLO11 model.
    coco :
        COCO API object for val2017 annotations.
    class_names :
        List of class names in index order for the ONNX model.
    name_to_cat_id :
        Mapping from class name to COCO category id.
    images_dir :
        Directory containing COCO val2017 images.
    img_size :
        Inference image size (square).
    max_images :
        Maximum number of images to evaluate (0 means all).
    conf_thres :
        Confidence threshold before NMS.
    iou_thres :
        IoU threshold for NMS.

    Returns
    -------
    tuple
        A tuple ``(metrics, latencies)`` where ``metrics`` is a dict of
        COCO mAP statistics and ``latencies`` is a list of per-image
        inference times in seconds.
    """
    input_name = session.get_inputs()[0].name
    detections: List[Dict[str, Any]] = []
    latencies: List[float] = []

    img_ids = sorted(coco.getImgIds())
    if max_images > 0:
        img_ids = img_ids[:max_images]
    imgs = coco.loadImgs(img_ids)

    for img in imgs:
        img_id = int(img["id"])
        file_name = img["file_name"]
        image_path = images_dir / file_name

        if not image_path.is_file():
            raise FileNotFoundError(f"Image for COCO id {img_id} not found at {image_path}")

        input_tensor, orig_shape, pads = preprocess_image(image_path, img_size=img_size)

        start = time.perf_counter()
        outputs = session.run(None, {input_name: input_tensor})
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

        if not outputs:
            continue

        # Assume Ultralytics YOLO-style output: [1, 84, 8400] -> [8400, 84]
        raw = outputs[0]
        if raw.ndim != 3:
            raise ValueError(f"Unexpected ONNX output shape {raw.shape}, expected [1, 84, N].")
        pred = np.squeeze(raw, axis=0)  # [84, N]
        pred = pred.transpose(1, 0)  # [N, 84]

        xywh = pred[:, :4]
        cls_scores = pred[:, 4:]

        scores = cls_scores.max(axis=1)
        cls_indices = cls_scores.argmax(axis=1)

        keep = scores >= conf_thres
        if not np.any(keep):
            continue

        xywh = xywh[keep]
        scores = scores[keep]
        cls_indices = cls_indices[keep]

        # Convert back to original-image xyxy
        boxes_xyxy = xywh_to_xyxy(xywh, pads=pads, orig_shape=orig_shape, img_size=img_size)

        # Apply NMS globally (class-agnostic) for simplicity
        keep_idx = nms_xyxy(boxes_xyxy, scores, iou_thres=iou_thres)
        boxes_xyxy = boxes_xyxy[keep_idx]
        scores = scores[keep_idx]
        cls_indices = cls_indices[keep_idx]

        for bbox, score, cls_idx in zip(boxes_xyxy, scores, cls_indices):
            cls_idx_int = int(cls_idx)
            if cls_idx_int < 0 or cls_idx_int >= len(class_names):
                continue
            name = class_names[cls_idx_int]
            cat_id = name_to_cat_id.get(name)
            if cat_id is None:
                continue

            x1, y1, x2, y2 = bbox.tolist()
            w = x2 - x1
            h = y2 - y1
            detections.append(
                {
                    "image_id": img_id,
                    "category_id": int(cat_id),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score),
                }
            )

    # COCO evaluation
    coco_dt = coco.loadRes(detections) if detections else coco.loadRes([])
    coco_eval = COCOeval(coco, coco_dt, iouType="bbox")
    coco_eval.params.imgIds = [int(i) for i in img_ids]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats  # type: ignore[assignment]
    metrics = {
        "mAP_50_95": float(stats[0]),
        "mAP_50": float(stats[1]),
        "mAP_75": float(stats[2]),
        "mAP_small": float(stats[3]),
        "mAP_medium": float(stats[4]),
        "mAP_large": float(stats[5]),
        "num_images": len(img_ids),
        "num_detections": len(detections),
    }
    return metrics, latencies


def summarize_latency(latencies: List[float]) -> Dict[str, float]:
    """Summarize per-image latency statistics.

    Parameters
    ----------
    latencies :
        List of per-image inference times in seconds.

    Returns
    -------
    dict
        Dictionary containing mean, median, and p90 latency in
        milliseconds plus throughput in frames per second.
    """
    if not latencies:
        return {"mean_ms": 0.0, "median_ms": 0.0, "p90_ms": 0.0, "throughput_fps": 0.0}

    times = np.array(latencies, dtype=np.float64)
    mean_ms = float(times.mean() * 1000.0)
    median_ms = float(np.median(times) * 1000.0)
    p90_ms = float(np.percentile(times, 90) * 1000.0)
    throughput_fps = float(1.0 / times.mean())
    return {
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p90_ms": p90_ms,
        "throughput_fps": throughput_fps,
    }


def main(argv: List[str] | None = None) -> int:
    """Entry point for evaluating a YOLO11 ONNX model on COCO."""
    args = parse_args(argv)

    onnx_path = Path(args.onnx_path)
    data_root = Path(args.data_root)
    instances_path = Path(args.instances) if args.instances else data_root / "annotations" / "instances_val2017.json"
    images_dir = Path(args.images_dir) if args.images_dir else data_root / "val2017"
    coco_yaml_path = Path(args.coco_yaml)

    if not onnx_path.is_file():
        print(f"Error: ONNX model not found at {onnx_path}", file=sys.stderr)
        return 1
    if not instances_path.is_file():
        print(f"Error: COCO instances file not found at {instances_path}", file=sys.stderr)
        return 1
    if not images_dir.is_dir():
        print(f"Error: images directory not found at {images_dir}", file=sys.stderr)
        return 1
    if not coco_yaml_path.is_file():
        print(f"Error: COCO YAML not found at {coco_yaml_path}", file=sys.stderr)
        return 1

    print(f"Loading COCO annotations from {instances_path} ...")
    coco = COCO(str(instances_path))
    name_to_cat_id = build_name_to_cat_id(coco)
    class_names = load_coco_names(coco_yaml_path)

    print(f"Creating ONNX Runtime session for {onnx_path} ...")
    sess_options = ort.SessionOptions()
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_options,
        providers=args.providers,
    )

    print(
        f"Running ONNX evaluation on up to {args.max_images} images from {images_dir} "
        f"with providers={args.providers}, imgsz={args.imgsz}, conf={args.conf}, iou={args.iou} ..."
    )
    metrics, latencies = evaluate_onnx(
        session=session,
        coco=coco,
        class_names=class_names,
        name_to_cat_id=name_to_cat_id,
        images_dir=images_dir,
        img_size=args.imgsz,
        max_images=args.max_images,
        conf_thres=args.conf,
        iou_thres=args.iou,
    )

    latency_stats = summarize_latency(latencies)

    print("COCO bbox metrics (ONNX):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("Latency stats (ONNX):")
    for k, v in latency_stats.items():
        print(f"  {k}: {v:.3f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "onnx_path": str(onnx_path),
            "data_root": str(data_root),
            "instances": str(instances_path),
            "images_dir": str(images_dir),
            "providers": args.providers,
            "imgsz": args.imgsz,
            "max_images": args.max_images,
            "conf": args.conf,
            "iou": args.iou,
            "metrics": metrics,
            "latency": latency_stats,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Wrote ONNX metrics JSON to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

