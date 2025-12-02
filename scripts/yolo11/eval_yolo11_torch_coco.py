#!/usr/bin/env python
"""
Evaluate YOLO11 PyTorch checkpoints on COCO 2017 val using pycocotools.

This script runs single-image inference over a subset of COCO 2017 val,
computes COCO mAP (bbox) metrics, and measures simple latency statistics.

Example:
    pixi run python scripts/yolo11/eval_yolo11_torch_coco.py \\
        --model models/yolo11/checkpoints/yolo11n.pt \\
        --data-root datasets/coco2017/source-data \\
        --max-images 500 \\
        --device 0 \\
        --precision fp16 \\
        --out datasets/quantize-calib/baseline_yolo11n_fp16_coco.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


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
        description="Evaluate a YOLO11 PyTorch checkpoint on COCO 2017 val.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolo11/checkpoints/yolo11n.pt",
        help="Path to YOLO11 PyTorch checkpoint (.pt).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="datasets/coco2017/source-data",
        help="Root of COCO2017 dataset (should contain val2017/ and annotations/instances_val2017.json).",
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
        "--device",
        type=str,
        default="0",
        help="Device for inference (e.g., '0', 'cuda:0', or 'cpu').",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=("fp32", "fp16"),
        default="fp16",
        help="Inference precision for the PyTorch model.",
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
        "--out",
        type=str,
        default=None,
        help="Optional output JSON path for metrics and latency stats.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Confidence threshold for detections (default: 0.001).",
    )
    return parser.parse_args(argv)


def setup_ultralytics() -> None:
    """Ensure the local Ultralytics YOLO11 source is importable.

    This function prepends the cloned YOLO11 source directory under
    `models/yolo11/src` to `sys.path` so that all Ultralytics imports
    use the repository-managed code rather than any global installation.
    """
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[2]
    yolo_root = repo_root / "models" / "yolo11"
    src_dir = yolo_root / "src"
    if src_dir.is_dir():
        sys.path.insert(0, str(src_dir))


def load_model(model_path: Path, device: str, precision: str):
    """Load a YOLO11 detection model with the requested precision.

    Parameters
    ----------
    model_path :
        Path to the YOLO11 checkpoint file.
    device :
        Device selector string (e.g., ``\"0\"``, ``\"cuda:0\"``, or ``\"cpu\"``).
    precision :
        Inference precision, either ``\"fp32\"`` or ``\"fp16\"``.

    Returns
    -------
    tuple
        Loaded YOLO model and a boolean flag indicating whether FP16
        inference should be requested via the Ultralytics API.
    """
    from ultralytics import YOLO  # type: ignore[import]

    yolo_model = YOLO(str(model_path))
    # Precision (PyTorch half mode) is configured via predict arguments; YOLO will move model to device as needed.
    fp16 = precision.lower() == "fp16"
    return yolo_model, fp16


def build_category_map(coco: COCO) -> Dict[str, int]:
    """Build a mapping from category name to COCO category id.

    Parameters
    ----------
    coco :
        An initialized ``pycocotools.coco.COCO`` object for the target
        annotation file.

    Returns
    -------
    dict
        Mapping from category name to COCO category id.
    """
    cats = coco.loadCats(coco.getCatIds())
    return {c["name"]: int(c["id"]) for c in cats}


def evaluate(
    model,
    fp16: bool,
    coco: COCO,
    name_to_cat_id: Dict[str, int],
    images_dir: Path,
    device: str,
    imgsz: int,
    max_images: int,
    conf: float,
) -> Tuple[Dict[str, Any], List[float]]:
    """Run inference and compute COCO bbox mAP plus per-image latency.

    Parameters
    ----------
    model :
        YOLO11 model instance loaded from the Ultralytics API.
    fp16 :
        Whether to request half-precision inference from the model.
    coco :
        COCO ground-truth wrapper for val annotations.
    name_to_cat_id :
        Mapping from model class names to COCO category ids.
    images_dir :
        Directory containing COCO val2017 images.
    device :
        Inference device string.
    imgsz :
        Inference image size (square).
    max_images :
        Maximum number of images to evaluate (0 means all).
    conf :
        Confidence threshold for filtering detections.

    Returns
    -------
    tuple
        A tuple ``(metrics, latencies)`` where ``metrics`` is a dict of
        COCO mAP statistics and ``latencies`` is a list of per-image
        inference times in seconds.
    """
    # Collect detections in COCO-json style
    detections: List[Dict[str, Any]] = []
    latencies: List[float] = []

    img_ids = sorted(coco.getImgIds())
    if max_images > 0:
        img_ids = img_ids[:max_images]
    imgs = coco.loadImgs(img_ids)

    # Use model.names for mapping; ensure keys are str->str or int->str
    names = model.names if hasattr(model, "names") else {}

    for img in imgs:
        img_id = int(img["id"])
        file_name = img["file_name"]
        image_path = images_dir / file_name

        if not image_path.is_file():
            raise FileNotFoundError(f"Image for COCO id {img_id} not found at {image_path}")

        start = time.perf_counter()
        results_list = model.predict(
            source=str(image_path),
            imgsz=imgsz,
            device=device,
            half=fp16,
            conf=conf,
            verbose=False,
        )
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

        if not results_list:
            continue

        result = results_list[0]
        boxes = result.boxes
        if boxes is None or boxes.data is None:
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()

        for (x1, y1, x2, y2), score, cls_idx in zip(xyxy, scores, classes):
            cls_idx_int = int(cls_idx)
            name = names.get(cls_idx_int, str(cls_idx_int))
            cat_id = name_to_cat_id.get(name)
            if cat_id is None:
                # Skip categories that do not map directly to COCO ids.
                continue
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

    # Run COCO evaluation
    coco_dt = coco.loadRes(detections) if detections else coco.loadRes([])
    coco_eval = COCOeval(coco, coco_dt, iouType="bbox")
    coco_eval.params.imgIds = [int(i) for i in img_ids]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats  # type: ignore[assignment]
    # COCOeval stats meaning:
    # 0: mAP@[0.5:0.95], 1: mAP@0.5, 2: mAP@0.75, 3..5: small/medium/large
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
    times = np.array(latencies, dtype=np.float64)
    if times.size == 0:
        return {"mean_ms": 0.0, "median_ms": 0.0, "p90_ms": 0.0, "throughput_fps": 0.0}
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
    """Entry point for YOLO11 COCO evaluation.

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
    setup_ultralytics()

    model_path = Path(args.model)
    data_root = Path(args.data_root)
    instances_path = Path(args.instances) if args.instances else data_root / "annotations" / "instances_val2017.json"
    images_dir = Path(args.images_dir) if args.images_dir else data_root / "val2017"

    if not model_path.is_file():
        print(f"Error: model checkpoint not found at {model_path}", file=sys.stderr)
        return 1
    if not instances_path.is_file():
        print(f"Error: COCO instances file not found at {instances_path}", file=sys.stderr)
        return 1
    if not images_dir.is_dir():
        print(f"Error: images directory not found at {images_dir}", file=sys.stderr)
        return 1

    print(f"Loading COCO annotations from {instances_path} ...")
    coco = COCO(str(instances_path))
    name_to_cat_id = build_category_map(coco)

    print(f"Loading YOLO11 model from {model_path} ...")
    model, fp16 = load_model(model_path, device=args.device, precision=args.precision)

    print(
        f"Running evaluation on up to {args.max_images} images from {images_dir} "
        f"with precision={args.precision}, device={args.device}, imgsz={args.imgsz}, conf={args.conf} ..."
    )
    metrics, latencies = evaluate(
        model=model,
        fp16=fp16,
        coco=coco,
        name_to_cat_id=name_to_cat_id,
        images_dir=images_dir,
        device=args.device,
        imgsz=args.imgsz,
        max_images=args.max_images,
        conf=args.conf,
    )

    latency_stats = summarize_latency(latencies)

    print("COCO bbox metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("Latency stats:")
    for k, v in latency_stats.items():
        print(f"  {k}: {v:.3f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": str(model_path),
            "data_root": str(data_root),
            "instances": str(instances_path),
            "images_dir": str(images_dir),
            "device": args.device,
            "precision": args.precision,
            "imgsz": args.imgsz,
            "max_images": args.max_images,
            "conf": args.conf,
            "metrics": metrics,
            "latency": latency_stats,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Wrote metrics JSON to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
