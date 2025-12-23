#!/usr/bin/env python
"""Evaluate a YOLOv10m ONNX model on COCO 2017 val using ONNX Runtime.

This evaluator supports the YOLOv10m ONNX output format documented in
`models/cv-models/yolov10m/README.md`:

  output shape: [1, 144, 8400]

Where 144 = 64 (DFL box regression with reg_max=16) + 80 class logits.

Example (100-image medium subset):
    pixi run python scripts/cv-models/eval_yolov10m_onnx_coco.py \\
        --onnx-path models/cv-models/yolov10m/checkpoints/yolov10m.onnx \\
        --data-root datasets/coco2017/source-data \\
        --max-images 100 \\
        --providers CPUExecutionProvider \\
        --imgsz 640 \\
        --out tmp/yolov10m_lowbit/<run-id>/baseline-coco/metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import onnxruntime as ort
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import yaml

from auto_quantize_model.cv_models.yolo_preprocess import (
    find_repo_root,
    preprocess_image_path,
    unletterbox_boxes_xyxy,
)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a YOLOv10m ONNX model (baseline or quantized) on COCO 2017 val.",
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        required=True,
        help="Path to the YOLOv10m ONNX model (quantized or baseline).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("datasets/coco2017/source-data"),
        help="COCO2017 root directory with val2017/ and annotations/instances_val2017.json.",
    )
    parser.add_argument(
        "--instances",
        type=Path,
        default=None,
        help="Path to instances_val2017.json (defaults to <data-root>/annotations/instances_val2017.json).",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Directory with COCO val2017 images (defaults to <data-root>/val2017).",
    )
    parser.add_argument(
        "--coco-yaml",
        type=Path,
        default=Path("models/yolo10/src/ultralytics/cfg/datasets/coco.yaml"),
        help="Path to the Ultralytics COCO dataset YAML defining class names.",
    )
    parser.add_argument(
        "--providers",
        type=str,
        nargs="+",
        default=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        help="ONNX Runtime execution providers, in priority order.",
    )
    parser.add_argument(
        "--disable-cpu-fallback",
        action="store_true",
        help=(
            "Fail instead of silently falling back to CPU if a higher-priority EP "
            "(e.g., CUDA) is present but cannot initialize."
        ),
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Run N warmup inferences on a zero tensor before evaluation (default: 0).",
    )
    parser.add_argument(
        "--skip-latency",
        type=int,
        default=0,
        help="Skip the first N per-image latency samples when computing latency stats (default: 0).",
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
        default=100,
        help="Maximum number of COCO val images to evaluate (default: 100).",
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
        "--max-det",
        type=int,
        default=300,
        help="Maximum detections per image after NMS (default: 300).",
    )
    parser.add_argument(
        "--pre-nms-topk",
        type=int,
        default=30000,
        help="Keep top-K candidates by score before NMS (default: 30000).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output JSON path for metrics and latency stats.",
    )
    parser.add_argument(
        "--detections-out",
        type=Path,
        default=None,
        help="Optional path to write COCO-format detections JSON.",
    )
    parser.add_argument(
        "--image-ids-list",
        type=Path,
        default=None,
        help="Optional text file listing COCO image ids (one per line) for a fixed subset.",
    )
    return parser.parse_args(argv)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x - x_max)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def load_coco_names(coco_yaml_path: Path) -> List[str]:
    data = yaml.safe_load(coco_yaml_path.read_text(encoding="utf-8"))
    names = data.get("names", {})
    if isinstance(names, dict):
        return [names[i] for i in sorted(names.keys())]
    if isinstance(names, list):
        return names
    raise ValueError(f"Unexpected names format in {coco_yaml_path}")


def build_name_to_cat_id(coco: COCO) -> Dict[str, int]:
    cats = coco.loadCats(coco.getCatIds())
    return {c["name"]: int(c["id"]) for c in cats}


def build_yolo_anchors(
    *,
    img_size: int,
    strides: Sequence[int] = (8, 16, 32),
) -> Tuple[np.ndarray, np.ndarray]:
    """Return anchor points and per-anchor stride for a 3-head YOLO detector."""

    anchors: list[np.ndarray] = []
    stride_per_anchor: list[np.ndarray] = []
    for stride in strides:
        grid = img_size // int(stride)
        grid_y, grid_x = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
        points = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2).astype(np.float32)
        points = (points + 0.5) * float(stride)
        anchors.append(points)
        stride_per_anchor.append(np.full((points.shape[0],), float(stride), dtype=np.float32))

    anchors_all = np.concatenate(anchors, axis=0)
    stride_all = np.concatenate(stride_per_anchor, axis=0)
    return anchors_all, stride_all


def dfl_integral(dfl_logits: np.ndarray) -> np.ndarray:
    """Convert DFL logits [N, 4, 16] to distances [N, 4] (in grid units)."""

    probs = softmax(dfl_logits, axis=-1)
    bins = np.arange(dfl_logits.shape[-1], dtype=np.float32)
    distances = (probs * bins[None, None, :]).sum(axis=-1)
    return distances.astype(np.float32)


def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    x, y, w, h = np.split(xywh, 4, axis=1)
    return np.concatenate([x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0], axis=1)


def nms_xyxy(boxes_xyxy: np.ndarray, scores: np.ndarray, *, iou_thres: float, max_det: int) -> np.ndarray:
    if boxes_xyxy.size == 0:
        return np.empty((0,), dtype=np.int64)

    x1 = boxes_xyxy[:, 0]
    y1 = boxes_xyxy[:, 1]
    x2 = boxes_xyxy[:, 2]
    y2 = boxes_xyxy[:, 3]

    areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
    order = scores.argsort()[::-1]

    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if len(keep) >= max_det or order.size == 1:
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


def decode_yolo_output(
    output: np.ndarray,
    *,
    anchors: np.ndarray,
    stride_per_anchor: np.ndarray,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
    pre_nms_topk: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode YOLO outputs to per-image detections in letterboxed xyxy space."""

    raw = output
    if raw.ndim == 3 and raw.shape[0] == 1:
        raw = raw[0]

    if raw.ndim != 2:
        raise ValueError(f"Unexpected ONNX output shape {output.shape}; expected [1, C, N] or [C, N].")

    if raw.shape[0] in {84, 144}:
        pred = raw.transpose(1, 0)  # [N, C]
    elif raw.shape[1] in {84, 144}:
        pred = raw
    else:
        raise ValueError(f"Unexpected ONNX output shape {output.shape}; could not find channel dim (84 or 144).")

    num_preds, channels = pred.shape
    if num_preds != anchors.shape[0]:
        raise ValueError(
            f"Anchor count mismatch: predictions={num_preds}, anchors={anchors.shape[0]}. "
            "This evaluator expects a 640x640 3-head YOLO layout (8400 preds)."
        )

    if channels == 84:
        xywh = pred[:, :4]
        cls_prob = sigmoid(pred[:, 4:])
        scores = cls_prob.max(axis=1)
        cls_ids = cls_prob.argmax(axis=1)

        keep = scores >= float(conf_thres)
        if not np.any(keep):
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)

        boxes = xywh_to_xyxy(xywh[keep])
        scores = scores[keep].astype(np.float32)
        cls_ids = cls_ids[keep].astype(np.int64)
    elif channels == 144:
        dfl = pred[:, :64].reshape(num_preds, 4, 16)
        cls_prob = sigmoid(pred[:, 64:])
        scores = cls_prob.max(axis=1)
        cls_ids = cls_prob.argmax(axis=1)

        keep = scores >= float(conf_thres)
        if not np.any(keep):
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)

        dfl = dfl[keep]
        scores = scores[keep].astype(np.float32)
        cls_ids = cls_ids[keep].astype(np.int64)
        anchors_keep = anchors[keep]
        strides_keep = stride_per_anchor[keep]

        dist = dfl_integral(dfl) * strides_keep[:, None]
        left = dist[:, 0]
        top = dist[:, 1]
        right = dist[:, 2]
        bottom = dist[:, 3]
        boxes = np.stack(
            [
                anchors_keep[:, 0] - left,
                anchors_keep[:, 1] - top,
                anchors_keep[:, 0] + right,
                anchors_keep[:, 1] + bottom,
            ],
            axis=1,
        ).astype(np.float32)
    else:
        raise ValueError(f"Unsupported output channel count: {channels}")

    if boxes.shape[0] > int(pre_nms_topk):
        order = scores.argsort()[::-1][: int(pre_nms_topk)]
        boxes = boxes[order]
        scores = scores[order]
        cls_ids = cls_ids[order]

    max_wh = 4096.0
    boxes_for_nms = boxes + (cls_ids.astype(np.float32)[:, None] * max_wh)
    keep_idx = nms_xyxy(boxes_for_nms, scores, iou_thres=float(iou_thres), max_det=int(max_det))
    return boxes[keep_idx], scores[keep_idx], cls_ids[keep_idx]


def summarize_latency(latencies: List[float], *, skip: int = 0) -> Dict[str, float]:
    samples = latencies[int(skip) :] if int(skip) > 0 else latencies
    if not samples:
        return {"mean_ms": 0.0, "median_ms": 0.0, "p90_ms": 0.0, "throughput_fps": 0.0}

    times = np.array(samples, dtype=np.float64)
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


def resolve_image_ids(coco: COCO, *, image_ids_list: Path | None, max_images: int) -> List[int]:
    if image_ids_list is None:
        img_ids = sorted(coco.getImgIds())
        return img_ids[: int(max_images)] if int(max_images) > 0 else img_ids

    img_ids: List[int] = []
    path = Path(image_ids_list)
    if not path.is_absolute():
        repo_root = find_repo_root(Path(__file__))
        path = (repo_root / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"COCO image id list not found: {path}")

    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        img_ids.append(int(raw))
    if int(max_images) > 0:
        img_ids = img_ids[: int(max_images)]
    return img_ids


def main(argv: List[str] | None = None) -> int:
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

    anchors, stride_per_anchor = build_yolo_anchors(img_size=int(args.imgsz))

    print(f"Creating ONNX Runtime session for {onnx_path} ...")
    available = set(ort.get_available_providers())
    requested = list(args.providers)
    providers = [p for p in requested if p in available]
    if not providers:
        raise RuntimeError(f"No requested providers available. requested={requested}, available={sorted(available)}")
    if providers != requested:
        print(f"[WARN] Provider list filtered: requested={requested} available={sorted(available)} using={providers}")

    sess_options = ort.SessionOptions()
    if args.disable_cpu_fallback:
        if "CPUExecutionProvider" in providers:
            print(
                "[WARN] --disable-cpu-fallback ignored because CPUExecutionProvider is explicitly enabled "
                "(ORT treats this as a conflicting configuration)."
            )
        else:
            sess_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

    session = ort.InferenceSession(str(onnx_path), sess_options=sess_options, providers=providers)
    input_name = session.get_inputs()[0].name

    warmup_runs = int(args.warmup_runs)
    if warmup_runs > 0:
        warmup_tensor = np.zeros((1, 3, int(args.imgsz), int(args.imgsz)), dtype=np.float32)
        print(f"[INFO] Warming up ORT session: warmup_runs={warmup_runs}")
        for _ in range(warmup_runs):
            _ = session.run(None, {input_name: warmup_tensor})

    img_ids = resolve_image_ids(coco, image_ids_list=args.image_ids_list, max_images=int(args.max_images))
    imgs = coco.loadImgs(img_ids)

    detections: List[Dict[str, Any]] = []
    latencies: List[float] = []

    print(
        f"Running ONNX evaluation on {len(imgs)} images with providers={args.providers}, "
        f"imgsz={args.imgsz}, conf={args.conf}, iou={args.iou} ..."
    )

    for idx, img in enumerate(imgs, start=1):
        img_id = int(img["id"])
        image_path = images_dir / str(img["file_name"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Image for COCO id {img_id} not found at {image_path}")

        input_tensor, meta = preprocess_image_path(image_path, img_size=int(args.imgsz), add_batch_dim=True)

        start = time.perf_counter()
        outputs = session.run(None, {input_name: input_tensor})
        latencies.append(time.perf_counter() - start)

        if not outputs:
            continue

        boxes, scores, cls_ids = decode_yolo_output(
            outputs[0],
            anchors=anchors,
            stride_per_anchor=stride_per_anchor,
            conf_thres=float(args.conf),
            iou_thres=float(args.iou),
            max_det=int(args.max_det),
            pre_nms_topk=int(args.pre_nms_topk),
        )
        boxes_orig = unletterbox_boxes_xyxy(boxes, meta=meta, clip=True)

        for bbox, score, cls_idx in zip(boxes_orig, scores, cls_ids):
            cls_idx_int = int(cls_idx)
            if cls_idx_int < 0 or cls_idx_int >= len(class_names):
                continue
            name = class_names[cls_idx_int]
            cat_id = name_to_cat_id.get(name)
            if cat_id is None:
                continue

            x1, y1, x2, y2 = bbox.tolist()
            w = max(0.0, float(x2 - x1))
            h = max(0.0, float(y2 - y1))
            detections.append(
                {
                    "image_id": img_id,
                    "category_id": int(cat_id),
                    "bbox": [float(x1), float(y1), w, h],
                    "score": float(score),
                }
            )

        if idx % 10 == 0 or idx == len(imgs):
            print(f"  processed {idx}/{len(imgs)} (detections so far: {len(detections)})")

    if args.detections_out:
        args.detections_out.parent.mkdir(parents=True, exist_ok=True)
        args.detections_out.write_text(json.dumps(detections), encoding="utf-8")
        print(f"Wrote detections JSON to {args.detections_out}")

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
    latency_stats = summarize_latency(latencies, skip=int(args.skip_latency))

    print("COCO bbox metrics (ONNX):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("Latency stats (ONNX, inference-only):")
    for k, v in latency_stats.items():
        print(f"  {k}: {v:.3f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "onnx_path": str(onnx_path),
            "data_root": str(data_root),
            "instances": str(instances_path),
            "images_dir": str(images_dir),
            "providers": list(args.providers),
            "imgsz": int(args.imgsz),
            "max_images": int(args.max_images),
            "warmup_runs": int(args.warmup_runs),
            "skip_latency": int(args.skip_latency),
            "conf": float(args.conf),
            "iou": float(args.iou),
            "max_det": int(args.max_det),
            "pre_nms_topk": int(args.pre_nms_topk),
            "metrics": metrics,
            "latency": latency_stats,
        }
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote metrics JSON to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
