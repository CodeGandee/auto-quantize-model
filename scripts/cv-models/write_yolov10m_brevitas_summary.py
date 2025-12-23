#!/usr/bin/env python
"""Write a Markdown summary for YOLOv10m Brevitas PTQ/QAT runs.

The runner writes multiple artifacts under a run root, e.g.:
`tmp/yolov10m_brevitas_w4a8_w4a16/<run-id>/`.

This script aggregates:
- accuracy + latency from `*-coco/metrics.json`,
- calibration metadata from `ptq_*.json` / `qat_*.json`,
- QAT training metadata (Lightning logs, loss curve files).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write summary.md for a YOLOv10m Brevitas run root.")
    parser.add_argument("--run-root", type=Path, required=True, help="Run root directory under tmp/.")
    parser.add_argument("--out", type=Path, default=None, help="Output markdown path (defaults to <run-root>/summary.md).")
    return parser.parse_args(argv)


def json_load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    return json_load(path)


def format_float(value: Any, *, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "0.0000"


def format_ms(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return "0.000"


def iter_variant_metrics(run_root: Path) -> List[Tuple[str, Path]]:
    order = [
        ("baseline", run_root / "baseline-coco" / "metrics.json"),
        ("ptq-w8a16", run_root / "ptq-w8a16-coco" / "metrics.json"),
        ("ptq-w8a8", run_root / "ptq-w8a8-coco" / "metrics.json"),
        ("ptq-w4a16", run_root / "ptq-w4a16-coco" / "metrics.json"),
        ("ptq-w4a8", run_root / "ptq-w4a8-coco" / "metrics.json"),
        ("qat-w4a8-pl", run_root / "qat-w4a8-pl-coco" / "metrics.json"),
        ("qat-w4a8", run_root / "qat-w4a8-coco" / "metrics.json"),
    ]
    return [(name, path) for name, path in order if path.is_file()]


def find_eval_reference(metrics_payloads: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for payload in metrics_payloads:
        if payload.get("instances") and payload.get("images_dir"):
            return payload
    return None


def build_summary(run_root: Path) -> str:
    variants = iter_variant_metrics(run_root)
    payloads: List[Dict[str, Any]] = []
    for _, path in variants:
        payloads.append(json_load(path))

    eval_ref = find_eval_reference(payloads)

    lines: List[str] = ["# Summary", ""]
    lines.append("| variant | mAP_50_95 | mAP_50 | mean_ms | p90_ms | providers | onnx |")
    lines.append("|---|---:|---:|---:|---:|---|---|")

    for name, metrics_path in variants:
        payload = json_load(metrics_path)
        metrics = payload.get("metrics", {})
        latency = payload.get("latency", {})
        providers = " ".join(payload.get("providers", []))
        onnx_path = payload.get("onnx_path", "")
        lines.append(
            "| {name} | {m5095} | {m50} | {mean} | {p90} | {providers} | `{onnx}` |".format(
                name=name,
                m5095=format_float(metrics.get("mAP_50_95", 0.0)),
                m50=format_float(metrics.get("mAP_50", 0.0)),
                mean=format_ms(latency.get("mean_ms", 0.0)),
                p90=format_ms(latency.get("p90_ms", 0.0)),
                providers=providers,
                onnx=onnx_path,
            )
        )

    lines.append("")
    lines.append("## Datasets")
    lines.append("")

    if eval_ref is not None:
        lines.append("### Evaluation")
        lines.append("")
        lines.append(f"- `data_root`: `{eval_ref.get('data_root')}`")
        lines.append(f"- `instances`: `{eval_ref.get('instances')}`")
        lines.append(f"- `images_dir`: `{eval_ref.get('images_dir')}`")
        lines.append(f"- `max_images`: `{eval_ref.get('max_images')}`")
        lines.append(f"- `imgsz`: `{eval_ref.get('imgsz')}`")
        lines.append(f"- `conf`: `{eval_ref.get('conf')}`")
        lines.append(f"- `iou`: `{eval_ref.get('iou')}`")
        lines.append(f"- `max_det`: `{eval_ref.get('max_det')}`")
        lines.append(f"- `pre_nms_topk`: `{eval_ref.get('pre_nms_topk')}`")
        lines.append(f"- `warmup_runs`: `{eval_ref.get('warmup_runs')}`")
        lines.append(f"- `skip_latency`: `{eval_ref.get('skip_latency')}`")
        lines.append("")

    lines.append("### Calibration (A8 variants)")
    lines.append("")
    calib_sources = [
        ("ptq-w8a8", run_root / "ptq_w8a8_export.json"),
        ("ptq-w4a8", run_root / "ptq_w4a8_export.json"),
        ("qat-w4a8-pl", run_root / "qat_w4a8_lightning_export.json"),
        ("qat-w4a8", run_root / "qat_w4a8_export.json"),
    ]
    any_calib = False
    for name, export_path in calib_sources:
        export_payload = maybe_load(export_path)
        if export_payload is None:
            continue
        calib = export_payload.get("calibration")
        if not isinstance(calib, dict):
            continue
        any_calib = True
        lines.append(f"- `{name}`: list=`{calib.get('image_list_path')}`, used={calib.get('calib_images')}, batch={calib.get('batch_size')}, device=`{calib.get('device')}`")
    if not any_calib:
        lines.append("- (none found)")
    lines.append("")

    qat_payload = maybe_load(run_root / "qat_w4a8_lightning_export.json")
    if qat_payload and isinstance(qat_payload.get("qat_dataset"), dict):
        qat_dataset = qat_payload["qat_dataset"]
        lines.append("### QAT (Lightning)")
        lines.append("")
        lines.append(f"- `dataset_yaml`: `{qat_dataset.get('dataset_yaml')}`")
        lines.append(f"- `dataset_root`: `{qat_dataset.get('dataset_root')}`")
        lines.append(f"- `train_images`: `{qat_dataset.get('train_images')}` (list: `{qat_dataset.get('train_list_path')}`)")
        lines.append(f"- `val_images`: `{qat_dataset.get('val_images')}` (val_max_images: `{qat_dataset.get('val_max_images')}`)")
        if isinstance(qat_payload.get("lightning"), dict):
            lt = qat_payload["lightning"]
            lines.append(f"- TensorBoard: `{lt.get('log_dir')}`")
            lines.append(f"- Loss curve: `{lt.get('loss_curve_png')}` (csv: `{lt.get('loss_curve_csv')}`)")
        lines.append("")

    return "\n".join(lines) + "\n"


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    run_root = args.run_root
    out_path = args.out or (run_root / "summary.md")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_summary(run_root), encoding="utf-8")
    print(out_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
