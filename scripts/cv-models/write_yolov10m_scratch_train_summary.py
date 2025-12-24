#!/usr/bin/env python
"""Write a run-local summary for YOLOv10m scratch FP16 vs W4A16 QAT runs.

Expected run layout (see runner script):
  <run-root>/
    fp16/run_summary.json
    qat-w4a16/run_summary.json
    eval/fp16/metrics.json
    eval/qat-w4a16/metrics.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args(argv)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_loss_curve_csv(path: Path) -> Tuple[List[int], List[float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        epochs: List[int] = []
        loss: List[float] = []
        for row in reader:
            epochs.append(int(float(str(row.get("epoch", "0")).strip())))
            loss.append(float(str(row.get("train_loss_total", "0")).strip()))
    return epochs, loss


def _plot_compare(
    *,
    fp16_csv: Path,
    qat_csv: Path,
    out_png: Path,
) -> None:
    fp16_epochs, fp16_loss = _read_loss_curve_csv(fp16_csv)
    qat_epochs, qat_loss = _read_loss_curve_csv(qat_csv)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    if fp16_epochs:
        plt.plot(fp16_epochs, fp16_loss, linewidth=1.0, label="fp16")
    if qat_epochs:
        plt.plot(qat_epochs, qat_loss, linewidth=1.0, label="qat-w4a16")
    plt.title("YOLOv10m scratch: train loss (sum of train/*)")
    plt.xlabel("epoch")
    plt.ylabel("train_loss_total")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    run_root = args.run_root.resolve()

    fp16_summary = _read_json(run_root / "fp16" / "run_summary.json")
    qat_summary = _read_json(run_root / "qat-w4a16" / "run_summary.json")

    fp16_metrics_path = run_root / "eval" / "fp16" / "metrics.json"
    qat_metrics_path = run_root / "eval" / "qat-w4a16" / "metrics.json"
    fp16_metrics = _read_json(fp16_metrics_path) if fp16_metrics_path.is_file() else {}
    qat_metrics = _read_json(qat_metrics_path) if qat_metrics_path.is_file() else {}

    fp16_loss_csv = Path(fp16_summary["loss_artifacts"]["loss_curve_csv"])
    qat_loss_csv = Path(qat_summary["loss_artifacts"]["loss_curve_csv"])

    summary_dir = run_root / "summary"
    loss_png = summary_dir / "loss_curve_comparison.png"
    _plot_compare(fp16_csv=fp16_loss_csv, qat_csv=qat_loss_csv, out_png=loss_png)

    summary_md = summary_dir / "summary.md"
    lines: List[str] = []
    lines.append("# YOLOv10m scratch: FP16 vs W4A16 QAT (Brevitas)")
    lines.append("")
    lines.append(f"- Run root: `{run_root}`")
    lines.append(f"- Loss comparison: `{loss_png}`")
    lines.append("")
    lines.append("## FP16")
    lines.append("")
    lines.append(f"- Train dir: `{fp16_summary.get('save_dir')}`")
    lines.append(f"- ONNX: `{fp16_summary.get('onnx_path')}`")
    if fp16_metrics:
        lines.append(f"- Eval metrics: `{fp16_metrics_path}`")
        if "metrics" in fp16_metrics and isinstance(fp16_metrics["metrics"], dict):
            metrics = fp16_metrics["metrics"]
            lines.append(f"- mAP_50_95: `{metrics.get('mAP_50_95')}`")
            lines.append(f"- mAP_50: `{metrics.get('mAP_50')}`")
    lines.append("")
    lines.append("## QAT W4A16")
    lines.append("")
    lines.append(f"- Train dir: `{qat_summary.get('save_dir')}`")
    lines.append(f"- ONNX: `{qat_summary.get('onnx_path')}`")
    if qat_metrics:
        lines.append(f"- Eval metrics: `{qat_metrics_path}`")
        if "metrics" in qat_metrics and isinstance(qat_metrics["metrics"], dict):
            metrics = qat_metrics["metrics"]
            lines.append(f"- mAP_50_95: `{metrics.get('mAP_50_95')}`")
            lines.append(f"- mAP_50: `{metrics.get('mAP_50')}`")
    lines.append("")

    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_md.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

