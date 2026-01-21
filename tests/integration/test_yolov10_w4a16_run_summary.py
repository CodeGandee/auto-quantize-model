from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

from auto_quantize_model.cv_models.yolov10_w4a16_validation import write_run_summary_json


def test_run_summary_json_has_required_keys() -> None:
    out_dir = Path("tmp") / "test-yolov10-w4a16-validation" / uuid4().hex
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        out_path = out_dir / "run_summary.json"
        payload = {
            "schema_version": "1.0",
            "run_id": "2025-12-25_yolo10n_baseline_seed0",
            "created_at": "2025-12-25T00:00:00Z",
            "git": {"commit": "deadbeef", "branch": "main", "dirty": False},
            "model": {"variant": "yolo10n", "checkpoint": "models/yolo10/checkpoints/yolov10n.pt", "imgsz": 640},
            "quantization": {"mode": "w4a16", "library": "brevitas", "weight_bit_width": 4, "activation_bit_width": None},
            "method": {"variant": "baseline", "ema": {"enabled": False}, "qc": {"enabled": False}},
            "dataset": {"coco_root": "datasets/coco2017/source-data", "dataset_yaml": "tmp/dataset.yaml", "train_images": 2, "val_images": 1},
            "training": {"framework": "ultralytics", "epochs": 1, "batch": 16, "seed": 0, "device": "0", "amp": True},
            "artifacts": {"run_root": str(out_dir)},
            "metrics": {"primary_name": "metrics/mAP50-95(B)", "best_value": 0.3, "final_value": 0.2},
            "stability": {"collapse_threshold_ratio": 0.5, "final_over_best_ratio": 0.6667, "collapsed": False},
            "status": "completed-success",
        }
        write_run_summary_json(out_path=out_path, payload=payload)

        loaded = json.loads(out_path.read_text(encoding="utf-8"))
        for key in (
            "schema_version",
            "run_id",
            "created_at",
            "git",
            "model",
            "quantization",
            "method",
            "dataset",
            "training",
            "artifacts",
            "metrics",
            "stability",
            "status",
        ):
            assert key in loaded
        assert set(loaded["git"].keys()) >= {"commit", "branch", "dirty"}
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)

