# Subtask 1.2: Calibration preprocessing and baseline COCO evaluation (ONNX)

## Scope

- Define and implement a reproducible **YOLO-style preprocessing** pipeline for the YOLOv10m ONNX model:
  - Letterbox resize to `640x640`
  - Color channel order and normalization consistent with the ONNX export
- Generate a calibration tensor `.npy` from `datasets/quantize-calib/quant100.txt` suitable for ModelOpt ONNX PTQ.
- Implement a baseline COCO2017 val evaluation path for the YOLOv10m ONNX output tensor (`[1, 144, 8400]`) and record mAP + simple latency stats.

## Planned outputs

- A calibration tensor builder script (new or refactored from existing YOLO11 tooling)
- A YOLOv10m ONNX COCO evaluator (new script or a validated Ultralytics-backed ONNX evaluation path)
- Baseline metrics JSON (and optional detections JSON) under `tmp/yolov10m_lowbit/<run-id>/baseline-coco/`

## TODOs

- [x] Job-001-102-001 Decide preprocessing details (RGB/BGR, scale, letterbox padding color) and make it shared between calibration and evaluation.
- [x] Job-001-102-002 Implement calibration tensor generation from `datasets/quantize-calib/quant100.txt` with output shape `[N, 3, 640, 640]` (float32).
- [x] Job-001-102-003 Implement YOLOv10m ONNX COCO evaluation:
  - Prefer reusing Ultralytics post-processing / evaluation if the ONNX format is compatible.
  - Otherwise, derive the decode/NMS logic by inspecting outputs on a few images and matching expected box/class semantics.
- [x] Job-001-102-004 Run a **medium** baseline eval (100 images) and save metrics + latency stats under `tmp/`.

## Implementation notes

- Shared preprocessing (letterbox 640, BGR→RGB, float32/255, box unletterbox mapping):
  - `src/auto_quantize_model/cv_models/yolo_preprocess.py`
- Calibration tensor builder:
  - `scripts/cv-models/make_yolov10m_calib_npy.py`
- COCO evaluator (DFL decode + class-aware NMS, ORT providers configurable):
  - `scripts/cv-models/eval_yolov10m_onnx_coco.py`

## Outputs (completed)

- GPU run root: `tmp/yolov10m_lowbit/2025-12-23_04-40-28_gpu/`
  - Calibration tensor: `tmp/yolov10m_lowbit/2025-12-23_04-40-28_gpu/calib/calib_yolov10m_640.npy`
  - Baseline COCO eval (original): `tmp/yolov10m_lowbit/2025-12-23_04-40-28_gpu/baseline-coco/metrics.json`
  - Baseline COCO eval (warmup + skip-latency): `tmp/yolov10m_lowbit/2025-12-23_04-40-28_gpu/results/baseline_cuda_only.json`

## Notes

- Run Python via `pixi run -e rtx5090 python ...` (see `context/instructions/prep-rtx5090.md`).
- The ONNX output name and shape are documented in `models/cv-models/yolov10m/README.md`.
