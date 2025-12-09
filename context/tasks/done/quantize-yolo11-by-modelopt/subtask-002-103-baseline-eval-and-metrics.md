# Subtask 2.3: Implement baseline evaluation and metrics

## Scope

Implement or adapt an evaluation script for YOLO11 that can run FP32/FP16 inference (PyTorch, ONNXRuntime, or TensorRT) on a validation subset and compute baseline accuracy and latency metrics on the RTX 3090.

## Planned outputs

- A reusable evaluation script or entrypoint (preferably wired through `pixi run`) for running YOLO11 inference on a validation dataset.
- Baseline FP32/FP16 accuracy metrics (e.g., mAP@[0.5:0.95]) for the chosen variants and resolution.
- Baseline latency and throughput measurements for the selected variants on the RTX 3090.

## TODOs

- [x] Job-002-103-001: Implement or adapt a YOLO11 evaluation script that can run inference on the chosen validation subset and compute mAP/precision/recall metrics.
- [x] Job-002-103-002: Run the evaluation in FP32/FP16 mode for `yolo11n` (and optionally `yolo11s`) on the RTX 3090 at 640×640 resolution and record the resulting accuracy metrics.
- [x] Job-002-103-003: Measure latency and throughput (e.g., median per-image latency and images/sec) for the baseline setup and store the results in a simple, structured format (e.g., JSON/CSV or a Markdown table).

## Notes

- This subtask provides the baseline metrics that later TensorRT FP16/INT8 engines will be compared against.

## Implementation summary

- Implemented a PyTorch-based COCO evaluation script under `scripts/yolo11/eval_yolo11_torch_coco.py` that:
  - Loads a YOLO11 checkpoint via the local Ultralytics source (`models/yolo11/src`) to stay in sync with this repo.
  - Uses `pycocotools` with `annotations/instances_val2017.json` to compute COCO bbox mAP metrics.
  - Runs single-image inference over a configurable number of COCO 2017 val images, collecting detections and per-image latency.
  - Summarizes latency (mean/median/p90 in ms and throughput in FPS) and optionally writes all metrics to a JSON file.
- Prepared COCO2017 annotations for evaluation by extracting `annotations/instances_val2017.json` from `annotations_trainval2017.zip` under the real dataset root (`datasets/coco2017/source-data` -> `/data2/datasets/coco2017`).
- Ran a baseline FP16 evaluation for `yolo11n` on 100 COCO val images at 640×640, using GPU device 0:
  - Command:
    - `pixi run python scripts/yolo11/eval_yolo11_torch_coco.py --model models/yolo11/checkpoints/yolo11n.pt --data-root datasets/coco2017/source-data --max-images 100 --device 0 --precision fp16 --imgsz 640 --out datasets/quantize-calib/baseline_yolo11n_fp16_coco100.json`
  - Key results (COCO bbox, 100-image subset):
    - `mAP_50_95 ≈ 0.476`, `mAP_50 ≈ 0.631`, `mAP_75 ≈ 0.524`.
  - Latency stats (per image, end-to-end with Ultralytics preprocessing/postprocessing on RTX 3090):
    - Mean ≈ 106.7 ms, median ≈ 69.1 ms, p90 ≈ 193.0 ms, throughput ≈ 9.4 FPS over 100 images.
  - Metrics JSON stored at `datasets/quantize-calib/baseline_yolo11n_fp16_coco100.json` for reuse by later subtasks.
