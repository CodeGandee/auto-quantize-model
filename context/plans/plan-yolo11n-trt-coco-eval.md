Plan: YOLO11n TensorRT INT8 / FP16 / mixed evaluation on COCO2017

## HEADER
- **Purpose**: Quantize YOLO11n ONNX with ModelOpt using COCO2017-based calibration, build TensorRT engines (FP16, pure INT8, INT8+FP16), and compare mAP and throughput on COCO2017 val via TensorRT inference.
- **Status**: Draft / in progress
- **Date**: 2025-12-04
- **Owner**: AI assistant (Codex CLI)
- **Scope**: YOLO11n only, COCO2017 val2017 evaluation, batch size 1, single-GPU TensorRT.

## TODO

1. **Dataset inspection and calibration sampling**
   - Confirm COCO2017 layout under `datasets/coco2017/source-data` (images and `instances_val2017.json`).
   - Decide on image preprocessing to match the existing YOLO11n ONNX export (letterbox, resize to 640x640, normalization).
   - Randomly sample ~200 images from `val2017` and build a calibration tensor `[N, C, H, W]` with proper preprocessing.
   - Save calibration artifacts under `tmp/yolo11-trt/calib/` (e.g., `yolo11n-coco-calib.npy`) for reuse.

2. **ModelOpt ONNX quantization (COCO-based)**
   - Starting from `models/yolo11/onnx/yolo11n.onnx`, run ModelOpt ONNX PTQ with the COCO calibration tensor:
     - Produce a **pure INT8 QDQ ONNX** (full quantization, or as aggressive as possible): `yolo11n-int8-coco-qdq.onnx`.
     - Produce a **mixed INT8+FP16-friendly QDQ ONNX** if needed (e.g., by excluding certain ops/nodes) to compare against the more aggressive INT8 version.
   - Save all intermediate QDQ ONNX models under `models/yolo11/onnx/` (consistent naming) and log quantization settings in `tmp/yolo11-trt/`.

3. **TensorRT engine builds**
   - Ensure we have the FP16 baseline engine (or recreate if needed): `yolo11n-fp16.plan` from `yolo11n.onnx`.
   - Build:
     - **INT8+FP16 mixed engine** from the QDQ ONNX designed for mixed precision.
     - **Pure INT8 engine** from the fully quantized QDQ ONNX.
   - Use consistent `trtexec` or TensorRT Python API flags to ensure explicit quantization and that Q/DQ operators are fused:
     - `--int8 --fp16 --best` for mixed.
     - `--int8 --best` for pure INT8.
   - Optionally export layer info JSONs with `--exportLayerInfo` for inspection (stored under `tmp/yolo11-trt/`).

4. **TensorRT COCO2017 val inference harness**
   - Implement a TRT inference script (e.g., `tmp/yolo11-trt/eval_trt_yolo11n_coco.py`) that:
     - Loads a given `.plan` engine.
     - Iterates over COCO2017 `val2017` images with YOLO11-compatible preprocessing.
     - Runs inference via TensorRT, decodes YOLO outputs into bounding boxes, class IDs, and scores using Ultralytics YOLO11 utilities where possible.
     - Records detections and evaluation timing (end-to-end latency or images/sec).
   - Integrate COCO mAP computation:
     - Use `pycocotools` or Ultralytics’ COCO evaluation helper to compute standard COCO metrics (e.g., AP@[.5:.95]).
     - Ensure annotations from `instances_val2017.json` are correctly wired.

5. **Experiments and comparison**
   - Run full-val evaluations for:
     - FP16 engine.
     - INT8+FP16 mixed engine.
     - Pure INT8 engine.
   - For each engine, capture:
     - COCO mAP metrics (at least AP@[.5:.95]).
     - Average throughput (images/sec) and per-image latency (ms) at batch size 1.
   - Summarize results in a short table and narrative under `tmp/yolo11-trt/results-yolo11n-trt-coco.md` (or similar), highlighting:
     - Speedup vs FP16 baseline.
     - mAP degradation (if any) vs FP16 baseline.

6. **Follow-ups / optional hardening**
   - If INT8 accuracy is unacceptably degraded, iterate on ModelOpt configuration (e.g., exclude some ops, tweak calibration method).
   - If TRT engines show unexpected FP32/FP16 fallbacks, inspect layer info JSON and adjust QDQ placement or TRT flags.
   - If the approach is stable and useful, promote scripts from `tmp/` into `models/yolo11/helpers/` with proper docs and tests.

