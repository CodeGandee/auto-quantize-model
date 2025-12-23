# Subtask 1.8: PTQ W8A8 (INT8 weights + INT8 activations) export + validation

## Scope

- Add a **PTQ W8A8** path as a second **sanity check**:
  - load `models/yolo10/checkpoints/yolov10m.pt`,
  - quantize weights to **INT8**,
  - add **INT8 activation quantizers** and run a calibration pass (images-only) using `datasets/quantize-calib/quant100.txt`,
  - export QCDQ ONNX (`yolov10m-w8a8-qcdq-ptq.onnx`),
  - validate ORT inference with CUDA EP preferred,
  - evaluate on the same COCO subset as baseline.
- Expectation: with correct calibration and a correct eval contract, **W8A8 PTQ should not have a large accuracy drop** vs baseline on the fixed 100-image COCO subset.

## Planned outputs

- Implementation:
  - Extend `scripts/cv-models/quantize_yolov10m_brevitas_w4.py` to accept `--mode w8a8`.
  - Extend `src/auto_quantize_model/cv_models/yolov10_brevitas.py` to support `weight_bit_width=8` + `act_bit_width=8`.
- ONNX artifacts under `tmp/yolov10m_brevitas_w4a8_w4a16/<run-id>/onnx/`:
  - `yolov10m-w8a8-qcdq-ptq.onnx`
  - `yolov10m-w8a8-qcdq-ptq-opt.onnx` (optional)
- Calibration logs/config snapshot and COCO subset metrics JSON under the same run root.

## Dataset plan

- **Calibration (activations)**:
  - Default: `datasets/quantize-calib/quant100.txt` (100 COCO train2017 images; images-only).
  - If results are noisy, scale calibration up under the run root (do not commit).
- **Evaluation (must match baseline)**:
  - Same fixed COCO2017 val subset used for baseline/W4 variants.

## TODOs

- [ ] Job-001-108-001 Add `w8a8` mode plumbing in `scripts/cv-models/quantize_yolov10m_brevitas_w4.py`.
- [ ] Job-001-108-002 Add an INT8 weight quantizer option in `quantize_model_brevitas_ptq`.
- [ ] Job-001-108-003 Reuse `calibrate_activation_quantizers` on `datasets/quantize-calib/quant100.txt`.
- [ ] Job-001-108-004 Export `yolov10m-w8a8-qcdq-ptq.onnx` and validate ORT CUDA EP smoke inference.
- [ ] Job-001-108-005 Run COCO subset evaluation and compare against baseline (expect only modest drop).

## Notes

- This subtask is meant to validate our quantization/cali/eval wiring. If W8A8 collapses, fix pipeline issues before interpreting any W4 results.

## Summary

- Implemented PTQ **W8A8** sanity-check export:
  - Added `w8a8` mode to `scripts/cv-models/quantize_yolov10m_brevitas_w4.py`.
  - Reused the same activation calibration flow as W4A8 (INT8 activations via Q/DQ).
- Tested run root: `tmp/yolov10m_brevitas_w4a8_w4a16/2025-12-23_16-12-40`
  - Calibration list: `datasets/quantize-calib/quant100.txt` (100 images)
  - ONNX: `tmp/yolov10m_brevitas_w4a8_w4a16/2025-12-23_16-12-40/onnx/yolov10m-w8a8-qcdq-ptq-opt.onnx`
  - COCO subset (100 images) metrics: `mAP_50_95=0.5932`, `mAP_50=0.7696` (baseline `mAP_50_95=0.6022`)
    - Metrics JSON: `tmp/yolov10m_brevitas_w4a8_w4a16/2025-12-23_16-12-40/ptq-w8a8-coco/metrics.json`
  - Export config snapshot: `tmp/yolov10m_brevitas_w4a8_w4a16/2025-12-23_16-12-40/ptq_w8a8_export.json`
- Logs:
  - Export: `context/logs/quantize-yolov10m-w4a8-w4a16-brevitas/subtask-001-108-ptq-w8a8-export.log`
  - Smoke: `context/logs/quantize-yolov10m-w4a8-w4a16-brevitas/subtask-001-108-ptq-w8a8-smoke.log`
  - Eval: `context/logs/quantize-yolov10m-w4a8-w4a16-brevitas/subtask-001-108-ptq-w8a8-eval.log`
