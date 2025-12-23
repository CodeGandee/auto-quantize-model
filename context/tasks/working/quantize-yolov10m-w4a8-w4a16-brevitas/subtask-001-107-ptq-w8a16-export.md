# Subtask 1.7: PTQ W8A16 (INT8 weights, float activations) export + validation

## Scope

- Add a **PTQ W8A16** path as a **sanity check** for the quantization/export/eval pipeline:
  - load `models/yolo10/checkpoints/yolov10m.pt`,
  - quantize weights to **INT8** (prefer per-channel if available),
  - leave activations floating (FP16/FP32 compute),
  - export QCDQ ONNX (`yolov10m-w8a16-qcdq-ptq.onnx`),
  - validate ORT inference with CUDA EP preferred,
  - evaluate on the same COCO subset as baseline.
- Expectation: if the quantization framework and evaluation tooling are wired correctly, **W8A16 PTQ should not have a large accuracy drop** vs baseline on the fixed 100-image COCO subset. If it does, treat as evidence that the pipeline is wrong (not that INT8 is inherently bad).

## Planned outputs

- Implementation:
  - Extend `scripts/cv-models/quantize_yolov10m_brevitas_w4.py` to accept `--mode w8a16`.
  - Extend `src/auto_quantize_model/cv_models/yolov10_brevitas.py` to support `weight_bit_width=8`.
- ONNX artifacts under `tmp/yolov10m_brevitas_w4a8_w4a16/<run-id>/onnx/`:
  - `yolov10m-w8a16-qcdq-ptq.onnx`
  - `yolov10m-w8a16-qcdq-ptq-opt.onnx` (optional)
- Smoke inference logs and COCO subset metrics JSON under the same run root.

## Dataset plan

- **Calibration**: none required for W8A16 (activations remain floating).
- **Evaluation (must match baseline)**:
  - COCO2017 val subset via `scripts/cv-models/eval_yolov10m_onnx_coco.py`:
    - `datasets/coco2017/source-data/val2017/` + `instances_val2017.json`
    - `--max-images 100` (deterministic first-N IDs) or a shared `--image-ids-list` under the run root.

## TODOs

- [ ] Job-001-107-001 Add `w8a16` mode plumbing in `scripts/cv-models/quantize_yolov10m_brevitas_w4.py`.
- [ ] Job-001-107-002 Add an INT8 weight quantizer option in `quantize_model_brevitas_ptq` (prefer per-channel if supported by Brevitas).
- [ ] Job-001-107-003 Export `yolov10m-w8a16-qcdq-ptq.onnx` and validate ORT CUDA EP smoke inference.
- [ ] Job-001-107-004 Run COCO subset evaluation and compare against baseline (expect only modest drop).

## Notes

- This subtask is primarily a **pipeline correctness check**. If W8A16 collapses, fix pipeline issues (quant insertion points, quantizer config, calibration expectations, evaluator I/O contract) before iterating further on W4.

## Summary

- Implemented PTQ **W8A16** sanity-check export:
  - Added `w8a16` mode to `scripts/cv-models/quantize_yolov10m_brevitas_w4.py`.
  - Extended `quantize_model_brevitas_ptq(..., weight_bit_width=8)` in `src/auto_quantize_model/cv_models/yolov10_brevitas.py` (INT8 weights, per-channel).
- Tested run root: `tmp/yolov10m_brevitas_w4a8_w4a16/2025-12-23_16-12-40`
  - ONNX: `tmp/yolov10m_brevitas_w4a8_w4a16/2025-12-23_16-12-40/onnx/yolov10m-w8a16-qcdq-ptq-opt.onnx`
  - COCO subset (100 images) metrics: `mAP_50_95=0.5983`, `mAP_50=0.7697` (baseline `mAP_50_95=0.6022`)
    - Metrics JSON: `tmp/yolov10m_brevitas_w4a8_w4a16/2025-12-23_16-12-40/ptq-w8a16-coco/metrics.json`
  - Export config snapshot: `tmp/yolov10m_brevitas_w4a8_w4a16/2025-12-23_16-12-40/ptq_w8a16_export.json`
- Caveat: exporting with FP16 input on CUDA failed (`QuantTensor is not valid`) for W8A16; export succeeds with CPU FP32 input via `--no-export-fp16-input`.
- Logs:
  - Failed CUDA FP16 export: `context/logs/quantize-yolov10m-w4a8-w4a16-brevitas/subtask-001-107-ptq-w8a16-export.log`
  - Successful CPU export: `context/logs/quantize-yolov10m-w4a8-w4a16-brevitas/subtask-001-107-ptq-w8a16-export-v2.log`
  - Smoke: `context/logs/quantize-yolov10m-w4a8-w4a16-brevitas/subtask-001-107-ptq-w8a16-smoke.log`
  - Eval: `context/logs/quantize-yolov10m-w4a8-w4a16-brevitas/subtask-001-107-ptq-w8a16-eval.log`
