# Subtask 1.4: PTQ W4A8 (INT8 activations) export + validation

## Scope

- Extend the PTQ pipeline to support **W4A8 (INT8 activations)**:
  - 4-bit weight quantization (same W4 approach as Subtask 1.3),
  - INT8 activation quantizers at stable cut points (e.g., post-conv / post-activation in blocks),
  - a calibration pass over `datasets/quantize-calib/quant100.txt` to set activation quantization parameters,
  - export QCDQ ONNX (`yolov10m-w4a8-qcdq-ptq.onnx`),
  - validate ORT CUDA EP inference and COCO subset metrics.

## Planned outputs

- W4A8 PTQ implementation integrated into `scripts/cv-models/quantize_yolov10m_brevitas_w4.py`.
- ONNX artifact under `tmp/yolov10m_brevitas_w4a8_w4a16/<run-id>/onnx/`:
  - `yolov10m-w4a8-qcdq-ptq.onnx`
- Calibration logs/config snapshot and a COCO subset metrics JSON for the PTQ W4A8 model.

## Dataset plan

- **Calibration (activations)**:
  - Default calibration set: `datasets/quantize-calib/quant100.txt` (100 COCO2017 train2017 images; repo-relative paths).
  - Calibration does not require labels; we use the images only to set activation quantizer parameters.
  - If PTQ quality is unstable, scale up to a larger calibration set (e.g., 500–2000 images) by sampling from `datasets/coco2017/source-data/train2017/` with a fixed seed and writing the list under the run root (do not commit).
- **Evaluation (must match baseline)**:
  - Use the same fixed COCO2017 val subset plan as Subtask 1.1 so baseline/W4A16/W4A8 are directly comparable.

## TODOs

- [ ] Job-001-104-001 Choose and document activation quantizer insertion points for YOLOv10m (keep it minimal at first).
- [ ] Job-001-104-002 Implement a calibration dataloader from `datasets/quantize-calib/quant100.txt` using YOLO-style preprocessing (match `scripts/cv-models/make_yolov10m_calib_npy.py`).
- [ ] Job-001-104-003 Run calibration for the W4A8 quantized model and record calibration settings (N, seed, preprocessing).
- [ ] Job-001-104-004 Export QCDQ ONNX for W4A8 PTQ and validate it runs in ORT CUDA EP.
- [ ] Job-001-104-005 Run COCO subset evaluation for W4A8 PTQ and compare vs baseline and W4A16(-like) PTQ.

## Notes

- W4A8 here means **INT8 activations via Q/DQ**, not FP8.
