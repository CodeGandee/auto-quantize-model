# Subtask 1.3: INT8 ONNX PTQ baseline (ModelOpt) + validation

## Scope

- Add a reproducible wrapper around ModelOpt ONNX PTQ to produce an **INT8 Q/DQ ONNX** artifact for:
  - `models/cv-models/yolov10m/checkpoints/yolov10m.onnx`
- Use the calibration tensor produced in Subtask 1.2.
- Validate the quantized model via:
  - ORT inference (CUDA EP preferred)
  - COCO evaluation on the **medium** subset (100 images) (reuse the evaluator from Subtask 1.2)

## Planned outputs

- `scripts/cv-models/quantize_yolov10m_int8_onnx.sh` (ModelOpt ONNX PTQ wrapper)
- Quantized QDQ ONNX artifact under `tmp/yolov10m_lowbit/<run-id>/onnx/`
- PTQ logs/config snapshot under `tmp/.../quantize-int8/`
- COCO metrics JSON under `tmp/.../int8-coco/`

## TODOs

- [x] Job-001-103-001 Implement a YOLOv10m INT8 PTQ wrapper script (mirroring `scripts/yolo11/quantize_yolo11n_int8_onnx.sh`).
- [x] Job-001-103-002 Run ModelOpt ONNX PTQ (`quantize_mode=int8`, start with `calibration_method=max`) using ORT CUDA EP during calibration.
- [x] Job-001-103-003 Validate the output QDQ ONNX with ORT (smoke inference + provider selection).
- [x] Job-001-103-004 Run COCO evaluation on the **medium** subset (100 images) and record mAP + latency vs baseline.

## Outputs (completed)

- Wrapper script:
  - `scripts/cv-models/quantize_yolov10m_int8_onnx.sh`
- GPU run root: `tmp/yolov10m_lowbit/2025-12-23_04-40-28_gpu/`
  - INT8 QDQ ONNX: `tmp/yolov10m_lowbit/2025-12-23_04-40-28_gpu/onnx/yolov10m-int8-qdq.onnx`
  - PTQ logs: `tmp/yolov10m_lowbit/2025-12-23_04-40-28_gpu/quantize-int8/modelopt-onnx-ptq.log`
  - COCO eval (original): `tmp/yolov10m_lowbit/2025-12-23_04-40-28_gpu/int8-coco/metrics.json`
  - COCO eval (warmup + skip-latency): `tmp/yolov10m_lowbit/2025-12-23_04-40-28_gpu/results/int8_cuda_cpu.json`
- CPU-only earlier run root (shows max-calib collapse + entropy recovery): `tmp/yolov10m_lowbit/2025-12-22_19-44-58/`

## Notes

- Run Python via `pixi run -e rtx5090 python ...` (see `context/instructions/prep-rtx5090.md`).
- ModelOpt ONNX PTQ usage reference: `context/summaries/modelopt-kb/howto-modelopt-onnx-ptq-for-yolo11.md`
