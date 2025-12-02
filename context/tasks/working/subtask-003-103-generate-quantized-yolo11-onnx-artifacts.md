# Subtask 3.3: Generate quantized YOLO11 ONNX artifacts

## Scope

Configure ModelOpt ONNX PTQ for INT8 and mixed-precision-friendly settings and produce one or more quantized YOLO11 ONNX models, ready for TensorRT engine building.

## Planned outputs

- Quantized YOLO11 ONNX models (e.g., `yolo11n-int8-qdq.onnx`, optionally variants for different settings).
- A record of the quantization configuration (precision mode, calibration method, exclusion lists, providers).
- Clear naming conventions for quantized ONNX files stored under `models/yolo11/onnx/`.

## TODOs

- [ ] Job-003-103-001: Decide on initial quantization settings for YOLO11 (e.g., `quantize_mode=int8`, calibration method, providers, ops/nodes to exclude) based on earlier ModelOpt research.
- [ ] Job-003-103-002: Run PTQ with the chosen settings to generate quantized ONNX artifacts for the primary variant(s) and save them under `models/yolo11/onnx/` with informative filenames.
- [ ] Job-003-103-003: Document the final PTQ configuration and produced artifacts (including paths and file names) so that TensorRT engine-building subtasks can consume them directly.

## Notes

- Keep configuration flexible so that later experiments (section 5) can vary key knobs without rewriting the pipeline from scratch.

