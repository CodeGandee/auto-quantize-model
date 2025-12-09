# Subtask 4.2: Build mixed FP16/INT8 engines and integrate with evaluation

## Scope

Build mixed FP16/INT8 TensorRT engines from the quantized YOLO11 ONNX models and extend the evaluation script to run both FP16 and mixed engines, enabling side-by-side comparison.

## Planned outputs

- Mixed FP16/INT8 TensorRT engines built from the quantized ONNX artifacts.
- An evaluation script or entrypoint that can switch between FP16 and mixed engines on the same dataset and settings.
- Initial accuracy and latency comparisons between FP16 and mixed engines.

## TODOs

- [ ] Job-004-102-001: Build mixed FP16/INT8 TensorRT engines from the quantized YOLO11 ONNX models produced in Subtask 3.3, enabling both precisions during engine build.
- [ ] Job-004-102-002: Update or extend the evaluation script to support running inference with either FP16 or mixed engines, using the same inputs and metrics.
- [ ] Job-004-102-003: Run initial evaluations comparing FP16 vs mixed engines and note any obvious accuracy or stability issues.

## Notes

- Ensure engine build flags align with ModelOpt’s Q/DQ placement (e.g., allowing TensorRT to use INT8 where Q/DQ nodes exist while keeping sensitive ops in FP16).

