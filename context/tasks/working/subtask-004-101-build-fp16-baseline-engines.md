# Subtask 4.1: Build FP16 baseline TensorRT engines

## Scope

Choose a TensorRT engine-building interface (`trtexec` and/or Python APIs), wire it through `pixi` as needed, and build FP16-only TensorRT engines from the baseline YOLO11 ONNX models to serve as comparison baselines.

## Planned outputs

- A documented choice of engine-building approach and any helper scripts or commands.
- FP16 TensorRT engines for the primary YOLO11 variant(s), stored in a predictable location (e.g., `models/yolo11/trt/`).
- Verification that the FP16 engines run end-to-end on the RTX 3090.

## TODOs

- [ ] Job-004-101-001: Decide whether to rely primarily on `trtexec`, Python TensorRT APIs, or a small wrapper script, and sketch how it will be invoked via `pixi run`.
- [ ] Job-004-101-002: Build FP16 TensorRT engines from the baseline YOLO11 ONNX models (starting with `yolo11n`) and store them under a consistent directory structure.
- [ ] Job-004-101-003: Sanity-check the FP16 engines by running a small number of inferences and confirming outputs are reasonable and consistent with the baseline evaluation.

## Notes

- These FP16 engines are the reference point for assessing INT8/mixed-precision benefits and accuracy changes.

