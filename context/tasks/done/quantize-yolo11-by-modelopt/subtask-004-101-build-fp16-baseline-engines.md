# Subtask 4.1: Build FP16 baseline TensorRT engines

## Scope

Choose a TensorRT engine-building interface (`trtexec` and/or Python APIs), wire it through `pixi` as needed, and build FP16-only TensorRT engines from the baseline YOLO11 ONNX models to serve as comparison baselines.

## Planned outputs

- A documented choice of engine-building approach and any helper scripts or commands.
- FP16 TensorRT engines for the primary YOLO11 variant(s), stored in a predictable location (e.g., `models/yolo11/trt/`).
- Verification that the FP16 engines run end-to-end on the RTX 3090.

## TODOs

- [x] Job-004-101-001: Decide whether to rely primarily on `trtexec`, Python TensorRT APIs, or a small wrapper script, and sketch how it will be invoked via `pixi run`.
- [x] Job-004-101-002: Build FP16 TensorRT engines from the baseline YOLO11 ONNX models (starting with `yolo11n`) and store them under a consistent directory structure.
- [x] Job-004-101-003: Sanity-check the FP16 engines by running a small number of inferences and confirming outputs are reasonable and consistent with the baseline evaluation.

## Notes

- These FP16 engines are the reference point for assessing INT8/mixed-precision benefits and accuracy changes.

## Implementation summary

- Chosen interface: use the TensorRT `trtexec` CLI, invoked via Pixi, as the primary path for building FP16 baseline engines. This matches the patterns documented in the ModelOpt KB (`howto-qdq-onnx-to-mixed-precision-tensorrt.md`) and keeps engine builds reproducible from the command line.
- Directory layout: created (or reused) `models/yolo11/trt/` as the canonical location for YOLO11 TensorRT engines, keeping ONNX exports under `models/yolo11/onnx/` and engines under `models/yolo11/trt/`.
- FP16 baseline build for YOLO11n:
  - Starting from the existing ONNX export `models/yolo11/onnx/yolo11n.onnx` (from Subtask 2.1), built a pure FP16 engine with:
    - `pixi run trtexec --onnx=models/yolo11/onnx/yolo11n.onnx --saveEngine=models/yolo11/trt/yolo11n-fp16.plan --fp16`
  - This command was run in the TensorRT-enabled environment on an RTX 3090, producing `models/yolo11/trt/yolo11n-fp16.plan` as the FP16 baseline engine.
- Sanity checks:
  - Used `trtexec`’s built-in benchmarking to verify the engine runs end-to-end and exercises the expected input shape:
    - `pixi run trtexec --loadEngine=models/yolo11/trt/yolo11n-fp16.plan --shapes=images:1x3x640x640 --iterations=100 --avgRuns=10`
  - Compared basic detection outputs (object counts and representative boxes/scores on a small COCO2017 image subset) against the ONNX baseline using the existing YOLO11 evaluation helpers, confirming that FP16 predictions are numerically close and suitable as a reference for later INT8/mixed-precision experiments.
