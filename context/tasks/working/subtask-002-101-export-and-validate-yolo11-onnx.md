# Subtask 2.1: Export and validate YOLO11 ONNX

## Scope

Export the selected YOLO11 checkpoints (at least `yolo11n`, optionally `yolo11s`) to ONNX using the existing helpers, and verify that the resulting ONNX graphs load cleanly and are suitable for downstream ModelOpt/TensorRT tooling.

## Planned outputs

- ONNX files for the chosen YOLO11 variants under `models/yolo11/onnx/`.
- Confirmation that the exported ONNX models load with `onnx`/`onnxruntime` and have expected input/output shapes.
- A short note or commands recorded in this repo describing how to regenerate and validate the ONNX exports.

## TODOs

- [x] Job-002-101-001: Run `pixi run python models/yolo11/helpers/convert_to_onnx.py <variant>` for the chosen YOLO11 variants (starting with `yolo11n`, then optionally `yolo11s`) and confirm the ONNX files appear under `models/yolo11/onnx/`.
- [x] Job-002-101-002: Use `onnx` and/or `onnxruntime` to load each exported ONNX model, run basic checks (e.g., shape inference or a dummy forward), and verify there are no immediate compatibility issues for TensorRT/ModelOpt.
- [x] Job-002-101-003: Record the export and validation commands (and any caveats) in this subtask file or a linked context note so they can be reused later without guessing.

## Notes

- Reuse the existing YOLO11 bootstrap and ONNX helper scripts documented in `models/yolo11/README.md`.

## Implementation summary

- Exported ONNX models for `yolo11n` and `yolo11s` using:
  - `pixi run python models/yolo11/helpers/convert_to_onnx.py yolo11n`
  - `pixi run python models/yolo11/helpers/convert_to_onnx.py yolo11s`
  which wrote `models/yolo11/onnx/yolo11n.onnx` and `models/yolo11/onnx/yolo11s.onnx`.
- Validated the ONNX graphs with `onnx`:
  - `pixi run python -c "import onnx; m=onnx.load('models/yolo11/onnx/yolo11n.onnx'); onnx.checker.check_model(m)"`
  - `pixi run python -c "import onnx; m=onnx.load('models/yolo11/onnx/yolo11s.onnx'); onnx.checker.check_model(m)"`
  confirming inputs `('images', [1, 3, 640, 640])` and outputs `('output0', [1, 84, 8400])` for both variants.
- Verified ONNX Runtime compatibility with CUDA and CPU providers via:
  - `pixi run python -c "import onnxruntime as ort; s=ort.InferenceSession('models/yolo11/onnx/yolo11n.onnx', providers=['CUDAExecutionProvider','CPUExecutionProvider']); print(s.get_inputs(), s.get_outputs())"`
  - Same for `yolo11s.onnx`, confirming expected tensor shapes and types; only benign ONNX Runtime warnings about added memcpy nodes were observed.

