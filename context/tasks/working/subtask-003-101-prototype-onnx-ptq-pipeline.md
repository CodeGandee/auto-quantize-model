# Subtask 3.1: Prototype ONNX PTQ pipeline for YOLO11

## Scope

Stand up a minimal ModelOpt ONNX PTQ pipeline for YOLO11 using the exported ONNX model(s), with placeholder or random calibration data, to validate that the end-to-end quantization flow functions correctly.

## Planned outputs

- A working CLI or Python invocation of `modelopt.onnx.quantization` on a YOLO11 ONNX file.
- Confirmation that the quantization step completes without errors and produces an INT8/QDQ ONNX model.
- Notes on any required flags, providers, or environment setup discovered while prototyping.

## TODOs

- [x] Job-003-101-001: Using a YOLO11 ONNX export (e.g., `models/yolo11/onnx/yolo11n.onnx`), run a minimal `python -m modelopt.onnx.quantization` command with simple/random calibration data to confirm the PTQ pipeline runs.
- [x] Job-003-101-002: Inspect the generated quantized ONNX model (e.g., `*-int8-qdq.onnx`) to verify that Q/DQ nodes are present and the model still loads with `onnx`/`onnxruntime`.
- [x] Job-003-101-003: Capture the working prototype command (and any required env vars or provider settings) in this subtask file for later refinement.

## Notes

- This prototype can use synthetic or minimal calibration data; real calibration integration is handled in Subtask 3.2.

## Implementation summary

- Installed the ONNX extras required by NVIDIA ModelOpt for ONNX PTQ by adding the missing `onnx_graphsurgeon` dependency to the `pixi` environment:
  - `pixi run python -m pip install onnx_graphsurgeon`
  - Verified that `modelopt.onnx` and `modelopt.onnx.quantization` now import successfully.
- Ran a prototype INT8 ONNX PTQ pipeline on the YOLO11n ONNX export using the previously prepared calibration tensor:
  - Command:
    - `pixi run python -m modelopt.onnx.quantization --onnx_path=models/yolo11/onnx/yolo11n.onnx --quantize_mode=int8 --calibration_data=datasets/quantize-calib/calib_yolo11_640.npy --calibration_method=max --output_path=models/yolo11/onnx/yolo11n-int8-qdq-proto.onnx --calibration_eps "cuda:0 cpu"`
  - ModelOpt logs confirm preprocessing, calibration, and INT8 quantization completed successfully and produced `models/yolo11/onnx/yolo11n-int8-qdq-proto.onnx`.
- Validated the prototype quantized model with `onnx`:
  - Loaded `models/yolo11/onnx/yolo11n-int8-qdq-proto.onnx` and ran `onnx.checker.check_model`, confirming a valid graph.
  - Verified that graph inputs/outputs remain `images: [1, 3, 640, 640]` and `output0: [1, 84, 8400]`, and counted 378 `QuantizeLinear`/`DequantizeLinear` nodes, confirming that explicit Q/DQ insertion occurred.
- This establishes a working baseline CLI invocation for ModelOpt ONNX PTQ on YOLO11; Subtask 3.2 will refine this to use the calibration dataset more systematically and tune provider settings if needed.
