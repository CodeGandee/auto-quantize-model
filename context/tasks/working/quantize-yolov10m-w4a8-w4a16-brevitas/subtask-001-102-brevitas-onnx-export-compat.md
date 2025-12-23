# Subtask 1.2: Brevitas QCDQ ONNX export compatibility (Torch 2.9)

## Scope

- Implement a reusable helper that makes Brevitas `export_onnx_qcdq(..., dynamo=False)` work with Torch 2.9 in the `rtx5090` Pixi environment.
- Avoid ad-hoc monkeypatching in notebooks; the compatibility logic should live in `src/auto_quantize_model/`.
- Add a minimal smoke script that exercises the helper in `pixi run -e rtx5090` and produces a small QCDQ ONNX file.

## Planned outputs

- `src/auto_quantize_model/brevitas_onnx_export_compat.py`
  - A small API (e.g., `apply_brevitas_torch_onnx_compat()`) that patches Brevitas opset getter / internals as needed.
- A tiny smoke script (under `scripts/` or `tests/manual/`) that:
  - constructs a toy Brevitas quant module,
  - exports QCDQ ONNX with the helper applied,
  - validates the ONNX loads with `onnx` and runs a single ORT inference.

## TODOs

- [ ] Job-001-102-001 Reproduce the Brevitas export failure (if any) under Torch 2.9 and capture the stack trace / missing symbol.
- [ ] Job-001-102-002 Implement `src/auto_quantize_model/brevitas_onnx_export_compat.py` based on `context/hints/about-brevitas-yolo-w4a8-w4a16-onnx-nvidia-gpu.md`.
- [ ] Job-001-102-003 Add and run a minimal smoke script that exports a toy QCDQ ONNX in `pixi run -e rtx5090`.
- [ ] Job-001-102-004 Verify the helper is usable by the YOLOv10m Brevitas export pipeline (Subtasks 1.3–1.5).

## Notes

- Prefer the smallest possible compatibility shim that is easy to delete once Brevitas supports Torch 2.9 natively.

