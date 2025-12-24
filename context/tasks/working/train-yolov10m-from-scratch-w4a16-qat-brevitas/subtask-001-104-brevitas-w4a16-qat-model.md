# Subtask 1.4: Brevitas W4A16 model builder (scratch)

## Scope

- Build YOLOv10m from YAML (random init) and apply Brevitas quant layers:
  - Quantize **Conv2d weights to int4 fake-quant** (W4),
  - Keep activations floating (A16 / no activation quantizers),
  - Keep model output contract compatible with the repo’s ONNX evaluator.
- Support a minimal quant scheme configuration in code and record it in artifacts.

## Planned outputs

- A helper to create a quantized YOLOv10m model suitable for training (QAT) and export:
  - `Conv2d -> QuantConv2d` via Brevitas `layerwise_quantize`
  - configurable module-name blacklist for layers to skip (if needed)

## TODOs

- [ ] Job-001-104-001 Add a QAT-friendly quantization helper in `src/auto_quantize_model/cv_models/yolov10_brevitas.py`.
- [ ] Job-001-104-002 Validate a forward pass and ensure loss computation runs under Ultralytics training loop.
- [ ] Job-001-104-003 Decide/implement any first/last layer exceptions (optional, only if needed for stability).

## Notes

- Keep the initial implementation weight-only (no activation quant).

