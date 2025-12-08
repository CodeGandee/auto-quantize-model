# Subtask 1.10: Generalize INC sensitivity + quantization pipeline to other models

## Scope

Adapt the Qwen2.5-VL INC sensitivity and quantization pipeline into a reusable pattern for other models in this repo (e.g., ViT-like models, YOLO/ONNX flows). This involves identifying which parts are Qwen-specific and extracting general guidance and examples.

## Planned outputs

- An updated or new INC KB hint (e.g., a section in `howto-inc-layer-sensitivity-for-mixed-precision.md` or a sibling document) that:
  - Describes the general steps to run sensitivity analysis with INC.
  - Shows how to repurpose calibration/eval and profile design for non-LLM models.
- Optionally, a brief example or stub for a non-LLM model (e.g., ViT or YOLO) demonstrating how to plug into the same pattern.

## TODOs

- [ ] Job-001-110-001 Identify the Qwen-specific parts of the current pipeline (model loading, tokenizer, dataset choices) vs generic INC patterns.
- [ ] Job-001-110-002 Update `howto-inc-layer-sensitivity-for-mixed-precision.md` (or add a new hint) with a generalized step-by-step guide and references to the Qwen tasks/scripts.
- [ ] Job-001-110-003 (Optional) Sketch or implement a minimal INC sensitivity run for a non-LLM model in this repo to validate the generalization.
- [ ] Job-001-110-004 Ensure cross-links from relevant plans/tasks so future work can reuse this pipeline.

## Notes

- Keep the generalized guidance high-level enough to cover multiple architectures, but concrete enough that a developer can start from it without reverse-engineering the Qwen scripts.

