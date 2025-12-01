# Subtask 1.1: Study ModelOpt docs and ONNX/CNN usage

## Scope

Understand how NVIDIA ModelOpt (TensorRT Model Optimizer) supports ONNX and CNN-style models, with an emphasis on the quantization workflows and options that will be relevant for applying mixed FP16/INT8 quantization to YOLO11 in the **auto-quantize-model** project.

## Planned outputs

- A short written summary of the recommended ONNX quantization flow using ModelOpt (CLI and/or Python APIs).
- Notes on any special considerations for CNNs or object detection models (e.g., calibration data requirements, Q/DQ insertion, supported ops).
- A list of key CLI flags, config options, or example scripts that we expect to reuse for YOLO11 quantization.
- Links or references to the most relevant documents and example files in `extern/TensorRT-Model-Optimizer`.

## TODOs

- [ ] Job-001-101-001: Read the main `extern/TensorRT-Model-Optimizer/README.md` to understand the overall scope and techniques offered by ModelOpt.
- [ ] Job-001-101-002: Review ModelOpt documentation for ONNX quantization and CNN/object-detection-relevant examples (e.g., `examples/onnx_ptq` and associated docs).
- [ ] Job-001-101-003: Identify and record the primary CLI and/or Python entrypoints we will likely use for ONNX quantization (including any calibration-related flags).
- [ ] Job-001-101-004: Capture any constraints or recommendations from TensorRT “Best Practices” (especially around ONNX Q/DQ models and mixed precision) that affect how we quantize YOLO11.
- [ ] Job-001-101-005: Summarize the findings in a brief note and link it from the main task file or a dedicated context doc so later milestones can reference it.

## Notes

- Prioritize up-to-date docs (ModelOpt’s official site and the current `extern/TensorRT-Model-Optimizer` checkout) to avoid stale guidance.
- Focus on ONNX/PTQ material; LLM-specific content can be skimmed unless it reveals general quantization patterns that also apply to CNNs.

