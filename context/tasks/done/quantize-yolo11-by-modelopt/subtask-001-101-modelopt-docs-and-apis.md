# Subtask 1.1: Study ModelOpt docs and ONNX/CNN usage

## Scope

Understand how NVIDIA ModelOpt (TensorRT Model Optimizer) supports ONNX and CNN-style models, with an emphasis on the quantization workflows and options that will be relevant for applying mixed FP16/INT8 quantization to YOLO11 in the **auto-quantize-model** project.

## Planned outputs

- A short written summary of the recommended ONNX quantization flow using ModelOpt (CLI and/or Python APIs).
- Notes on any special considerations for CNNs or object detection models (e.g., calibration data requirements, Q/DQ insertion, supported ops).
- A list of key CLI flags, config options, or example scripts that we expect to reuse for YOLO11 quantization.
- Links or references to the most relevant documents and example files in `extern/TensorRT-Model-Optimizer`.

## TODOs

- [x] Job-001-101-001: Read the main `extern/TensorRT-Model-Optimizer/README.md` to understand the overall scope and techniques offered by ModelOpt.
- [x] Job-001-101-002: Review ModelOpt documentation for ONNX quantization and CNN/object-detection-relevant examples (e.g., `examples/onnx_ptq` and associated docs).
- [x] Job-001-101-003: Identify and record the primary CLI and/or Python entrypoints we will likely use for ONNX quantization (including any calibration-related flags).
- [x] Job-001-101-004: Capture any constraints or recommendations from TensorRT “Best Practices” (especially around ONNX Q/DQ models and mixed precision) that affect how we quantize YOLO11.
- [x] Job-001-101-005: Summarize the findings in a brief note and link it from the main task file or a dedicated context doc so later milestones can reference it.

## Summaries and knowledge base references

Detailed findings for this subtask have been moved into the shared ModelOpt
knowledge base under `context/summaries/modelopt-kb`. The sections below
summarize each document and provide links for follow-up work.

### Howto: ModelOpt ONNX PTQ for YOLO11

- Focus: end-to-end, practical recipe for using NVIDIA ModelOpt’s ONNX PTQ
  tooling on YOLO11-style detectors (exported ONNX models), including
  calibration data preparation, CLI/Python entrypoints, and integration with
  TensorRT for mixed-precision deployment.
- Use when you need concrete commands or code snippets for quantizing a YOLO11
  ONNX model to INT8/QDQ and building TensorRT engines.
- Reference: `context/summaries/modelopt-kb/howto-modelopt-onnx-ptq-for-yolo11.md`

### Intro: ModelOpt mixed precision and sensitivity for YOLO-style CNNs

- Focus: conceptual and practical overview of how ModelOpt estimates layer
  sensitivity and applies mixed precision in both PyTorch and ONNX paths,
  including `auto_quantize`, AutoCast, and how these relate to TensorRT
  mixed-precision engines.
- Use when designing mixed FP16/INT8 (or FP32/FP16) schemes for YOLO11 and
  deciding which layers to keep in higher precision.
- Reference: `context/summaries/modelopt-kb/intro-modelopt-mixed-precision-sensitivity.md`

