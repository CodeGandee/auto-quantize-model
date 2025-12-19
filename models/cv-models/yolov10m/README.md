# CV Model: yolov10m

## HEADER
- **Purpose**: Provide a stable local path for the yolov10m ONNX checkpoint
- **Status**: Active
- **Date**: 2025-12-19
- **Dependencies**: Local model storage at /workspace/model-to-quantize/General-CV-Models
- **Target**: AI assistants and developers

## Overview

YOLOv10 is a real-time end-to-end object detector. This ONNX checkpoint is the medium variant exported for image-only inference.

References:
- https://arxiv.org/abs/2405.14458

Important note:

- The ONNX checkpoint in `models/cv-models/yolov10m/` is managed independently from the Ultralytics YOLOv10 PyTorch assets under `models/yolo10/`.
- Even if the model names match (`yolov10m`), do not assume they share identical weights/graphs or that per-layer sensitivity/quantization results transfer 1:1 between them.

## Content

This directory contains symlinks to externally stored ONNX checkpoints:

- `checkpoints/yolov10m.onnx` -> `/workspace/model-to-quantize/General-CV-Models/yolov10m.onnx`

These checkpoint files are not committed to the repository. Update the symlink or replace the source file if the model location changes.

## Model Stats

- **ONNX file size**: 58.70 MiB
- **Parameter count**: 15,359,498
- **Parameter bytes**: 58.59 MiB
- **Compute (MACs)**: 29.618 GMACs @ input [1, 3, 640, 640] (Conv/MatMul/Gemm)
- **Compute (FLOPs)**: 59.235 GFLOPs @ 2 FLOPs per MAC
- **Assumed batch**: 1 (for symbolic batch dims)

- **Notes**: Parameter count/bytes are computed from ONNX initializer tensors; compute is MACs from Conv/MatMul/Gemm nodes only (elementwise ops excluded).

## ONNX I/O

**Inputs** (shape-inferred)
- `images`: `FLOAT` `[1, 3, 640, 640]`

**Outputs**
- `/model.23/Concat_3_output_0`: `FLOAT` `[1, 144, 8400]`
