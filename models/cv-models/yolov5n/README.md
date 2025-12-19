# CV Model: yolov5n

## HEADER
- **Purpose**: Provide a stable local path for the yolov5n ONNX checkpoint
- **Status**: Active
- **Date**: 2025-12-19
- **Dependencies**: Local model storage at /workspace/model-to-quantize/General-CV-Models
- **Target**: AI assistants and developers

## Overview

Ultralytics YOLOv5 is a real-time object detector; `yolov5n` is the nano variant optimized for speed and small size.

References:
- https://github.com/ultralytics/yolov5

## Content

This directory contains symlinks to externally stored ONNX checkpoints:

- `checkpoints/yolov5n.onnx` -> `/workspace/model-to-quantize/General-CV-Models/yolov5n.onnx`

These checkpoint files are not committed to the repository. Update the symlink or replace the source file if the model location changes.

## Model Stats

- **ONNX file size**: 7.54 MiB
- **Parameter count**: 1,867,437
- **Parameter bytes**: 7.12 MiB
- **Compute (MACs)**: 2.234 GMACs @ input [1, 3, 640, 640] (Conv/MatMul/Gemm)
- **Compute (FLOPs)**: 4.468 GFLOPs @ 2 FLOPs per MAC
- **Assumed batch**: 1 (for symbolic batch dims)

- **Notes**: Parameter count/bytes are computed from ONNX initializer tensors; compute is MACs from Conv/MatMul/Gemm nodes only (elementwise ops excluded).

## ONNX I/O

**Inputs** (shape-inferred)
- `images`: `FLOAT` `[1, 3, 640, 640]`

**Outputs**
- `output`: `FLOAT` `[1, 25200, 85]`
- `onnx::Sigmoid_339`: `FLOAT` `[1, 3, 80, 80, 85]`
- `onnx::Sigmoid_391`: `FLOAT` `[1, 3, 40, 40, 85]`
- `onnx::Sigmoid_443`: `FLOAT` `[1, 3, 20, 20, 85]`
