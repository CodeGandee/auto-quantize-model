# CV Model: yolov8s

## HEADER
- **Purpose**: Provide a stable local path for the yolov8s ONNX checkpoint
- **Status**: Active
- **Date**: 2025-12-19
- **Dependencies**: Local model storage at /workspace/model-to-quantize/General-CV-Models
- **Target**: AI assistants and developers

## Overview

Ultralytics YOLOv8 is a modern real-time object detector; `yolov8s` is the small variant.

References:
- https://github.com/ultralytics/ultralytics

## Content

This directory contains symlinks to externally stored ONNX checkpoints:

- `checkpoints/yolov8s.onnx` -> `/workspace/model-to-quantize/General-CV-Models/yolov8s.onnx`

These checkpoint files are not committed to the repository. Update the symlink or replace the source file if the model location changes.

## Model Stats

- **ONNX file size**: 42.75 MiB
- **Parameter count**: 11,156,558
- **Parameter bytes**: 42.56 MiB
- **Compute (MACs)**: 14.301 GMACs @ input [1, 3, 640, 640] (Conv/MatMul/Gemm)
- **Compute (FLOPs)**: 28.603 GFLOPs @ 2 FLOPs per MAC
- **Assumed batch**: 1 (for symbolic batch dims)

- **Notes**: Parameter count/bytes are computed from ONNX initializer tensors; compute is MACs from Conv/MatMul/Gemm nodes only (elementwise ops excluded).

## ONNX I/O

**Inputs** (shape-inferred)
- `images`: `FLOAT` `[1, 3, 640, 640]`

**Outputs**
- `output0`: `FLOAT` `[1, 84, 8400]`
