# CV Model: yolov5s

## HEADER
- **Purpose**: Provide a stable local path for the yolov5s ONNX checkpoint
- **Status**: Active
- **Date**: 2025-12-19
- **Dependencies**: Local model storage at /workspace/model-to-quantize/General-CV-Models
- **Target**: AI assistants and developers

## Overview

Ultralytics YOLOv5 is a real-time object detector; `yolov5s` is the small variant balancing speed and accuracy.

References:
- https://github.com/ultralytics/yolov5

## Content

This directory contains symlinks to externally stored ONNX checkpoints:

- `checkpoints/yolov5s.onnx` -> `/workspace/model-to-quantize/General-CV-Models/yolov5s.onnx`

These checkpoint files are not committed to the repository. Update the symlink or replace the source file if the model location changes.

## Model Stats

- **ONNX file size**: 27.82 MiB
- **Parameter count**: 7,283,825
- **Parameter bytes**: 27.79 MiB
- **Compute (MACs)**: 8.479 GMACs @ input [1, 3, 640, 640] (Conv/MatMul/Gemm)
- **Compute (FLOPs)**: 16.958 GFLOPs @ 2 FLOPs per MAC
- **Assumed batch**: 1 (for symbolic batch dims)

- **Notes**: Parameter count/bytes are computed from ONNX initializer tensors; compute is MACs from Conv/MatMul/Gemm nodes only (elementwise ops excluded).

## ONNX I/O

**Inputs** (shape-inferred)
- `images`: `FLOAT` `[1, 3, 640, 640]`

**Outputs**
- `pred`: `FLOAT` `[1, 25200, 85]`
- `output2`: `FLOAT` `[1, 3, 80, 80, 85]`
- `output3`: `FLOAT` `[1, 3, 40, 40, 85]`
- `output4`: `FLOAT` `[1, 3, 20, 20, 85]`
