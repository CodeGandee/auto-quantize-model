# CV Model: yolov3

## HEADER
- **Purpose**: Provide a stable local path for the yolov3 ONNX checkpoint
- **Status**: Active
- **Date**: 2025-12-19
- **Dependencies**: Local model storage at /workspace/model-to-quantize/General-CV-Models
- **Target**: AI assistants and developers

## Overview

YOLOv3 is a single-stage object detector with multi-scale prediction heads (e.g., 13x13, 26x26, 52x52 feature maps).

References:
- https://arxiv.org/abs/1804.02767

## Content

This directory contains symlinks to externally stored ONNX checkpoints:

- `checkpoints/yolov3.onnx` -> `/workspace/model-to-quantize/General-CV-Models/yolov3.onnx`

These checkpoint files are not committed to the repository. Update the symlink or replace the source file if the model location changes.

## Model Stats

- **ONNX file size**: 236.58 MiB
- **Parameter count**: 62,003,365
- **Parameter bytes**: 236.52 MiB
- **Compute (MACs)**: 32.932 GMACs @ input [1, 3, 416, 416] (Conv/MatMul/Gemm)
- **Compute (FLOPs)**: 65.864 GFLOPs @ 2 FLOPs per MAC
- **Assumed batch**: 1 (for symbolic batch dims)

- **Notes**: Parameter count/bytes are computed from ONNX initializer tensors; compute is MACs from Conv/MatMul/Gemm nodes only (elementwise ops excluded).

## ONNX I/O

**Inputs** (shape-inferred)
- `data`: `FLOAT` `[1, 3, 416, 416]`

**Outputs**
- `layer82-conv`: `FLOAT` `[1, 255, 13, 13]`
- `layer94-conv`: `FLOAT` `[1, 255, 26, 26]`
- `layer106-conv`: `FLOAT` `[1, 255, 52, 52]`
