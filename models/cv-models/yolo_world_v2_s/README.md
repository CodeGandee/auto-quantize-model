# CV Model: yolo_world_v2_s

## HEADER
- **Purpose**: Provide a stable local path for the yolo_world_v2_s ONNX checkpoint
- **Status**: Active
- **Date**: 2025-12-19
- **Dependencies**: Local model storage at /workspace/model-to-quantize/General-CV-Models
- **Target**: AI assistants and developers

## Overview

YOLO-World is a real-time open-vocabulary object detector. This exported ONNX checkpoint takes images as input and emits detection boxes and per-class scores.

References:
- https://arxiv.org/abs/2401.17270

## Content

This directory contains symlinks to externally stored ONNX checkpoints:

- `checkpoints/yolo_world_v2_s.onnx` -> `/workspace/model-to-quantize/General-CV-Models/yolo_world_v2_s.onnx`

These checkpoint files are not committed to the repository. Update the symlink or replace the source file if the model location changes.

## Model Stats

- **ONNX file size**: 50.58 MiB
- **Parameter count**: 13,222,681
- **Parameter bytes**: 50.44 MiB
- **Compute (MACs)**: 17.825 GMACs @ input [1, 3, 640, 640] (Conv/MatMul/Gemm)
- **Compute (FLOPs)**: 35.651 GFLOPs @ 2 FLOPs per MAC
- **Assumed batch**: 1 (for symbolic batch dims)

- **Notes**: Parameter count/bytes are computed from ONNX initializer tensors; compute is MACs from Conv/MatMul/Gemm nodes only (elementwise ops excluded).

## ONNX I/O

**Inputs** (shape-inferred)
- `images`: `FLOAT` `[1, 3, 640, 640]`

**Outputs**
- `scores`: `FLOAT` `[1, 8400, 365]`
- `boxes`: `FLOAT` `[1, 8400, 4]`
