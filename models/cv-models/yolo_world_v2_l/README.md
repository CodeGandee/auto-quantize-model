# CV Model: yolo_world_v2_l

## HEADER
- **Purpose**: Provide a stable local path for the yolo_world_v2_l ONNX checkpoint
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

- `checkpoints/yolo_world_v2_l.onnx` -> `/workspace/model-to-quantize/General-CV-Models/yolo_world_v2_l.onnx`

These checkpoint files are not committed to the repository. Update the symlink or replace the source file if the model location changes.

## Model Stats

- **ONNX file size**: 180.37 MiB
- **Parameter count**: 47,233,955
- **Parameter bytes**: 180.18 MiB
- **Compute (MACs)**: 89.457 GMACs @ input [1, 3, 640, 640] (Conv/MatMul/Gemm)
- **Compute (FLOPs)**: 178.914 GFLOPs @ 2 FLOPs per MAC
- **Assumed batch**: 1 (for symbolic batch dims)

- **Notes**: Parameter count/bytes are computed from ONNX initializer tensors; compute is MACs from Conv/MatMul/Gemm nodes only (elementwise ops excluded).

## ONNX I/O

**Inputs** (shape-inferred)
- `images`: `FLOAT` `[1, 3, 640, 640]`

**Outputs**
- `scores`: `FLOAT` `[1, 8400, 365]`
- `boxes`: `FLOAT` `[1, 8400, 4]`
