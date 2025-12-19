# CV Model: yolov2-coco-9

## HEADER
- **Purpose**: Provide a stable local path for the yolov2-coco-9 ONNX checkpoint
- **Status**: Active
- **Date**: 2025-12-19
- **Dependencies**: Local model storage at /workspace/model-to-quantize/General-CV-Models
- **Target**: AI assistants and developers

## Overview

YOLOv2 (YOLO9000) is a single-stage object detector. This ONNX checkpoint emits a YOLOv2-style prediction map (grid output) suitable for downstream decoding.

References:
- https://arxiv.org/abs/1612.08242

## Content

This directory contains symlinks to externally stored ONNX checkpoints:

- `checkpoints/yolov2-coco-9.onnx` -> `/workspace/model-to-quantize/General-CV-Models/yolov2-coco-9.onnx`

These checkpoint files are not committed to the repository. Update the symlink or replace the source file if the model location changes.

## Model Stats

- **ONNX file size**: 194.50 MiB
- **Parameter count**: 50,983,561
- **Parameter bytes**: 194.49 MiB
- **Compute (MACs)**: 14.732 GMACs @ input [1, 3, 416, 416] (Conv/MatMul/Gemm)
- **Compute (FLOPs)**: 29.464 GFLOPs @ 2 FLOPs per MAC
- **Assumed batch**: 1 (for symbolic batch dims)

- **Notes**: Parameter count/bytes are computed from ONNX initializer tensors; compute is MACs from Conv/MatMul/Gemm nodes only (elementwise ops excluded).

## ONNX I/O

**Inputs** (shape-inferred)
- `input.1`: `FLOAT` `[1, 3, 416, 416]`

**Outputs**
- `218`: `FLOAT` `[1, 425, 13, 13]`
