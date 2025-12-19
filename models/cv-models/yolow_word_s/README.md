# CV Model: yolow_word_s

## HEADER
- **Purpose**: Provide a stable local path for the yolow_word_s ONNX checkpoint
- **Status**: Active
- **Date**: 2025-12-19
- **Dependencies**: Local model storage at /workspace/model-to-quantize/General-CV-Models
- **Target**: AI assistants and developers

## Overview

YOLO-World variant (as provided) exported to ONNX for object detection; takes images and emits detection boxes and per-class scores.

References:
- https://arxiv.org/abs/2401.17270

## Content

This directory contains symlinks to externally stored ONNX checkpoints:

- `checkpoints/yolow_word_s.onnx` -> `/workspace/model-to-quantize/General-CV-Models/yolow_word_s.onnx`

These checkpoint files are not committed to the repository. Update the symlink or replace the source file if the model location changes.

## Model Stats

- **ONNX file size**: 47.90 MiB
- **Parameter count**: 12,527,483
- **Parameter bytes**: 47.79 MiB
- **Compute (MACs)**: 4.457 GMACs @ input [1, 3, 256, 448] (Conv/MatMul/Gemm)
- **Compute (FLOPs)**: 8.914 GFLOPs @ 2 FLOPs per MAC
- **Assumed batch**: 1 (for symbolic batch dims)

- **Notes**: Parameter count/bytes are computed from ONNX initializer tensors; compute is MACs from Conv/MatMul/Gemm nodes only (elementwise ops excluded).

## ONNX I/O

**Inputs** (shape-inferred)
- `images`: `FLOAT` `[1, 3, 256, 448]`

**Outputs**
- `boxes`: `FLOAT` `[1, 2352, 4]`
- `scores`: `FLOAT` `[1, 2352, 64]`
