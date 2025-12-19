# CV Model: mobilenet_v1_1.0_224

## HEADER
- **Purpose**: Provide a stable local path for the mobilenet_v1_1.0_224 ONNX checkpoint
- **Status**: Active
- **Date**: 2025-12-19
- **Dependencies**: Local model storage at /workspace/model-to-quantize/General-CV-Models
- **Target**: AI assistants and developers

## Overview

MobileNetV1 is a lightweight convolutional network for image classification built around depthwise separable convolutions (this checkpoint is the 1.0 width / 224px configuration).

References:
- https://arxiv.org/abs/1704.04861

## Content

This directory contains symlinks to externally stored ONNX checkpoints:

- `checkpoints/mobilenet_v1_1.0_224.onnx` -> `/workspace/model-to-quantize/General-CV-Models/mobilenet_v1_1.0_224.onnx`

These checkpoint files are not committed to the repository. Update the symlink or replace the source file if the model location changes.

## Model Stats

- **ONNX file size**: 16.13 MiB
- **Parameter count**: 4,222,059
- **Parameter bytes**: 16.11 MiB
- **Compute (MACs)**: 0.569 GMACs @ input [1, 3, 224, 224] (Conv/MatMul/Gemm)
- **Compute (FLOPs)**: 1.137 GFLOPs @ 2 FLOPs per MAC
- **Assumed batch**: 1 (for symbolic batch dims)

- **Notes**: Parameter count/bytes are computed from ONNX initializer tensors; compute is MACs from Conv/MatMul/Gemm nodes only (elementwise ops excluded).

## ONNX I/O

**Inputs** (shape-inferred)
- `input:0`: `FLOAT` `[unk__343, 3, 224, 224]`

**Outputs**
- `MobilenetV1/Predictions/Reshape_1:0`: `FLOAT` `[unk__344, 1001]`
