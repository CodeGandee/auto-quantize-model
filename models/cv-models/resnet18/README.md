# CV Model: resnet18

## HEADER
- **Purpose**: Provide a stable local path for the resnet18 ONNX checkpoint
- **Status**: Active
- **Date**: 2025-12-19
- **Dependencies**: Local model storage at /workspace/model-to-quantize/General-CV-Models
- **Target**: AI assistants and developers

## Overview

ResNet-18 is a residual convolutional network for ImageNet-style image classification using skip connections to enable deeper, easier-to-train models.

References:
- https://arxiv.org/abs/1512.03385

## Content

This directory contains symlinks to externally stored ONNX checkpoints:

- `checkpoints/resnet18.onnx` -> `/workspace/model-to-quantize/General-CV-Models/resnet18.onnx`

These checkpoint files are not committed to the repository. Update the symlink or replace the source file if the model location changes.

## Model Stats

- **ONNX file size**: 44.66 MiB
- **Parameter count**: 11,703,464
- **Parameter bytes**: 44.65 MiB
- **Compute (MACs)**: 1.827 GMACs @ input [1, 3, 224, 224] (Conv/MatMul/Gemm)
- **Compute (FLOPs)**: 3.654 GFLOPs @ 2 FLOPs per MAC
- **Assumed batch**: 1 (for symbolic batch dims)

- **Notes**: Parameter count/bytes are computed from ONNX initializer tensors; compute is MACs from Conv/MatMul/Gemm nodes only (elementwise ops excluded).

## ONNX I/O

**Inputs** (shape-inferred)
- `data`: `FLOAT` `[1, 3, 224, 224]`

**Outputs**
- `loss`: `FLOAT` `[1, 1000, 1, 1]`
