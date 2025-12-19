# CV Model: inception-v3

## HEADER
- **Purpose**: Provide a stable local path for the inception-v3 ONNX checkpoint
- **Status**: Active
- **Date**: 2025-12-19
- **Dependencies**: Local model storage at /workspace/model-to-quantize/General-CV-Models
- **Target**: AI assistants and developers

## Overview

Inception-v3 is an Inception-family convolutional network for ImageNet-style image classification, designed to be compute-efficient via factorized convolutions.

References:
- https://arxiv.org/abs/1512.00567

## Content

This directory contains symlinks to externally stored ONNX checkpoints:

- `checkpoints/inception-v3.onnx` -> `/workspace/model-to-quantize/General-CV-Models/inception-v3.onnx`

These checkpoint files are not committed to the repository. Update the symlink or replace the source file if the model location changes.

## Model Stats

- **ONNX file size**: 91.14 MiB
- **Parameter count**: 23,869,000
- **Parameter bytes**: 91.05 MiB
- **Compute (MACs)**: 5.713 GMACs @ input [1, 3, 299, 299] (Conv/MatMul/Gemm)
- **Compute (FLOPs)**: 11.426 GFLOPs @ 2 FLOPs per MAC
- **Assumed batch**: 1 (for symbolic batch dims)

- **Notes**: Parameter count/bytes are computed from ONNX initializer tensors; compute is MACs from Conv/MatMul/Gemm nodes only (elementwise ops excluded).

## ONNX I/O

**Inputs** (shape-inferred)
- `data`: `FLOAT` `[1, 3, 299, 299]`

**Outputs**
- `prob`: `FLOAT` `[1, 1000, 1, 1]`
