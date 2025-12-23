# CV Model: yolov10m

## HEADER
- **Purpose**: Provide a stable local path for the yolov10m ONNX checkpoint
- **Status**: Active
- **Date**: 2025-12-19
- **Dependencies**: Local model storage at /workspace/model-to-quantize/General-CV-Models
- **Target**: AI assistants and developers

## Overview

YOLOv10 is a real-time end-to-end object detector. This ONNX checkpoint is the medium variant exported for image-only inference.

References:
- https://arxiv.org/abs/2405.14458

Important note:

- The ONNX checkpoint in `models/cv-models/yolov10m/` is managed independently from the Ultralytics YOLOv10 PyTorch assets under `models/yolo10/`.
- Even if the model names match (`yolov10m`), do not assume they share identical weights/graphs or that per-layer sensitivity/quantization results transfer 1:1 between them.

## Content

This directory contains symlinks to externally stored ONNX checkpoints:

- `checkpoints/yolov10m.onnx` -> `/workspace/model-to-quantize/General-CV-Models/yolov10m.onnx`

These checkpoint files are not committed to the repository. Update the symlink or replace the source file if the model location changes.

## Model Stats

- **ONNX file size**: 58.70 MiB
- **Parameter count**: 15,359,498
- **Parameter bytes**: 58.59 MiB
- **Compute (MACs)**: 29.618 GMACs @ input [1, 3, 640, 640] (Conv/MatMul/Gemm)
- **Compute (FLOPs)**: 59.235 GFLOPs @ 2 FLOPs per MAC
- **Assumed batch**: 1 (for symbolic batch dims)

- **Notes**: Parameter count/bytes are computed from ONNX initializer tensors; compute is MACs from Conv/MatMul/Gemm nodes only (elementwise ops excluded).

## ONNX I/O

**Inputs** (shape-inferred)
- `images`: `FLOAT` `[1, 3, 640, 640]`

**Outputs**
- `/model.23/Concat_3_output_0`: `FLOAT` `[1, 144, 8400]`

## Quantization workflow (ModelOpt ONNX PTQ)

All commands below are designed to write run artifacts under `tmp/` (not committed).

### 1) Baseline smoke + COCO eval (ONNX Runtime)

```bash
RUN_ROOT="tmp/yolov10m_lowbit/$(date +%Y-%m-%d_%H-%M-%S)"

# Random-tensor sanity check
pixi run -e rtx5090 python models/cv-models/helpers/run_random_onnx_inference.py \
  --model models/cv-models/yolov10m/checkpoints/yolov10m.onnx \
  --output-root "$RUN_ROOT/baseline-onnx"

# Baseline COCO2017 val (fixed 100-image slice, inference-only latency stats)
pixi run -e rtx5090 python scripts/cv-models/eval_yolov10m_onnx_coco.py \
  --onnx-path models/cv-models/yolov10m/checkpoints/yolov10m.onnx \
  --data-root datasets/coco2017/source-data \
  --max-images 100 \
  --providers TensorrtExecutionProvider CUDAExecutionProvider \
  --disable-cpu-fallback \
  --warmup-runs 10 \
  --skip-latency 10 \
  --imgsz 640 \
  --out "$RUN_ROOT/baseline-coco/metrics.json"
```

Notes:

- Prefer `--providers TensorrtExecutionProvider CUDAExecutionProvider --disable-cpu-fallback` to enforce “GPU only”.
- If a model requires CPU for unsupported ops, include it explicitly instead:
  - `--providers TensorrtExecutionProvider CUDAExecutionProvider CPUExecutionProvider`
  - (and omit `--disable-cpu-fallback`).

### 2) Build calibration tensor (float32, NCHW)

```bash
RUN_ROOT="tmp/yolov10m_lowbit/$(date +%Y-%m-%d_%H-%M-%S)"

pixi run -e rtx5090 python scripts/cv-models/make_yolov10m_calib_npy.py \
  --list datasets/quantize-calib/quant100.txt \
  --out "$RUN_ROOT/calib/calib_yolov10m_640.npy" \
  --imgsz 640
```

### 3) INT8 PTQ (Q/DQ ONNX) + eval

```bash
RUN_ROOT="tmp/yolov10m_lowbit/$(date +%Y-%m-%d_%H-%M-%S)"

RUN_ROOT="$RUN_ROOT" \
CALIB_PATH="$RUN_ROOT/calib/calib_yolov10m_640.npy" \
CALIBRATION_METHOD="entropy" \
USE_ZERO_POINT=True \
CALIBRATION_EPS="cuda:0 cpu" \
pixi run -e rtx5090 bash scripts/cv-models/quantize_yolov10m_int8_onnx.sh

pixi run -e rtx5090 python scripts/cv-models/eval_yolov10m_onnx_coco.py \
  --onnx-path "$RUN_ROOT/onnx/yolov10m-int8-qdq.onnx" \
  --data-root datasets/coco2017/source-data \
  --max-images 100 \
  --providers CUDAExecutionProvider CPUExecutionProvider \
  --warmup-runs 10 \
  --skip-latency 10 \
  --imgsz 640 \
  --out "$RUN_ROOT/int8-coco/metrics.json"
```

### 4) (Optional) Torch sensitivity → FP8 candidates (node exclusions)

Note: ModelOpt ONNX `quantize_mode=int4` targets Gemm/MatMul by default and does not materially quantize this
Conv-dominated YOLOv10m graph; FP8 is the practical “low-bit” ONNX PTQ candidate here.

```bash
# 4a) Run sensitivity sweep (Torch / AutoQuant proxy)
pixi run -e rtx5090 bash scripts/cv-models/run_yolov10m_layer_sensitivity_sweep.sh

# 4b) Convert a sensitivity report to ONNX node exclusion schemes
RUN_ROOT="tmp/yolov10m_lowbit/$(date +%Y-%m-%d_%H-%M-%S)"
SENS_RUN="tmp/yolov10m_layer_sensitivity/<run-id>"
pixi run -e rtx5090 python scripts/cv-models/make_yolov10m_candidate_schemes.py \
  --report-json "$SENS_RUN/outputs/yolov10m/fp8-fp8/per_layer/layer-sensitivity-report.json" \
  --out-dir "$RUN_ROOT/schemes"

# 4c) Materialize FP8 candidates (K ∈ {0,5,10,20})
RUN_ROOT="$RUN_ROOT" \
CALIB_PATH="$RUN_ROOT/calib/calib_yolov10m_640.npy" \
pixi run -e rtx5090 bash scripts/cv-models/materialize_yolov10m_lowbit_candidates.sh
```
