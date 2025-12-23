# YOLOv10 Model Assets

## HEADER
- **Purpose**: Manage YOLOv10 source and pretrained checkpoints for this project
- **Status**: Active
- **Date**: 2025-11-28
- **Dependencies**: Git, curl or wget, internet access
- **Target**: AI assistants and developers

## Content

This directory contains helper tooling and local assets for working with YOLOv10 models:

- `bootstrap.sh` — bootstrap script that:
  - Clones the official YOLOv10 repository into `src/` with `--depth 1`.
  - Downloads YOLOv10 nano/small/medium/base/large/xlarge checkpoints (`yolov10n/s/m/b/l/x.pt`) into `checkpoints/`.
  - Ensures `src/`, `checkpoints/`, and `tmp/` are ignored by Git via `.gitignore`.
- `src/` — YOLOv10 source code cloned from the YOLOv10 GitHub repository (not committed to this repo).
- `checkpoints/` — downloaded YOLOv10 pretrained weights (not committed to this repo).
- `tmp/` — optional temporary download/extraction area used by scripts; removed after successful bootstrap.

To (re)initialize the YOLOv10 assets, run from the project root:

```bash
./models/yolo10/bootstrap.sh
```

## Brevitas PTQ/QAT W4A16(-like) and W4A8 (INT8 activations)

This repo includes a Brevitas-based workflow that starts from the public Ultralytics checkpoint
`models/yolo10/checkpoints/yolov10m.pt` and exports inspectable **QCDQ ONNX** artifacts:

- **W4A16(-like)**: 4-bit weights, floating activations/compute (typically FP16/FP32).
- **W4A8**: 4-bit weights + INT8 fake-quant activations (Q/DQ in ONNX).

Important caveat: these exports are typically *fake-quantized* (Q/DQ around ops); convolution compute is not expected
to run in true INT4 kernels on ORT CUDA EP.

### End-to-end runner (baseline + PTQ, optional QAT)

```bash
RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
RUN_ROOT="tmp/yolov10m_brevitas_w4a8_w4a16/${RUN_ID}"

# Baseline + PTQ exports + smoke + COCO subset eval (100 images)
pixi run -e rtx5090 bash scripts/cv-models/run_yolov10m_brevitas_w4_ptq_qat.sh

# Optional: run a short QAT step (writes under $RUN_ROOT/qat/)
RUN_QAT=1 QAT_MODE=w4a8 pixi run -e rtx5090 bash scripts/cv-models/run_yolov10m_brevitas_w4_ptq_qat.sh
```

Outputs:

- `tmp/yolov10m_brevitas_w4a8_w4a16/<run-id>/onnx/` (baseline/PTQ/QAT ONNX)
- `tmp/yolov10m_brevitas_w4a8_w4a16/<run-id>/*-coco/metrics.json` (COCO subset metrics/latency)
- `tmp/yolov10m_brevitas_w4a8_w4a16/<run-id>/summary.md`

### Building blocks (script entrypoint)

The runner calls `scripts/cv-models/quantize_yolov10m_brevitas_w4.py`:

```bash
RUN_ROOT="tmp/yolov10m_brevitas_w4a8_w4a16/$(date +%Y-%m-%d_%H-%M-%S)"

pixi run -e rtx5090 python scripts/cv-models/quantize_yolov10m_brevitas_w4.py baseline --run-root "$RUN_ROOT"
pixi run -e rtx5090 python scripts/cv-models/quantize_yolov10m_brevitas_w4.py ptq --mode w4a16 --run-root "$RUN_ROOT"
pixi run -e rtx5090 python scripts/cv-models/quantize_yolov10m_brevitas_w4.py ptq --mode w4a8 --run-root "$RUN_ROOT"
```
