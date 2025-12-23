# Task: Quantize Ultralytics YOLOv10m to W4A16(-like) and W4A8 (INT8) ONNX (Brevitas)

## What to do

- Starting from the **public Ultralytics checkpoint** `models/yolo10/checkpoints/yolov10m.pt`, produce reproducible ONNX artifacts that express:
  - **W4A16(-like)**: 4-bit weights with FP16 activations/compute (weight-only quant + FP16 graph where feasible).
  - **W4A8 (INT8 activations)**: 4-bit weights with INT8 fake-quant activations (Q/DQ in ONNX; compute after dequant is FP16/FP32).
- Validate all artifacts run on **ONNX Runtime CUDA EP** in the Pixi `rtx5090` environment and compare baseline vs PTQ vs optional QAT on a small COCO val subset.

Related:
- Main plan: `context/plans/plan-quantize-yolov10m-w4a8-w4a16-brevitas.md`
- Brevitas/ONNX GPU notes: `context/hints/about-brevitas-yolo-w4a8-w4a16-onnx-nvidia-gpu.md`
- COCO eval script: `scripts/cv-models/eval_yolov10m_onnx_coco.py`

## 1. Brevitas PTQ/QAT export pipeline

Short description: Establish a baseline export and a reproducible Brevitas PTQ pipeline for W4A16-like and W4A8 (INT8 activations), with an optional QAT recovery step when PTQ accuracy loss is too large.

### Scope

- Work from `models/yolo10/checkpoints/yolov10m.pt` (Ultralytics Torch).
- Export QCDQ ONNX via Brevitas (`export_onnx_qcdq(..., dynamo=False)`) with Torch 2.9 compatibility fixes.
- Evaluate baseline/PTQ/QAT ONNX models using ORT CUDA EP (CPU fallback only for diagnostics).

### Planned outputs

- A run script that produces `*-ptq.onnx` (and optionally `*-qat.onnx`) artifacts under `tmp/yolov10m_brevitas_w4a8_w4a16/<run-id>/`.
- A small repro bundle per run (logs + config snapshots) and a `summary.md` comparing baseline vs PTQ vs QAT on the fixed COCO subset.
- A short runbook update under `models/yolo10/README.md` with commands and caveats.

### Milestones (subtasks)

#### 1.1 Baseline export + evaluation

Goal: Export a baseline ONNX from `yolov10m.pt`, confirm ORT CUDA EP runs it, and establish baseline COCO subset metrics/latency.

- Subtask spec: `context/tasks/working/quantize-yolov10m-w4a8-w4a16-brevitas/subtask-001-101-baseline-export-and-eval.md`

#### 1.2 Brevitas ONNX export compatibility helper

Goal: Add a small helper to make Brevitas QCDQ ONNX export work on Torch 2.9 (rtx5090 env) without ad-hoc monkeypatching.

- Subtask spec: `context/tasks/working/quantize-yolov10m-w4a8-w4a16-brevitas/subtask-001-102-brevitas-onnx-export-compat.md`

#### 1.3 PTQ W4A16(-like) export + validation

Goal: Implement weight-only 4-bit quantization (W4A16-like) and export a runnable QCDQ ONNX artifact.

- Subtask spec: `context/tasks/working/quantize-yolov10m-w4a8-w4a16-brevitas/subtask-001-103-ptq-w4a16-export.md`

#### 1.4 PTQ W4A8 (INT8 activations) export + validation

Goal: Add INT8 activation fake-quant (with calibration) and export a runnable QCDQ ONNX artifact.

- Subtask spec: `context/tasks/working/quantize-yolov10m-w4a8-w4a16-brevitas/subtask-001-104-ptq-w4a8-int8-act-export.md`

#### 1.5 Optional QAT recovery + export

Goal: If PTQ accuracy loss is too large, run a short QAT fine-tune and export QAT QCDQ ONNX artifacts.

- Subtask spec: `context/tasks/working/quantize-yolov10m-w4a8-w4a16-brevitas/subtask-001-105-qat-finetune-export.md`

#### 1.6 Orchestration + optimization + summary + quality gates

Goal: Wire an end-to-end runner, apply `onnxoptimizer` cleanup (keeping Q/DQ), run evals, write `summary.md`, and add a runbook section.

- Subtask spec: `context/tasks/working/quantize-yolov10m-w4a8-w4a16-brevitas/subtask-001-106-optimize-eval-summary.md`

### TODOs

- [ ] Job-001-101: Complete subtask 1.1: Baseline export + evaluation.
- [ ] Job-001-102: Complete subtask 1.2: Brevitas ONNX export compatibility helper.
- [ ] Job-001-103: Complete subtask 1.3: PTQ W4A16(-like) export + validation.
- [ ] Job-001-104: Complete subtask 1.4: PTQ W4A8 (INT8 activations) export + validation.
- [ ] Job-001-105: Complete subtask 1.5: Optional QAT recovery + export.
- [ ] Job-001-106: Complete subtask 1.6: Orchestration + optimization + summary + quality gates.

