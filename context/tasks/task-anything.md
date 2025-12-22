# Task: Quantize CV YOLOv10m low-bit via layer sensitivity analysis

## What to do

- Quantize `models/cv-models/yolov10m/checkpoints/yolov10m.onnx` using low-bit schemes (INT8 baseline + low-bit/mixed candidates) guided by per-layer sensitivity, then compare accuracy (COCO mAP) and performance (ORT/TensorRT) against baseline.

Related:
- Main implementation plan: `context/plans/plan-quantize-yolov10m-low-bit-sensitivity.md`
- RTX 5090 environment: `context/instructions/prep-rtx5090.md`
- Model reference: `models/cv-models/yolov10m/README.md`

## 1. YOLOv10m low-bit quantization workflow

Short description: Establish a reproducible workflow for baseline evaluation, ModelOpt ONNX PTQ (INT8), and sensitivity-guided mixed-precision experiments, with outputs captured under `tmp/` for later comparison.

### Scope

> Important: `models/cv-models/yolov10m` (ONNX) and `models/yolo10` (Ultralytics PyTorch) are independent model artifacts. Even if the names match (`yolov10m`), do not assume identical weights/graphs or that sensitivity rankings transfer 1:1.

- **In scope**:
  - Baseline smoke checks + baseline COCO evaluation for the target ONNX model.
  - Calibration tensor generation for ONNX PTQ (YOLO-style preprocessing).
  - ModelOpt ONNX PTQ to produce a deployable **INT8 Q/DQ ONNX** baseline.
  - Per-layer sensitivity runs (Torch/Ultralytics YOLOv10m) as a methodology/proxy, plus candidate scheme definition.
  - Materialize and evaluate a small set of mixed/low-bit candidates (subject to tool support and layer-name alignment).
- **Out of scope (for this task)**:
  - Training/fine-tuning.
  - Claiming exact transferability between Torch and the independently managed `models/cv-models` ONNX checkpoint without verification.

### Planned outputs

- Baseline evaluation + benchmarking outputs under `tmp/yolov10m_lowbit/<run-id>/...`
- For testing/iteration, use the **medium** COCO2017 subset (100 val images) consistently across baseline and candidates.
- Calibration tensor (`.npy`) for YOLOv10m ONNX PTQ
- ModelOpt ONNX PTQ wrapper script for YOLOv10m INT8 QDQ generation
- YOLOv10m ONNX COCO evaluation script (or a validated Ultralytics-backed ONNX evaluation path)
- A short results summary (accuracy + latency) comparing baseline vs candidates

### Milestones (subtasks)

#### 1.1 Bootstrap assets and baseline smoke checks

Goal: Verify the target ONNX checkpoint + datasets are present and confirm baseline ONNXRuntime inference runs end-to-end in the `rtx5090` Pixi environment.

- Subtask spec: `context/tasks/working/quantize-yolov10m-low-bit-sensitivity/subtask-001-101-bootstrap-baseline.md`

#### 1.2 Calibration preprocessing and baseline COCO evaluation (ONNX)

Goal: Create a reproducible preprocessing pipeline (letterbox + normalization), generate a calibration tensor, and establish a baseline COCO evaluation path for the YOLOv10m ONNX outputs.

- Subtask spec: `context/tasks/working/quantize-yolov10m-low-bit-sensitivity/subtask-001-102-calibration-and-baseline-coco-eval.md`

#### 1.3 INT8 ONNX PTQ baseline (ModelOpt) + validation

Goal: Produce an INT8 Q/DQ ONNX artifact for `models/cv-models/yolov10m` using ModelOpt ONNX PTQ and validate it via ORT + COCO evaluation.

- Subtask spec: `context/tasks/working/quantize-yolov10m-low-bit-sensitivity/subtask-001-103-int8-onnx-ptq.md`

#### 1.4 Layer sensitivity sweep (Torch) and candidate mixed schemes

Goal: Run the existing YOLOv10 Torch sensitivity sweep tooling and translate the results into a small, explicit set of candidate mixed/low-bit schemes (top‑K policy), noting assumptions/limitations for ONNX transfer.

- Subtask spec: `context/tasks/working/quantize-yolov10m-low-bit-sensitivity/subtask-001-104-layer-sensitivity-and-schemes.md`

#### 1.5 Materialize low-bit candidates and benchmark (ORT/TensorRT)

Goal: Generate mixed/low-bit candidate artifacts via the chosen approach (ONNX-native exclusions vs Torch→export), then evaluate accuracy + benchmark latency/throughput.

- Subtask spec: `context/tasks/working/quantize-yolov10m-low-bit-sensitivity/subtask-001-105-materialize-lowbit-candidates.md`

#### 1.6 Summarize results, update docs, and optional INC cross-check

Goal: Write a concise results summary (baseline vs candidates), update local docs with reproduction commands, and optionally run an INC cross-check for INT8.

- Subtask spec: `context/tasks/working/quantize-yolov10m-low-bit-sensitivity/subtask-001-106-summarize-and-docs-inc.md`

### TODOs

- [ ] Job-001-101: Complete subtask 1.1: Bootstrap assets and baseline smoke checks.
- [ ] Job-001-102: Complete subtask 1.2: Calibration preprocessing and baseline COCO evaluation (ONNX).
- [ ] Job-001-103: Complete subtask 1.3: INT8 ONNX PTQ baseline (ModelOpt) + validation.
- [ ] Job-001-104: Complete subtask 1.4: Layer sensitivity sweep (Torch) and candidate mixed schemes.
- [ ] Job-001-105: Complete subtask 1.5: Materialize low-bit candidates and benchmark (ORT/TensorRT).
- [ ] Job-001-106: Complete subtask 1.6: Summarize results, update docs, and optional INC cross-check.
