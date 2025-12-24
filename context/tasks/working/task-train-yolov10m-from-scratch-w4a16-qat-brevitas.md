# Task: Train YOLOv10m from scratch (COCO2017) — FP16 baseline vs Brevitas W4A16 QAT

## What to do

- Implement a reproducible Pixi `rtx5090` workflow that:
  - trains **YOLOv10m from random init** on COCO2017 (full train2017 + val2017),
  - runs **(1) FP16/AMP baseline** and **(2) Brevitas W4A16 QAT** (weight-only int4 fake-quant),
  - saves checkpoints **every 5 epochs** and keeps **TensorBoard logs**,
  - exports both runs to ONNX and evaluates both with the repo’s ONNX COCO evaluator,
  - writes a run-local `summary.md` and produces a **training loss graph** artifact.

Related:
- Main plan: `context/plans/plan-train-yolov10m-from-scratch-w4a16-qat-brevitas.md`
- Dev env: `context/instructions/prep-rtx5090.md`
- Scratch hyperparams reference: `extern/quantized-yolov5/data/hyps/hyp.scratch.yaml`
- Existing Brevitas YOLOv10 helpers: `src/auto_quantize_model/cv_models/yolov10_brevitas.py`
- ONNX COCO eval: `scripts/cv-models/eval_yolov10m_onnx_coco.py`

## 1. Scratch FP16 vs W4A16 QAT training pipeline

### Scope

- Use local YOLOv10 sources under `models/yolo10/src/` (Ultralytics fork) and YOLOv10m cfg:
  - `models/yolo10/src/ultralytics/cfg/models/v10/yolov10m.yaml`
- Use COCO2017 via `datasets/coco2017/source-data/` (train2017/ val2017/ annotations/).
- Run artifacts must be written under:
  - `tmp/yolov10m_scratch_fp16_vs_w4a16_qat_brevitas/<run-id>/` (not committed)

### Planned outputs

- A single runner command that produces:
  - `fp16/` and `qat-w4a16/` training dirs (checkpoints, TensorBoard logs, results CSV, loss PNG),
  - `onnx/` exports for baseline and QAT (QCDQ for QAT),
  - `eval/` metrics JSON for both,
  - `summary/summary.md` and a loss graph (PNG) consumable without TensorBoard.

### Milestones (subtasks)

#### 1.1 Run-root + dataset conversion + provenance

Goal: Define a stable run-root layout under `tmp/`, create a run-local YOLO-format COCO dataset (labels + symlinked images), and record provenance/counts.

- Subtask spec: `context/tasks/working/train-yolov10m-from-scratch-w4a16-qat-brevitas/subtask-001-101-run-root-and-dataset.md`

#### 1.2 Scratch hyperparameters config

Goal: Port/mirror `extern/quantized-yolov5/data/hyps/hyp.scratch.yaml` into an Ultralytics-friendly config under `conf/`.

- Subtask spec: `context/tasks/working/train-yolov10m-from-scratch-w4a16-qat-brevitas/subtask-001-102-scratch-hypers.md`

#### 1.3 FP16 scratch training (baseline)

Goal: Train YOLOv10m from YAML (random init) with AMP, TensorBoard logs, and checkpoints every 5 epochs.

- Subtask spec: `context/tasks/working/train-yolov10m-from-scratch-w4a16-qat-brevitas/subtask-001-103-fp16-scratch-training.md`

#### 1.4 Brevitas W4A16 model builder (scratch)

Goal: Build YOLOv10m from YAML and apply Brevitas weight-only int4 fake-quant (Conv2d → QuantConv2d), keeping activations floating (A16).

- Subtask spec: `context/tasks/working/train-yolov10m-from-scratch-w4a16-qat-brevitas/subtask-001-104-brevitas-w4a16-qat-model.md`

#### 1.5 QAT training + checkpointing

Goal: Integrate the W4A16 model into Ultralytics training (from scratch), producing TensorBoard logs, loss curves, and checkpoints every 5 epochs.

- Subtask spec: `context/tasks/working/train-yolov10m-from-scratch-w4a16-qat-brevitas/subtask-001-105-qat-training-and-checkpoints.md`

#### 1.6 Export + eval + summary (incl. loss graph)

Goal: Export baseline and QAT models to ONNX, run COCO val evaluation via `scripts/cv-models/eval_yolov10m_onnx_coco.py`, and write a run-local `summary.md` plus a final loss graph artifact.

- Subtask spec: `context/tasks/working/train-yolov10m-from-scratch-w4a16-qat-brevitas/subtask-001-106-export-eval-summary.md`

### TODOs

- [ ] Job-001-101: Complete subtask 1.1: Run-root + dataset conversion + provenance.
- [ ] Job-001-102: Complete subtask 1.2: Scratch hyperparameters config.
- [ ] Job-001-103: Complete subtask 1.3: FP16 scratch training (baseline).
- [ ] Job-001-104: Complete subtask 1.4: Brevitas W4A16 model builder (scratch).
- [ ] Job-001-105: Complete subtask 1.5: QAT training + checkpointing.
- [ ] Job-001-106: Complete subtask 1.6: Export + eval + summary (incl. loss graph).

