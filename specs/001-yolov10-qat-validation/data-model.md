# Data Model: YOLOv10 W4A16 QAT Validation Artifacts

**Branch**: `001-yolov10-qat-validation`  
**Created**: 2025-12-25  
**Purpose**: Define the entities and fields required to record, compare, and gate W4A16 QAT stabilization experiments (baseline vs EMA vs EMA+QC) for `yolo10n` and `yolo10s`.

## Entity: Experiment Run

Represents one training-and-evaluation attempt for a specific (model variant, method variant, seed).

**Fields**

- `run_id`: Unique identifier (timestamp + model + method + seed).
- `created_at`: ISO-8601 timestamp.
- `git`: commit hash, branch name, and whether the workspace was dirty.
- `model.variant`: `yolo10n | yolo10s | yolo10m`
- `model.checkpoint`: starting checkpoint path (pretrained `.pt`).
- `quantization.mode`: `w4a16`
- `method.variant`: `baseline | ema | ema+qc`
- `training`: epochs, batch, imgsz, device, workers, seed, AMP enabled, and the trainer overrides used.
- `dataset`: provenance and selection information (see Dataset Selection).
- `artifacts`: run-root directory and key files/dirs produced (trainer save_dir, tensorboard, results.csv, checkpoints, exports).
- `metrics`: primary metric name and extracted best/final values (see Metrics Summary).
- `stability`: collapse detection result (see Stability Assessment).
- `status`: `completed-success | completed-failed | incomplete`
- `error`: optional error message/trace (present when `status=incomplete`).

**State Transitions**

- `planned` → `running` → `completed-success`
- `planned` → `running` → `completed-failed` (collapsed)
- `planned` → `running` → `incomplete` (exception/interruption)

## Entity: Dataset Selection

Captures enough detail to ensure runs are comparable.

**Fields**

- `coco_root`: root directory for COCO2017 source data.
- `dataset_yaml`: Ultralytics dataset YAML used for training/eval.
- `train_selection`: source and parameters (e.g., subset list path and selection strategy).
- `val_selection`: source and parameters (e.g., deterministic first-N from COCO val annotations).
- `train_images`: count.
- `val_images`: count.
- `imgsz`: training/eval image size.

**Validation Rules**

- Two runs are “comparable” only if `dataset_yaml` (or its referenced root/splits) and all selection parameters match.

## Entity: Metrics Summary

Minimal metrics needed for stability checks and reviewer comparisons.

**Fields**

- `primary_name`: e.g. `metrics/mAP50-95(B)`
- `best_value`: maximum observed value of the primary metric during the run.
- `final_value`: value of the primary metric at the final evaluated epoch.
- `history`: optional per-epoch series (epoch → metric value) for plotting/diagnostics.

## Entity: Stability Assessment

Formalizes “collapse” and exposes the numbers used to decide it.

**Fields**

- `collapse_threshold_ratio`: fixed ratio (0.5 per spec).
- `final_over_best_ratio`: `final_value / best_value` (if `best_value > 0`).
- `collapsed`: boolean.

**Rule**

- `collapsed = (best_value > 0) and (final_value < collapse_threshold_ratio * best_value)`

## Entity: Stage Gate (yolo10m)

Represents the decision to allow/disallow running `yolo10m` based on `yolo10n` and `yolo10s` validation results.

**Fields**

- `gate_name`: `yolo10m_allowed`
- `inputs`: list of Experiment Runs used for the decision.
- `criteria`: textual description of criteria (aligned with spec success criteria).
- `passed`: boolean.
- `reason`: short text explaining failures (missing runs, collapses, non-comparable configs, etc.).

## Entity: Comparison Summary

A reviewer-facing summary for one model variant, comparing all method variants and repeats.

**Fields**

- `model.variant`
- `runs`: list of Experiment Run references.
- `table`: normalized fields for display (method, seed, status, best/final metric, collapsed).
- `stage_gate`: optional Stage Gate decision (when summarizing the n/s ladder).
