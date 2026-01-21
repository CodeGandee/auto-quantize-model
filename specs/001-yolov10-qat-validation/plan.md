# Implementation Plan: Validate YOLOv10 W4A16 QAT Stability (EMA + QC)

**Branch**: `001-yolov10-qat-validation` | **Date**: 2025-12-25 | **Spec**: `specs/001-yolov10-qat-validation/spec.md`
**Input**: Feature specification from `specs/001-yolov10-qat-validation/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command.

## Summary

Validate whether the WACV’24 stabilization approach (EMA + post-hoc Quantization Correction) prevents the known “early peak then collapse” failure mode during W4A16 (weight-only int4) QAT for YOLO-family detectors.

Execution priority is correctness and stability validation on `yolo10n` and `yolo10s` first (fast validation ladder). Only if those are consistently stable do we proceed to `yolo10m`.

## Technical Context

**Language/Version**: Python 3.12 (Pixi-managed; `pyproject.toml` pins `python = "3.12.*"`)  
**Primary Dependencies**: Ultralytics (local clone under `models/yolo10/src/`), PyTorch, Brevitas (W4A16 fake-quant), Hydra/OmegaConf (experiment configs), ONNX/onnxruntime (export/eval), TensorBoard + matplotlib (logging/plots)  
**Storage**: Filesystem artifacts under `tmp/` (not committed) and curated reports under `models/yolo10/reports/<run-id>/`  
**Testing**: `pytest` (unit/integration); heavy GPU training validation as manual tests  
**Target Platform**: Linux + CUDA GPU; development and runs target the Pixi environment `cu128`  
**Project Type**: Single Python repo (library under `src/auto_quantize_model/` + runnable scripts under `scripts/`)  
**Performance Goals**: yolo10n smoke runs in minutes-to-~1h; yolo10s short runs in hours; yolo10m deferred until validated  
**Constraints**: Deterministic experiment metadata (seed/config/provenance), stage-gate before yolo10m, no committed `tmp/` artifacts, DDP-compatible trainer class import paths  
**Scale/Scope**: Dozens of runs (variants × repeats) with small-to-medium run metadata and plots; avoid committing large binaries

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] **Pixi-first** All commands are written as `pixi run -e cu128 ...` (or explicitly justified).
- [x] **Quality gates planned** Lint/type/test commands are specified: `pixi run -e cu128 ruff check .`, `pixi run -e cu128 mypy .`, `pixi run -e cu128 pytest`.
- [x] **Testing strategy** Unit tests for deterministic logic (run classification, summary parsing, gating), integration tests for metadata writing; manual tests for GPU training runs.
- [x] **Reproducibility** Each run writes resolved config, dataset provenance, seed, and code version; `tmp/` remains uncommitted; curated summaries go under `models/yolo10/reports/`.
- [x] **Documentation** Feature docs are produced under `specs/001-yolov10-qat-validation/` (plan/research/data-model/contracts/quickstart).

## Project Structure

### Documentation (this feature)

```text
specs/001-yolov10-qat-validation/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
src/auto_quantize_model/
└── cv_models/
    ├── yolov10_brevitas.py
    ├── yolov10_coco_dataset.py
    ├── yolov10_ultralytics_trainers.py
    └── ... (new EMA/QC helpers for YOLOv10 W4A16 validation)

scripts/cv-models/
├── train_yolov10m_scratch_fp16_vs_w4a16_qat_brevitas.py
└── ... (new runner script to execute yolo10n/s validation ladder and emit comparable summaries)

conf/
└── cv-models/
    └── ... (new configs/presets for yolo10n/yolo10s QAT validation runs)

models/yolo10/
├── src/ultralytics/                  # local Ultralytics YOLOv10 source
└── checkpoints/yolov10{n,s,m}.pt     # pretrained starting points

tests/
├── unit/
├── integration/
└── manual/
```

**Structure Decision**: Single Python project. Implement reusable logic under `src/auto_quantize_model/` and keep experiment entrypoints under `scripts/cv-models/`, configured via `conf/` and emitting artifacts under `tmp/`/`models/yolo10/reports/` per the repository constitution.

## Complexity Tracking

No constitution violations identified for this feature.
