# Implementation Plan: Refactor Qwen3-VL Tutorial Pack Runner (Shared Utility)

**Branch**: `[003-refactor-qwen3-tutorial-runner]` | **Date**: 2026-01-22 | **Spec**: `/data1/huangzhe/code/auto-quantize-model/specs/003-refactor-qwen3-tutorial-runner/spec.md`
**Input**: Feature specification from `/data1/huangzhe/code/auto-quantize-model/specs/003-refactor-qwen3-tutorial-runner/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command.

## Summary

Consolidate the duplicated Qwen3-VL tutorial-pack runners into a shared Python runner utility under `src/auto_quantize_model/qwen/`, keeping the tutorial packs’ `run_demo.sh` as thin wrappers with a stable flag contract. The shared runner owns scenario enumeration, workspace layout, summary generation, snapshotting, and strict verification (summary-only), with CPU-only unit tests for deterministic logic.

## Technical Context

**Language/Version**: Python 3.12 (Pixi-managed; `pyproject.toml` pins `python = "3.12.*"`)  
**Primary Dependencies**: PyTorch, (optionally) Brevitas, Hydra/OmegaConf, ONNX/onnxruntime, Ultralytics (vendored under `models/*/src/`), TensorBoard + matplotlib  
**Storage**: Filesystem artifacts under `tmp/` (uncommitted) and curated reports under `models/*/reports/<run-id>/`  
**Testing**: `pytest` (unit/integration); heavy GPU training validation as manual tests  
**Target Platform**: Linux + CUDA GPU; runs target the appropriate Pixi env feature (e.g., `-e cu128`)  
**Project Type**: Single Python repo (library under `src/auto_quantize_model/` + runnable scripts under `scripts/`)  
**Performance Goals**: No explicit runtime/throughput target; runner overhead should be negligible relative to GPU runs and keep startup/validation steps fast (seconds, not minutes).  
**Constraints**: Deterministic experiment metadata (seed/config/provenance), no committed `tmp/` artifacts, follow repo layout  
**Scale/Scope**: Typical runs are a small cartesian product of modes × quant-pairs (default 2×2=4 scenarios); workspaces may be large but remain under `tmp/`, while expected snapshots are summary-only and small (per-scenario `summary.json`).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] **Pixi-first** Commands use `pixi run ...` (and `-e <env>` when required); no system Python assumptions.
- [x] **Quality gates planned** Lint/type/test commands are specified: `pixi run ruff check .`, `pixi run mypy .`, `pixi run pytest`.
- [x] **Testing strategy** Unit/integration/manual tests are planned appropriately; omissions are explicitly justified in `spec.md`.
- [x] **Reproducibility** Runs write resolved config, dataset provenance, seed, and code/version metadata.
- [x] **Artifact hygiene** `tmp/` remains uncommitted; curated summaries go under `models/*/reports/<run-id>/` when needed.
- [x] **Documentation** Feature docs are produced under `specs/003-refactor-qwen3-tutorial-runner/` (plan/research/data-model/quickstart/contracts).

Post-design re-check (2026-01-22): PASS.

## Project Structure

### Documentation (this feature)

```text
specs/003-refactor-qwen3-tutorial-runner/
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
└── ... (reusable modules)

scripts/
└── ... (experiment entrypoints; delegate to `src/auto_quantize_model/`)

conf/
└── ... (Hydra/OmegaConf configs when applicable)

models/
└── ... (model assets, bootstrap scripts, ONNX/export helpers, curated reports)

tests/
├── unit/
├── integration/
└── manual/
```

**Structure Decision**: Implement the shared tutorial-pack orchestration under `src/auto_quantize_model/qwen/` with a CLI module invoked by each tutorial pack’s `run_demo.sh`. Keep the tutorial packs under `docs/tutorial/howto/` as thin wrappers and add CPU-only tests under `tests/unit/` for scenario selection, snapshot/verify rules, and non-degeneracy gating.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
