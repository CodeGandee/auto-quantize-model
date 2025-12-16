# Task: Qwen3-VL LM-only per-layer sensitivity (Hydra)

## What to do

- Implement a Hydra-configurable workflow to run **Qwen3‑VL LM-only** per-layer sensitivity analysis across weight/activation precision pairs: **W∈{FP4, FP8, INT8} × A∈{FP8, FP16}**, refactoring existing ModelOpt AutoQuant drivers in this repo into reusable library code and thin Hydra runners.

Related implementation plan (background + rationale): `context/plans/plan-qwen3-vl-lm-only-per-layer-sensitivity-hydra.md`.

## 1. Qwen3-VL LM-only per-layer sensitivity (Hydra)

Short description: Turn the existing Qwen3 sensitivity scripts into a Hydra-driven experiment that can sweep (W,A) precision pairs, produce reproducible per-layer sensitivity artifacts, and keep outputs organized under `models/qwen3_vl_4b_instruct/layer-analysis/`.

### Scope

- **In scope**:
  - LM-only extraction for Qwen3‑VL (vision tower kept higher precision).
  - ModelOpt AutoQuant sensitivity runs producing Markdown + JSON reports and a manifest.
  - Hydra config tree enabling easy sweeps over precision pairs and calibration subset sizes.
  - Refactors needed to move duplicated logic into `src/auto_quantize_model/`.
- **Out of scope (for this task)**:
  - “All-layers” (vision + LM) sensitivity and downstream serving integration (vLLM/TRT‑LLM), except where refactors must preserve existing behavior.
  - Large-scale benchmark suites; we focus on sensitivity artifacts and light sanity validation.

### Planned outputs

- A Hydra entry-point to run Qwen3‑VL **LM-only** sensitivity:
  - `scripts/qwen/qwen3_lm_sensitivity.py`
- A dedicated Hydra config composition for this workflow (model/dataset/quant-pair/autoquant/output):
  - `conf/...` additions under model, dataset, and new groups as needed.
- Shared, reusable implementation code (no more copy-paste across scripts):
  - `src/auto_quantize_model/...` helpers for Qwen LM-only sensitivity
- Reproducible artifacts per run (written to either `tmp/` or published under `models/.../layer-analysis/`):
  - `*_quant_manifest.json`, `per-layer-sensitivity.md`, `per-layer-sensitivity.json`
- Updated docs for regeneration and folder structure:
  - `models/qwen3_vl_4b_instruct/layer-analysis/README.md`

### Milestones (subtasks)

#### 1.1 Confirm precision-pair → ModelOpt config mapping

Goal: Decide the exact mapping for “FP4/FP8/INT8 weights” and “FP8/FP16 activations” to ModelOpt quantization configs (including any required custom configs), and document/encode that mapping so Hydra configs are unambiguous.

- Subtask spec: `context/tasks/working/qwen3-vl-lm-only-per-layer-sensitivity-hydra/subtask-001-101-confirm-precision-pair-mapping.md`

#### 1.2 Add Hydra config tree for Qwen3 LM-only sensitivity

Goal: Add `conf/` entries for the Qwen3 model, captions calibration subsets, AutoQuant defaults, and a user-facing “quant_pair” (or “format_set”) knob so we can sweep configurations without editing Python code.

- Subtask spec: `context/tasks/working/qwen3-vl-lm-only-per-layer-sensitivity-hydra/subtask-001-102-add-hydra-config-tree.md`

#### 1.3 Refactor existing Qwen sensitivity logic into shared library code

Goal: Extract/centralize model loading, LM-only extraction, calibration dataloaders, AutoQuant invocation, and report writing into `src/auto_quantize_model/` so both Hydra and legacy scripts can reuse it.

- Subtask spec: `context/tasks/working/qwen3-vl-lm-only-per-layer-sensitivity-hydra/subtask-001-103-refactor-to-shared-library.md`

#### 1.4 Implement the Hydra runner CLI for LM-only sensitivity

Goal: Provide a `@hydra.main` entry point that composes configs, runs the LM-only sensitivity pass, supports `report_only`, and writes artifacts to the configured output layout.

- Subtask spec: `context/tasks/working/qwen3-vl-lm-only-per-layer-sensitivity-hydra/subtask-001-104-implement-hydra-runner.md`

#### 1.5 Migrate existing Qwen3 sensitivity scripts/wrappers to Hydra

Goal: Keep current scripts working (or cleanly deprecate them) while switching the “official” workflow to Hydra, including a Hydra-based replacement for the 3-phase bash runner.

- Subtask spec: `context/tasks/working/qwen3-vl-lm-only-per-layer-sensitivity-hydra/subtask-001-105-migrate-existing-scripts.md`

#### 1.6 Validate, publish artifacts, add tests and docs

Goal: Regenerate the existing INT8 LM-only sensitivity outputs via Hydra for parity, run at least one new precision pair, publish artifacts under `models/.../layer-analysis/`, and add unit tests for non-GPU helpers.

- Subtask spec: `context/tasks/working/qwen3-vl-lm-only-per-layer-sensitivity-hydra/subtask-001-106-validate-and-document.md`

### TODOs

- [x] Job-001-101: Complete subtask 1.1: Confirm precision-pair mapping to ModelOpt configs and encode the mapping.
- [x] Job-001-102: Complete subtask 1.2: Add Hydra config tree (model/dataset/quant_pair/autoquant/output) for Qwen3 LM-only sensitivity.
- [x] Job-001-103: Complete subtask 1.3: Refactor duplicated Qwen sensitivity logic into shared library code under `src/auto_quantize_model/`.
- [x] Job-001-104: Complete subtask 1.4: Implement the Hydra runner CLI for LM-only sensitivity runs and sweeps.
- [x] Job-001-105: Complete subtask 1.5: Migrate existing scripts/wrappers to Hydra (or deprecate with pointers), including 3-phase orchestration.
- [x] Job-001-106: Complete subtask 1.6: Validate parity, publish artifacts, add unit tests, and update docs.
