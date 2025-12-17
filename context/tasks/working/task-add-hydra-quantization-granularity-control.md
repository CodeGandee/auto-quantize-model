# Task: Add Hydra quantization granularity control (ModelOpt AutoQuant)

## What to do

- Extend the Qwen3 Hydra sensitivity workflow to support **quantization granularity controls** for both weights and activations (e.g., per-channel/per-column/per-group/per-token), so we can grid-search combinations like `(W dtype, A dtype, granularity)` while keeping outputs reproducible and non-colliding.

Related design / background:
- `context/design/qwen3-modelopt-production-options.md`
- `context/design/qwen3-target-recipe.md`
- `context/hints/about-modelopt-quantization-granularity.md`
- Main implementation plan: `context/plans/plan-add-hydra-quantization-granularity-control.md`

## 1. Hydra quantization granularity control

Short description: Add a Hydra `quant_granularity` config group and the code plumbing to safely overlay granularity settings (`axis` / `block_sizes` / `type`) onto resolved ModelOpt quantization formats, then update naming, tests, and docs so multirun sweeps are straightforward.

### Scope

- **In scope**:
  - Hydra config support for quantization granularity overrides for `*weight_quantizer` and `*input_quantizer` (and `default` where needed).
  - Safe, ModelOpt-compatible overlay semantics (axis vs block_sizes mutual exclusion, key normalization).
  - Runner + artifact naming changes to avoid collisions in `hydra -m` sweeps.
  - Lightweight unit tests for pure config/name functions.
  - Documentation and example grid-search commands (Pixi env: `rtx5090-vllm`).
- **Out of scope (for this task)**:
  - Implementing GPTQ (ModelOpt AutoQuant is not GPTQ).
  - Adding new quantization methods/algorithms beyond config overlays.
  - Guaranteeing every combination works across all ModelOpt builds; we will fail fast on unsupported configs and document known limitations.

### Planned outputs

- Hydra config group: `conf/quant_granularity/*.yaml`
- Overlay helper for ModelOpt quant configs (deep-copy + normalized merge)
- Qwen3 Hydra runner integration: `scripts/qwen/qwen3_lm_sensitivity.py`
- Collision-free output naming for tmp/publish paths (including granularity)
- Unit tests under `tests/unit/` for overlay + naming helpers
- Updated docs with sweep examples: `models/qwen3_vl_4b_instruct/layer-analysis/README.md`

### Milestones (subtasks)

#### 1.1 Define granularity vocabulary and ModelOpt mapping

Goal: Decide the canonical user-facing granularity names (weights + activations) and document the exact ModelOpt mapping (`axis`, `block_sizes`, `type`) including the Qwen3 “recipe match” target.

- Subtask spec: `context/tasks/working/add-hydra-quantization-granularity-control/subtask-001-101-define-granularity-vocabulary.md`

#### 1.2 Add Hydra `quant_granularity` config group (initial overlays)

Goal: Implement the Hydra configuration surface area (`conf/quant_granularity/`) with a stable schema and a small initial set of named overlay options (default, recipe match, per-channel/per-column, per-group sizes).

- Subtask spec: `context/tasks/working/add-hydra-quantization-granularity-control/subtask-001-102-add-hydra-quant-granularity-configs.md`

#### 1.3 Implement ModelOpt quant_cfg overlay + normalization helper

Goal: Implement a small helper that deep-copies a resolved ModelOpt config dict and merges `quant_cfg_overrides` safely (axis/block_sizes exclusivity, `block_sizes` key normalization, optional `default` propagation).

- Subtask spec: `context/tasks/working/add-hydra-quantization-granularity-control/subtask-001-103-implement-quant-cfg-overlays.md`

#### 1.4 Wire granularity overlays into the Qwen3 Hydra runner + manifest

Goal: Apply the overlay in `scripts/qwen/qwen3_lm_sensitivity.py` so AutoQuant uses the effective config dict, and record base format + overrides in manifests for reproducibility.

- Subtask spec: `context/tasks/working/add-hydra-quantization-granularity-control/subtask-001-104-wire-overlays-into-hydra-runner.md`

#### 1.5 Update tmp/publish output naming for collision-free sweeps

Goal: Ensure both Hydra run/sweep directories and publish directories include the granularity identifier so `hydra -m` grid-searches don’t overwrite artifacts.

- Subtask spec: `context/tasks/working/add-hydra-quantization-granularity-control/subtask-001-105-update-output-naming-for-sweeps.md`

#### 1.6 Add unit tests, docs, and smoke multirun examples

Goal: Add unit tests for overlay + naming helpers, update the layer-analysis README with `quant_granularity` usage, and run a tiny smoke sweep in the `rtx5090-vllm` environment.

- Subtask spec: `context/tasks/working/add-hydra-quantization-granularity-control/subtask-001-106-add-tests-docs-and-smoke-sweeps.md`

### TODOs

- [ ] Job-001-101: Complete subtask 1.1: Define granularity vocabulary and ModelOpt mapping (including Qwen3 recipe match).
- [ ] Job-001-102: Complete subtask 1.2: Add Hydra `quant_granularity` config group with initial overlay options.
- [ ] Job-001-103: Complete subtask 1.3: Implement ModelOpt quant_cfg overlay + normalization helper.
- [ ] Job-001-104: Complete subtask 1.4: Wire overlays into the Qwen3 Hydra runner and record them in manifests.
- [ ] Job-001-105: Complete subtask 1.5: Update output naming for collision-free multirun sweeps.
- [ ] Job-001-106: Complete subtask 1.6: Add unit tests, docs, and smoke multirun examples.

