# Implementation Plan: Revise Qwen3-VL-8B Tutorial Pack (Introduce + Layer Sensitivity)

**Branch**: `[002-revise-qwen3-vl-tutorial]` | **Date**: 2026-01-21 | **Spec**: `/data1/huangzhe/code/auto-quantize-model/specs/002-revise-qwen3-vl-tutorial/spec.md`  
**Input**: Feature specification from `/data1/huangzhe/code/auto-quantize-model/specs/002-revise-qwen3-vl-tutorial/spec.md`

## Summary

Revise the existing tutorial pack at `/data1/huangzhe/code/auto-quantize-model/docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/`
so that it runs end-to-end in the default Pixi environment and produces **non-degenerate**
layer sensitivity reports for both **all-layers** and **LM-only** modes.

Default behavior:

- Execute and verify **4 scenarios** (2 modes × 2 worked quantization examples).
- Use the **medium** dataset preset by default (seq_len=512, batch_size=8, num_calib_batches=16, score_size=128).
- Fail fast with clear instructions when required model or dataset assets are missing.
- Verify by diffing **only sanitized summaries** against `expected_report/`.

## Technical Context

**Language/Version**: Python 3.12 (Pixi-managed; `pyproject.toml` pins `python = "3.12.*"`)  
**Primary Dependencies**: PyTorch, Hydra/OmegaConf, ONNX/onnxruntime, TensorBoard + matplotlib  
**Storage**: Workspace artifacts under `/data1/huangzhe/code/auto-quantize-model/tmp/` (uncommitted) + sanitized snapshots under the tutorial pack’s `expected_report/` (committed)  
**Testing**: `pytest` (unit/integration) + manual GPU runs for end-to-end tutorial validation  
**Target Platform**: Linux + CUDA GPU (CPU permitted but potentially extremely slow)  
**Project Type**: Tutorial pack (docs + bash orchestrator) that calls Python drivers (library code under `/data1/huangzhe/code/auto-quantize-model/src/auto_quantize_model/`)  
**Performance Goals**: No strict runtime limit for the default medium run; prioritize correctness and meaningful (non-degenerate) outputs.  
**Constraints**: Pixi-first commands, deterministic run metadata, do not commit `/data1/huangzhe/code/auto-quantize-model/tmp/` artifacts, keep expected snapshots sanitized and reviewable.  
**Scale/Scope**: 4 scenarios per default run; medium preset uses 128 samples (16 batches × 8) and score_size=128; expected snapshots include only sanitized summaries + minimal sanitized artifacts needed for troubleshooting.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] **Pixi-first** Commands use `pixi run ...` (and `-e <env>` when required); no system Python assumptions.
- [x] **Quality gates planned** Lint/type/test commands are specified: `pixi run ruff check .`, `pixi run mypy .`, `pixi run pytest`.
- [x] **Testing strategy** Unit tests cover summary/verification logic; end-to-end GPU validation remains a manual run documented in the tutorial pack.
- [x] **Reproducibility** Runs record resolved config and dataset provenance inside per-scenario manifests/summaries.
- [x] **Artifact hygiene** `/data1/huangzhe/code/auto-quantize-model/tmp/` remains uncommitted; only sanitized artifacts are committed under the tutorial pack’s `expected_report/`.
- [x] **Documentation** Feature docs are produced under `/data1/huangzhe/code/auto-quantize-model/specs/002-revise-qwen3-vl-tutorial/` (plan/research/data-model/quickstart/contracts).

**Gate Result**: PASS (pre-research and post-design)

## Project Structure

### Documentation (this feature)

```text
/data1/huangzhe/code/auto-quantize-model/specs/002-revise-qwen3-vl-tutorial/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (data/CLI contracts)
└── tasks.md             # Phase 2 output (`/speckit.tasks` - NOT created here)
```

### Source Code (repository root)
```text
/data1/huangzhe/code/auto-quantize-model/src/auto_quantize_model/
└── ... (reusable modules)

/data1/huangzhe/code/auto-quantize-model/scripts/
└── ... (experiment entrypoints; delegate to `src/auto_quantize_model/`)

/data1/huangzhe/code/auto-quantize-model/conf/
└── ... (Hydra/OmegaConf configs when applicable)

/data1/huangzhe/code/auto-quantize-model/models/
└── ... (model assets, bootstrap scripts, ONNX/export helpers, curated reports)

/data1/huangzhe/code/auto-quantize-model/tests/
├── unit/
├── integration/
└── manual/
```

**Structure Decision**: Keep all planning artifacts in `/data1/huangzhe/code/auto-quantize-model/specs/002-revise-qwen3-vl-tutorial/`. Implement tutorial changes in-place under `/data1/huangzhe/code/auto-quantize-model/docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/`, reusing existing drivers under `/data1/huangzhe/code/auto-quantize-model/models/qwen3_vl_4b_instruct/helpers/` and `/data1/huangzhe/code/auto-quantize-model/scripts/qwen/` where feasible. Add any shared logic under `/data1/huangzhe/code/auto-quantize-model/src/auto_quantize_model/` and validate summary/verification behavior with unit tests under `/data1/huangzhe/code/auto-quantize-model/tests/unit/`.

## Complexity Tracking

No constitution violations anticipated for this change set.
