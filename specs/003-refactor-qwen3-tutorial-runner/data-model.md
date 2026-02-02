# Data Model: Refactor Qwen3-VL Tutorial Pack Runner (Shared Utility)

**Branch**: `003-refactor-qwen3-tutorial-runner`  
**Date**: 2026-01-22  
**Spec**: `/data1/huangzhe/code/auto-quantize-model/specs/003-refactor-qwen3-tutorial-runner/spec.md`

## Overview

This feature is workflow-centric. The “data model” describes the entities and filesystem
artifacts used by the shared tutorial-pack runner and its snapshot/verification logic.

## Entities

### Tutorial Pack

Represents one tutorial directory under:

- `/data1/huangzhe/code/auto-quantize-model/docs/tutorial/howto/`

For this feature, the primary packs are:

- `/data1/huangzhe/code/auto-quantize-model/docs/tutorial/howto/tut-qwen3-vl-4b-layer-sensitivity/`
- `/data1/huangzhe/code/auto-quantize-model/docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/`

Key attributes:

- `pack_name` (stable identifier used in workspace naming)
- `expected_report_dir` (pack-local expected snapshot directory)
- `run_demo.sh` wrapper (user entrypoint; delegates to shared runner)

### Shared Runner

The shared orchestration utility (importable code + CLI frontend) that:

- Validates assets (checkpoint + datasets)
- Enumerates scenarios (modes × quant-pairs)
- Creates a workspace and per-scenario directories
- Produces per-scenario sanitized summaries
- Snapshots and/or verifies per-scenario summaries

### Model Configuration

The pack/model-specific configuration needed to locate the checkpoint and run workflows.

Key attributes:

- `model_id` (user-selected identifier, e.g. a Qwen3-VL variant key)
- `checkpoint_link_path` (repo path expected to exist before running)
- (Optional) model-specific runner parameters (e.g., mode-specific identifiers)

### Checkpoint Link

A repo-internal path (often a symlink) pointing to a user-provided model snapshot.

Rules:

- Must exist before scenario execution.
- Must not be auto-created by the runner (runner fails with instructions).
- Must not be committed (local filesystem state).

### Dataset Preset

Named preset used to select calibration inputs and budgets.

Key attributes:

- `dataset_size`: `small` | `medium` | `large`
- Required asset paths:
  - COCO root: `/data1/huangzhe/code/auto-quantize-model/datasets/coco2017/source-data`
  - VLM calib DB: `/data1/huangzhe/code/auto-quantize-model/datasets/vlm-quantize-calib/coco2017_vlm_calib_<size>.db`
  - Captions: `/data1/huangzhe/code/auto-quantize-model/datasets/vlm-quantize-calib/coco2017_captions_<size>.txt`

### Quant Pair

Defines the quantization configuration used by a scenario.

Key attributes:

- `quant_pair` string (e.g., `wint4_afp16`)
- Must be validated against the allowed/known set for the tutorial (failure is user-actionable).

### Scenario

Single scenario defined by the cartesian product of:

- `mode`: `all_layers` | `lm_only`
- `quant_pair`: user-selected values
- `dataset_size`: `small` | `medium` | `large`
- `device`: user-selected compute device

Derived attributes:

- `scenario_id = "{mode}/{quant_pair}"`

### Run Workspace

Temporary directory under:

- `/data1/huangzhe/code/auto-quantize-model/tmp/`

Contains:

- `outputs/<mode>/<quant_pair>/` (raw runner outputs + logs)
- `outputs/<mode>/<quant_pair>/summary.json` (generated stable summary for snapshot/verify)

### Sanitized Summary

Per-scenario stable artifacts used for verification:

- `summary.json`

Key attributes captured in `summary.json`:

- Scenario identity (`scenario_id`, `mode`, `quant_pair`)
- Dataset preset and effective calibration budgets/settings
- Non-degeneracy indicator (`has_nonzero_sensitivity`)

### Expected Report Snapshot

Tracked golden outputs used for verification.

Path:

- `<tutorial-pack>/expected_report/`

Structure:

- `expected_report/outputs/<mode>/<quant_pair>/summary.json`
- Optional sanitized artifacts (for inspection/documentation), for example:
  - `expected_report/outputs/<mode>/<quant_pair>/layer-sensitivity-report.json`
  - `expected_report/outputs/all_layers/wint4_afp16/layer-sensitivity-report.md`
  - `expected_report/outputs/<mode>/<quant_pair>/quant_manifest.json`
  - `expected_report/outputs/<mode>/<quant_pair>/composed-config.yaml`

## Relationships

- A Tutorial Pack **delegates** execution to the Shared Runner.
- A Model Configuration **provides** the checkpoint path and model identifiers for scenario execution.
- A Dataset Preset **provides** required asset paths and calibration budgets.
- A Scenario **references** exactly one Dataset Preset and one Quant Pair.
- A Run Workspace **contains** per-scenario outputs and per-scenario sanitized summaries.
- The Expected Report Snapshot **contains** golden sanitized summaries for each Scenario.

## Validation Rules (testable)

- Scenario enumeration is exactly `selected_modes × selected_quant_pairs`.
- Scenario ID is stable: `"{mode}/{quant_pair}"`.
- Snapshot mode writes sanitized outputs per scenario under `expected_report/outputs/` (at minimum `summary.json`).
- Verify mode:
  - fails if the expected snapshot is missing/incomplete,
  - diffs only `summary.json`,
  - enforces non-degeneracy (`has_nonzero_sensitivity` must be true).
- Multi-scenario runs are fail-fast: stop on the first failing scenario and report failure.
- The runner must not auto-create checkpoint links; missing checkpoint paths are a hard error with instructions.

## State Transitions

- **Planned** → scenario parameters selected (defaults or user overrides).
- **Validated** → checkpoint and dataset assets checked; run stops here on missing assets.
- **Executed** → scenario produces raw outputs under the workspace.
- **Summarized** → per-scenario `summary.json` produced.
- **Verified** → summaries match expected snapshots (verify mode), or
- **Snapshotted** → expected snapshots refreshed (snapshot mode).
- **Failed** → any failure halts the run immediately (fail-fast).
