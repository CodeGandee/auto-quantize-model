# Data Model: Revise Qwen3-VL-8B Tutorial Pack (Introduce + Layer Sensitivity)

**Branch**: `002-revise-qwen3-vl-tutorial`  
**Date**: 2026-01-21  
**Spec**: `/data1/huangzhe/code/auto-quantize-model/specs/002-revise-qwen3-vl-tutorial/spec.md`

## Overview

This feature is documentation- and workflow-centric. The “data model” describes the
entities and artifacts produced by the tutorial pack runner and used for verification.

## Entities

### Tutorial Pack

Represents the self-contained tutorial directory:

- **Path**: `/data1/huangzhe/code/auto-quantize-model/docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/`
- **Contains**:
  - `inputs/` (minimal tracked inputs)
  - `expected_report/` (sanitized, tracked golden outputs)
  - `run_demo.sh` (orchestrator)
  - `scripts/` (sanitization + summarization helpers)

### Checkpoint Link

A local, repo-internal link to a user-provided model snapshot:

- **Path**: `/data1/huangzhe/code/auto-quantize-model/models/qwen3_vl_8b_instruct/checkpoints/Qwen3-VL-8B-Instruct`
- **Rules**:
  - Must exist before running scenarios.
  - Must not be committed (it is a local filesystem link).

### Dataset Preset

Named preset used to select calibration inputs and budgets:

- **Name**: `small` | `medium` | `large`
- **Inputs**:
  - **COCO root**: `/data1/huangzhe/code/auto-quantize-model/datasets/coco2017/source-data`
  - **VLM DB**: `/data1/huangzhe/code/auto-quantize-model/datasets/vlm-quantize-calib/coco2017_vlm_calib_<size>.db`
  - **Captions**: `/data1/huangzhe/code/auto-quantize-model/datasets/vlm-quantize-calib/coco2017_captions_<size>.txt`
- **Budgets (default mapping)**:
  - `small`: max_calib_samples=16
  - `medium`: max_calib_samples=128
  - `large`: max_calib_samples=512

### Quant Pair (Worked Example)

Defines the quantization configuration used by a scenario:

- **Name**: `wint4_afp16` | `wint4_aint8`
- **Source**:
  - `/data1/huangzhe/code/auto-quantize-model/conf/quant_pair/wint4_afp16.yaml`
  - `/data1/huangzhe/code/auto-quantize-model/conf/quant_pair/wint4_aint8.yaml`
- **Key attributes**:
  - weight precision category (int4)
  - activation precision category (fp16 or int8)
  - quantization format identifier (format_name)

### Run Scenario

Single scenario defined by:

- **mode**: `all_layers` | `lm_only`
- **quant_pair**: `wint4_afp16` | `wint4_aint8`
- **dataset_preset**: `small` | `medium` | `large`
- **device**: user-selected compute device (e.g., CUDA device)
- **auto-quantize score size**: default 128

### Run Workspace

Temporary directory where a run writes full artifacts:

- **Root path**: `/data1/huangzhe/code/auto-quantize-model/tmp/…`
- **Contents**:
  - per-scenario output dirs
  - logs
  - generated sanitized summaries

### Scenario Output

Per-scenario output directory containing:

- **Manifest JSON** (the source-of-truth for summarization and verification)
- **Human-readable sensitivity report** (Markdown)
- **Machine-readable sensitivity report** (JSON)
- **Run metadata** (resolved config, dataset metadata)

### Sanitized Summary

Small, stable artifact used for verification:

- `summary.json` (stable keys, scenario metadata, and non-degeneracy indicators)
- `summary.md` (human-readable rendering of the stable keys)

### Expected Report Snapshot

Tracked golden outputs used for verification:

- **Path**: `/data1/huangzhe/code/auto-quantize-model/docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/expected_report/`
- **Structure**: per-scenario directories containing sanitized summaries and minimal sanitized artifacts.

## Relationships

- Tutorial Pack **produces** a Run Workspace.
- Run Workspace **contains** Scenario Outputs and Sanitized Summaries.
- Each Scenario Output **belongs to** exactly one Run Scenario.
- Each Run Scenario **references** exactly one Dataset Preset and one Quant Pair.
- Expected Report Snapshot **contains** golden Sanitized Summaries for each Run Scenario.

## Validation Rules (testable)

- Default run uses `dataset_preset=medium` and executes all four scenarios.
- If any required asset is missing (checkpoint link or datasets), the run fails fast with an actionable error.
- Each scenario’s sanitized summary includes:
  - mode, quant_pair, dataset_preset
  - effective calibration settings (seq_len, batch_size, num_calib_batches, max/num samples)
  - score_size
- Verification diffs only the sanitized summaries for each scenario.
- Non-degeneracy: for each scenario, at least one layer sensitivity value is not exactly `0.0`.

## State Transitions

- **Planned** → scenario parameters are selected (defaults or user overrides).
- **Executed** → scenario produces raw artifacts in the workspace.
- **Summarized** → stable sanitized summaries are generated from the manifest.
- **Verified** → summaries match the expected snapshot (or)
- **Snapshotted** → expected snapshot is refreshed intentionally via snapshot mode.
