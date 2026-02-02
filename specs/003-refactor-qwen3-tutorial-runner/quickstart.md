# Quickstart: Refactor Qwen3-VL Tutorial Pack Runner (Shared Utility)

**Branch**: `003-refactor-qwen3-tutorial-runner`  
**Date**: 2026-01-22  
**Spec**: `/data1/huangzhe/code/auto-quantize-model/specs/003-refactor-qwen3-tutorial-runner/spec.md`

## Prerequisites

From `/data1/huangzhe/code/auto-quantize-model`:

1) Install the Pixi environment:

```bash
cd /data1/huangzhe/code/auto-quantize-model
pixi install
```

2) Provide local Qwen3-VL model snapshots and create repo checkpoint links (runner does not auto-create links):

- Qwen3-VL-4B:
  - `/data1/huangzhe/code/auto-quantize-model/models/qwen3_vl_4b_instruct/bootstrap.sh`
- Qwen3-VL-8B:
  - `/data1/huangzhe/code/auto-quantize-model/models/qwen3_vl_8b_instruct/bootstrap.sh`

Example (interactive, will propose a candidate path based on env/YAML):

```bash
cd /data1/huangzhe/code/auto-quantize-model
./models/qwen3_vl_4b_instruct/bootstrap.sh
./models/qwen3_vl_8b_instruct/bootstrap.sh
```

3) Provide a local COCO2017 dataset and create the repo dataset link:

- Expected link path:
  - `/data1/huangzhe/code/auto-quantize-model/datasets/coco2017/source-data`

Example:

```bash
cd /data1/huangzhe/code/auto-quantize-model
./datasets/coco2017/bootstrap.sh --path /absolute/path/to/coco2017
```

4) Ensure repo calibration assets exist (tracked in-repo):

- `/data1/huangzhe/code/auto-quantize-model/datasets/vlm-quantize-calib/coco2017_vlm_calib_{small,medium,large}.db`
- `/data1/huangzhe/code/auto-quantize-model/datasets/vlm-quantize-calib/coco2017_captions_{small,medium,large}.txt`

## Run the tutorial packs (verify mode)

From repo root:

```bash
cd /data1/huangzhe/code/auto-quantize-model

# Qwen3-VL-4B tutorial pack
bash /data1/huangzhe/code/auto-quantize-model/docs/tutorial/howto/tut-qwen3-vl-4b-layer-sensitivity/run_demo.sh

# Qwen3-VL-8B tutorial pack
bash /data1/huangzhe/code/auto-quantize-model/docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/run_demo.sh
```

Expected behavior (after this feature is implemented):

- Both packs delegate orchestration to the shared runner but keep the same flags.
- Verification fails if `expected_report/` is missing/incomplete.
- Verification diffs only `summary.json` per scenario and enforces non-degeneracy.
- Markdown reports are kept for all scenarios at `outputs/<mode>/<quant_pair>/layer-sensitivity-report.md` (with the tutorial summary table prepended).

## Subset runs (for iteration)

```bash
cd /data1/huangzhe/code/auto-quantize-model

# All-layers only (both quant pairs).
bash docs/tutorial/howto/tut-qwen3-vl-4b-layer-sensitivity/run_demo.sh --modes all_layers

# LM-only only, one quant pair.
bash docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/run_demo.sh --modes lm_only --quant-pairs wint4_afp16
```

## Refresh expected outputs (snapshot mode)

Snapshot mode refreshes sanitized per-scenario outputs under `expected_report/outputs/` (always `summary.json`, plus optional artifacts like the canonical `layer-sensitivity-report.md`).

```bash
cd /data1/huangzhe/code/auto-quantize-model
bash docs/tutorial/howto/tut-qwen3-vl-4b-layer-sensitivity/run_demo.sh --snapshot-report
bash docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/run_demo.sh --snapshot-report
```

## Quality gates (for maintainers)

```bash
cd /data1/huangzhe/code/auto-quantize-model
pixi run ruff check .
pixi run mypy .
pixi run pytest
```
