# Quickstart: Revise Qwen3-VL-8B Tutorial Pack (Introduce + Layer Sensitivity)

**Branch**: `002-revise-qwen3-vl-tutorial`  
**Date**: 2026-01-21  
**Spec**: `/data1/huangzhe/code/auto-quantize-model/specs/002-revise-qwen3-vl-tutorial/spec.md`

## Prerequisites

From `/data1/huangzhe/code/auto-quantize-model`:

1) Install the Pixi environment:

```bash
cd /data1/huangzhe/code/auto-quantize-model
pixi install
```

2) Provide a local Qwen3-VL-8B snapshot and create the repo checkpoint link:

- Expected link path:
  - `/data1/huangzhe/code/auto-quantize-model/models/qwen3_vl_8b_instruct/checkpoints/Qwen3-VL-8B-Instruct`

Example (local snapshot path will vary):

```bash
cd /data1/huangzhe/code/auto-quantize-model
ln -s /absolute/path/to/Qwen3-VL-8B-Instruct \
  models/qwen3_vl_8b_instruct/checkpoints/Qwen3-VL-8B-Instruct
```

3) Provide a local COCO2017 dataset and create the repo dataset link:

- Expected link path:
  - `/data1/huangzhe/code/auto-quantize-model/datasets/coco2017/source-data`

Example:

```bash
cd /data1/huangzhe/code/auto-quantize-model
datasets/coco2017/bootstrap.sh --path /absolute/path/to/coco2017
```

4) Ensure repo calibration assets exist (tracked in-repo):

- `/data1/huangzhe/code/auto-quantize-model/datasets/vlm-quantize-calib/coco2017_vlm_calib_{small,medium,large}.db`
- `/data1/huangzhe/code/auto-quantize-model/datasets/vlm-quantize-calib/coco2017_captions_{small,medium,large}.txt`

## Run the tutorial pack (verify mode)

From repo root:

```bash
cd /data1/huangzhe/code/auto-quantize-model
bash /data1/huangzhe/code/auto-quantize-model/docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/run_demo.sh
```

Expected behavior (after this feature is implemented):

- Default executes and verifies all **4 scenarios** (2 modes × 2 worked quant pairs).
- Default uses the **medium** dataset preset.
- Verification diffs only the sanitized summaries against `expected_report/`.

## Subset runs (for iteration)

```bash
cd /data1/huangzhe/code/auto-quantize-model

# All-layers only (both quant pairs).
bash docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/run_demo.sh --modes all_layers

# LM-only only, one quant pair.
bash docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/run_demo.sh --modes lm_only --quant-pairs wint4_afp16
```

## Refresh expected outputs (snapshot mode)

```bash
cd /data1/huangzhe/code/auto-quantize-model
bash /data1/huangzhe/code/auto-quantize-model/docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/run_demo.sh --snapshot-report
```

## Quality gates (for maintainers)

```bash
cd /data1/huangzhe/code/auto-quantize-model
pixi run ruff check .
pixi run mypy .
pixi run pytest
```
