# Qwen3-VL-4B AutoQuant All-Layers Sensitivity

## HEADER
- **Purpose**: Run ModelOpt AutoQuant all-layer sensitivity analysis for Qwen3-VL-4B-Instruct.
- **Status**: Draft
- **Date**: 2025-12-11
- **Dependencies**: Pixi `rtx5090-vllm` env, NVIDIA ModelOpt, COCO2017 VLM calib DB.
- **Target**: AI assistants and developers

## Overview

This helper subdirectory contains a driver that runs NVIDIA ModelOpt AutoQuant on
the Qwen3-VL-4B-Instruct model to produce **per-layer sensitivity** signals. The
output is a JSON manifest plus a Markdown table summarizing how sensitive each
layer is to quantization under an effective-bits constraint.

- Model: `models/qwen3_vl_4b_instruct/checkpoints/Qwen3-VL-4B-Instruct`
- Method: `modelopt.torch.quantization.auto_quantize` with custom configs
  (`FP8_ALL_LAYERS_CFG`, `INT8_ALL_LAYERS_CFG`) from
  `src/auto_quantize_model/modelopt_configs.py`.
- Calibration: image+text COCO2017 samples from `datasets/vlm-quantize-calib/coco2017_vlm_calib_large.db`
  and `datasets/coco2017/source-data`, processed via `AutoProcessor` and `qwen_vl_utils`.
- Output: `tmp/qwen3_vl_4b_autoquant_all_layers_fp8/` (for FP8) or a
  user-specified directory for INT8 runs.

## Output layout

By default the FP8 driver writes into:

- `tmp/qwen3_vl_4b_autoquant_all_layers_fp8/`
  - `fp8_autoquant_all_layers_fp8_autoquant_state.pt` — raw AutoQuant state dict.
  - `fp8_autoquant_all_layers_fp8_quant_manifest.json` — per-layer quantization + sensitivity.
  - `per-layer-sensitivity.md` — human-readable report built from the manifest.

You can change the output directory via `--output-dir`, and use a different
directory for INT8 runs if desired.

## Usage

Run the all-layers AutoQuant sensitivity pass from the repo root (FP8 by default):

```bash
pixi run -e rtx5090-vllm python \
  models/qwen3_vl_4b_instruct/helpers/qwen3_vl_4b_autoquant_all_layers/run_qwen3_vl_4b_autoquant_all_layers.py
```

To run an INT8 (W8A8) all-layers sensitivity pass:

```bash
pixi run -e rtx5090-vllm python \
  models/qwen3_vl_4b_instruct/helpers/qwen3_vl_4b_autoquant_all_layers/run_qwen3_vl_4b_autoquant_all_layers.py \
  --quant-format int8 \
  --output-dir tmp/qwen3_vl_4b_autoquant_all_layers_int8_large
```

To regenerate only the Markdown report from an existing manifest:

```bash
pixi run -e rtx5090-vllm python \
  models/qwen3_vl_4b_instruct/helpers/qwen3_vl_4b_autoquant_all_layers/run_qwen3_vl_4b_autoquant_all_layers.py \
  --report-only \
  --output-dir tmp/qwen3_vl_4b_autoquant_all_layers_fp8
```
