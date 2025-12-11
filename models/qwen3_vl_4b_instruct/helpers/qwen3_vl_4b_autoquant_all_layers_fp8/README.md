# Qwen3-VL-4B AutoQuant All-Layers FP8 Sensitivity

## HEADER
- **Purpose**: Run ModelOpt AutoQuant FP8 all-layer sensitivity analysis for Qwen3-VL-4B-Instruct.
- **Status**: Draft
- **Date**: 2025-12-11
- **Dependencies**: Pixi `rtx5090-vllm` env, NVIDIA ModelOpt, COCO2017 VLM calib DB.
- **Target**: AI assistants and developers

## Overview

This helper subdirectory contains a driver that runs NVIDIA ModelOpt AutoQuant FP8 on
the Qwen3-VL-4B-Instruct model to produce **per-layer sensitivity** signals. The output
is a JSON manifest plus a Markdown table summarizing how sensitive each layer is to FP8
quantization under an effective-bits constraint.

- Model: `models/qwen3_vl_4b_instruct/checkpoints/Qwen3-VL-4B-Instruct`
- Method: `modelopt.torch.quantization.auto_quantize` with a custom `FP8_ALL_LAYERS_CFG`
  config from `src/auto_quantize_model/modelopt_configs.py`.
- Calibration: image+text COCO2017 samples from `datasets/vlm-quantize-calib/coco2017_vlm_calib.db`
  and `datasets/coco2017/source-data`, processed via `AutoProcessor` and `qwen_vl_utils`.
- Output: `tmp/qwen3_vl_4b_autoquant_all_layers_fp8/`

## Output layout

By default the driver writes into:

- `tmp/qwen3_vl_4b_autoquant_all_layers_fp8/`
  - `fp8_autoquant_all_layers_fp8_autoquant_state.pt` — raw AutoQuant state dict.
  - `fp8_autoquant_all_layers_fp8_quant_manifest.json` — per-layer quantization + sensitivity.
  - `per-layer-sensitivity.md` — human-readable report built from the manifest.

You can change the output directory via `--output-dir`.

## Usage

Run the all-layers FP8 AutoQuant sensitivity pass from the repo root:

```bash
pixi run -e rtx5090-vllm python \
  models/qwen3_vl_4b_instruct/helpers/qwen3_vl_4b_autoquant_all_layers_fp8/run_qwen3_vl_4b_autoquant_all_layers_fp8.py
```

To regenerate only the Markdown report from an existing manifest:

```bash
pixi run -e rtx5090-vllm python \
  models/qwen3_vl_4b_instruct/helpers/qwen3_vl_4b_autoquant_all_layers_fp8/run_qwen3_vl_4b_autoquant_all_layers_fp8.py \
  --report-only \
  --output-dir tmp/qwen3_vl_4b_autoquant_all_layers_fp8
```

