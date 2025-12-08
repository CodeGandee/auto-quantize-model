# Quickstart: INC Op Sensitivity & W8A8 Spectrum for Qwen2.5-VL-3B-Instruct

This quickstart summarizes prerequisites, tools, and data locations for working on the Qwen2.5-VL-3B INC sensitivity and quantization spectrum tasks.

## Prerequisites

- Qwen2.5-VL-3B-Instruct checkpoint bootstrapped via:
  - `models/qwen2_5_vl_3b_instruct/bootstrap.sh`
- `rtx5090` Pixi environment available and activated for all Python/INC work:
  - Use `pixi run -e rtx5090 ...` for any Python command.
- Intel Neural Compressor installed in the `rtx5090` env (already handled via `pyproject.toml` / Pixi).
- CUDA-visible GPU (RTX 5090 on this host) with enough VRAM to load Qwen2.5-VL-3B.

## Model location

- Baseline HF snapshot (symlinked):
  - `models/qwen2_5_vl_3b_instruct/checkpoints/Qwen2.5-VL-3B-Instruct`
- This directory should contain the usual HF files:
  - `config.json`, `generation_config.json`, tokenizer files, `model-*.safetensors`, etc.

## Dataset / calibration data

- Primary calibration/text data (COCO captions):
  - `datasets/vlm-quantize-calib/coco2017_captions.txt`
    - Built from COCO2017 image caption annotations.
- COCO 2017 image data (for image+text sanity or optional vision-aware calibration):
  - `datasets/coco2017/source-data/train2017/`
  - `datasets/coco2017/source-data/val2017/`

## Core tools and scripts

- Sanity check script (baseline and quantized models):
  - `scripts/qwen/run_qwen2_5_vl_3b_sanity.py`
- Planned INC drivers (to be implemented by subtasks):
  - `scripts/qwen/inc_qwen2_5_vl_3b_sensitivity.py`
  - `scripts/qwen/inc_qwen2_5_vl_3b_quantize.py`
- INC source code is vendored under:
  - `extern/neural-compressor/`
  - You can read this code for reference or debugging (e.g., to understand `mse_v2`, `hawq_v2`, or adaptor internals), but **do not import or depend on it directly** in this repo; use the installed `neural_compressor` package APIs instead.

## Recommended command pattern

- Always run Python via Pixi in the `rtx5090` env, for example:

```bash
pixi run -e rtx5090 python scripts/qwen/run_qwen2_5_vl_3b_sanity.py \
  --model-dir models/qwen2_5_vl_3b_instruct/checkpoints/Qwen2.5-VL-3B-Instruct
```
