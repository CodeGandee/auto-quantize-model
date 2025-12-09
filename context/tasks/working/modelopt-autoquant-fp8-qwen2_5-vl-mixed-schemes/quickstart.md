# Quickstart: ModelOpt AutoQuant FP8 schemes for Qwen2.5-VL-3B (LM-only)

This quickstart summarizes prerequisites, tools, and data locations for working on the FP8 AutoQuant mixed-precision schemes for the Qwen2.5-VL-3B-Instruct language model, as described in `context/plans/plan-modelopt-autoquant-fp8-qwen2_5-vl-mixed-schemes.md`.

## Prerequisites

- Qwen2.5-VL-3B-Instruct checkpoint bootstrapped via:
  - `models/qwen2_5_vl_3b_instruct/bootstrap.sh`
- LM-only FP8 baseline checkpoint available (or planned) under:
  - `models/qwen2_5_vl_3b_instruct/quantized/fp8_fp8_coco2017`
- `rtx5090` Pixi environment available and activated for all Python/ModelOpt work:
  - Use `pixi run -e rtx5090 ...` for any Python command.
- NVIDIA ModelOpt, TensorRT, and vLLM dependencies installed via Pixi (see `pyproject.toml` and ModelOpt docs).
- CUDA-visible GPU (RTX 5090 on this host) with enough VRAM to load Qwen2.5-VL-3B LM-only.

## Model and data

- Baseline HF snapshot (symlinked):
  - `models/qwen2_5_vl_3b_instruct/checkpoints/Qwen2.5-VL-3B-Instruct`
- Planned LM-only FP8 baseline (used as a reference scheme and/or starting point):
  - `models/qwen2_5_vl_3b_instruct/quantized/fp8_fp8_coco2017`
- Primary calibration text data (COCO captions):
  - `datasets/vlm-quantize-calib/coco2017_captions.txt`
- Optional image data for sanity checks:
  - `datasets/coco2017/source-data/val2017/`

## Core tools and scripts

- Sanity check script (baseline and quantized models):
  - `scripts/qwen/run_qwen2_5_vl_3b_sanity.py`
- Planned ModelOpt AutoQuant driver (to be implemented by subtasks):
  - `scripts/qwen/qwen2_5_vl_3b_autoquant_fp8_schemes.py`
- vLLM FP8 compatibility script (to be extended for schemes):
  - `scripts/qwen/run_qwen2_5_vl_3b_vllm_fp8.py`
- ModelOpt / TensorRT-Model-Optimizer sources (read-only reference):
  - `extern/TensorRT-Model-Optimizer/examples/llm_ptq/hf_ptq.py`
  - `extern/TensorRT-Model-Optimizer/modelopt/torch/quantization/model_quant.py`
  - `extern/TensorRT-Model-Optimizer/modelopt/torch/quantization/utils.py`

## Recommended command pattern

Always run Python via Pixi in the `rtx5090` env, for example:

```bash
pixi run -e rtx5090 python scripts/qwen/run_qwen2_5_vl_3b_sanity.py \
  --model-dir models/qwen2_5_vl_3b_instruct/checkpoints/Qwen2.5-VL-3B-Instruct
```

Once the AutoQuant driver exists, a typical invocation will look like:

```bash
pixi run -e rtx5090 python scripts/qwen/qwen2_5_vl_3b_autoquant_fp8_schemes.py \
  --scheme-name fp8_autoquant_top50 \
  --effective-bits 4.5 \
  --auto-quantize-method gradient \
  --calib-text datasets/vlm-quantize-calib/coco2017_captions.txt
```

