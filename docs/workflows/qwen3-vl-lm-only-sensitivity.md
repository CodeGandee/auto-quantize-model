# Qwen3-VL LM-only per-layer sensitivity

This workflow runs **per-layer sensitivity analysis on the language model (LM) only** for **Qwen3-VL-4B-Instruct**, leaving the vision tower unquantized by default.

The Hydra entry point is:

- `scripts/qwen/qwen3_lm_sensitivity.py`

## Prerequisites

- Qwen3-VL checkpoint exists at `models/qwen3_vl_4b_instruct/checkpoints/Qwen3-VL-4B-Instruct` (see `models/qwen3_vl_4b_instruct/bootstrap.sh`).
- Captions calibration files exist under `datasets/vlm-quantize-calib/`:
  - `coco2017_captions_small.txt`, `coco2017_captions_medium.txt`, `coco2017_captions_large.txt`
- Use a Pixi environment that includes:
  - `nvidia-modelopt`
  - `transformers` new enough to load `model_type: qwen3_vl` (>= 4.57)

## Run a single sensitivity job

Write to a temporary Hydra run directory:

```bash
pixi run -e rtx5090-vllm python scripts/qwen/qwen3_lm_sensitivity.py \
  output_layout=tmp \
  quant_pair=wfp8_afp8 \
  dataset.size=small
```

Publish artifacts under `models/qwen3_vl_4b_instruct/layer-analysis/`:

```bash
pixi run -e rtx5090-vllm python scripts/qwen/qwen3_lm_sensitivity.py \
  output_layout=publish \
  quant_pair=wfp8_afp16 \
  dataset.size=small
```

## Sweep multiple precision pairs and sizes

```bash
pixi run -e rtx5090-vllm python scripts/qwen/qwen3_lm_sensitivity.py -m \
  output_layout=publish \
  quant_pair=wfp4_afp8,wfp8_afp8,wfp8_afp16,wint8_afp16 \
  dataset.size=small,medium,large
```

## Available precision pairs

These are defined in `conf/quant_pair/` and can be selected via `quant_pair=<name>`:

- `wfp4_afp8`: FP4 weights (MXFP4) + FP8 activations (W4A8).
- `wfp4_afp16`: FP4 weight-only (input quantizer disabled).
- `wfp8_afp8`: FP8 weights + FP8 activations.
- `wfp8_afp16`: FP8 weight-only.
- `wint8_afp16`: INT8 weight-only.
- `wint8_afp8`: INT8 weights + FP8 activations (**experimental**).
- `wint8_aint8`: legacy/published parity pair for INT8 default LLM config.

## Report-only mode

To regenerate `per-layer-sensitivity.md` and `per-layer-sensitivity.json` from an existing manifest:

```bash
pixi run -e rtx5090-vllm python scripts/qwen/qwen3_lm_sensitivity.py \
  runner.report_only=true \
  output_layout=publish \
  quant_pair=wint8_aint8 \
  dataset.size=small
```

If the manifest lives in an ad-hoc directory, set `runner.output_dir=/path/to/run_dir` instead of `output_layout=...`.

## Related scripts

- INT8 all-layers (vision + LM) sensitivity is still driven by the legacy helper:
  - `models/qwen3_vl_4b_instruct/helpers/qwen3_vl_4b_autoquant_all_layers/run_qwen3_vl_4b_autoquant_all_layers.py`
- The combined “small/medium/large” INT8 runner publishes under `models/.../layer-analysis/` by default:
  - `scripts/qwen/run_qwen3_vl_4b_int8_sensitivity_3phase.sh`
