# Qwen2.5-VL-3B Layer Analysis: INT8 (W8A8)

This directory contains per-layer sensitivity artifacts produced by NVIDIA
ModelOpt AutoQuant for Qwen2.5-VL-3B-Instruct with **weight INT8 + activation
INT8** (W8A8).

Only `.pt` artifacts under `models/qwen2_5_vl_3b_instruct/layer-analysis/` are
ignored by Git (for example, serialized AutoQuant state).

## Runs in this folder

- **All-layers (vision + text)**:
  - `qwen2_5_vl_3b_autoquant_all_layers_int8_small/`
  - `qwen2_5_vl_3b_autoquant_all_layers_int8_medium/`
  - `qwen2_5_vl_3b_autoquant_all_layers_int8_large/`
- **LM-only (text tower)**:
  - `qwen2_5_vl_3b_autoquant_int8_lm_small/`
  - `qwen2_5_vl_3b_autoquant_int8_lm_medium/`
  - `qwen2_5_vl_3b_autoquant_int8_lm_large/`

## What do `small`, `medium`, `large` mean?

Calibration subset sizes used during AutoQuant:

- `small`: 16 samples
- `medium`: 128 samples
- `large`: 512 samples

They correspond to the shared subsets in `datasets/vlm-quantize-calib/`:

- VLM DB: `coco2017_vlm_calib_{small,medium,large}.db`
- Captions: `coco2017_captions_{small,medium,large}.txt`

## Common artifacts inside each run directory

- `*_autoquant_state.pt`: serialized AutoQuant state (ignored by Git).
- `*_quant_manifest.json`: chosen formats + per-layer metadata.
- `per-layer-sensitivity.md`: human-readable per-layer report.
- `per-layer-sensitivity.json`: machine-readable per-layer report.

## What scripts produced these

These artifacts were generated via the 3-phase runner:

- `scripts/qwen/run_qwen2_5_vl_3b_int8_sensitivity_3phase.sh`

That runner calls the underlying drivers:

- LM-only INT8: `models/qwen2_5_vl_3b_instruct/helpers/qwen2_5_vl_3b_autoquant_fp8_schemes.py --scheme-name int8_autoquant_full`
- All-layers INT8: `models/qwen2_5_vl_3b_instruct/helpers/qwen2_5_vl_3b_autoquant_fp8_all_layers_per_scheme.py --quant-format int8`

Note: the all-layers driver also exports a quantized HF checkpoint; those large
artifacts are stored under `models/qwen2_5_vl_3b_instruct/quantized/` and are
ignored by Git.

