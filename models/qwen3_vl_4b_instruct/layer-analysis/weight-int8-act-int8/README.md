# Qwen3-VL-4B Layer Analysis: INT8 (W8A8)

This directory contains per-layer sensitivity artifacts produced by NVIDIA
ModelOpt AutoQuant for Qwen3-VL-4B-Instruct with **weight INT8 + activation INT8**
(W8A8).

Git ignores only `.pt` artifacts under `models/qwen3_vl_4b_instruct/layer-analysis/`
(for example, serialized AutoQuant state).

## Runs in this folder

- **All-layers (vision + text)**:
  - `qwen3_vl_4b_autoquant_all_layers_int8_small/`
  - `qwen3_vl_4b_autoquant_all_layers_int8_medium/`
  - `qwen3_vl_4b_autoquant_all_layers_int8_large/`
- **LM-only (text tower)**:
  - `qwen3_vl_4b_autoquant_int8_lm_small/`
  - `qwen3_vl_4b_autoquant_int8_lm_medium/`
  - `qwen3_vl_4b_autoquant_int8_lm_large/`

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

## How to regenerate

The simplest way to regenerate everything in this folder (all-layers + LM-only,
small/medium/large) is the 3-phase runner:

```bash
pixi run -e rtx5090-vllm bash scripts/qwen/run_qwen3_vl_4b_int8_sensitivity_3phase.sh
```

To write outputs under `tmp/` instead of publishing into this folder:

```bash
OUTPUT_MODE=tmp pixi run -e rtx5090-vllm bash scripts/qwen/run_qwen3_vl_4b_int8_sensitivity_3phase.sh
```

To regenerate only the **LM-only** INT8 runs via Hydra:

```bash
pixi run -e rtx5090-vllm python scripts/qwen/qwen3_lm_sensitivity.py -m \
  output_layout=publish \
  quant_pair=wint8_aint8 \
  dataset.size=small,medium,large
```

## What scripts produced these

These artifacts were generated via the 3-phase runner:

- `scripts/qwen/run_qwen3_vl_4b_int8_sensitivity_3phase.sh`

That runner calls the underlying drivers:

- All-layers INT8: `models/qwen3_vl_4b_instruct/helpers/qwen3_vl_4b_autoquant_all_layers/run_qwen3_vl_4b_autoquant_all_layers.py --quant-format int8`
- LM-only INT8 (Hydra): `scripts/qwen/qwen3_lm_sensitivity.py output_layout=publish quant_pair=wint8_aint8`
  - Legacy wrapper (still works): `models/qwen3_vl_4b_instruct/helpers/qwen3_vl_4b_autoquant_int8_lm/run_qwen3_vl_4b_autoquant_int8_lm.py`

From the repo root:

```bash
pixi run -e rtx5090-vllm bash scripts/qwen/run_qwen3_vl_4b_int8_sensitivity_3phase.sh
```
