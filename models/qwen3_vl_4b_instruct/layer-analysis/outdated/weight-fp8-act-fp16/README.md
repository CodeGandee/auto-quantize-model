# Qwen3-VL-4B Layer Analysis: FP8 weights + FP16 activations

This directory contains per-layer sensitivity artifacts produced by NVIDIA
ModelOpt AutoQuant for Qwen3-VL-4B-Instruct with **FP8 weights + FP16
activations** (weight-only FP8 quantization; input quantizer disabled).

Git ignores only `.pt` artifacts under `models/qwen3_vl_4b_instruct/layer-analysis/`
(for example, serialized AutoQuant state).

## Runs in this folder

- **LM-only (text tower)**:
  - `qwen3_vl_4b_instruct_autoquant_wfp8_afp16_lm_small/`

## How to regenerate

Run a published sweep (small/medium/large) from the repo root:

```bash
pixi run -e rtx5090-vllm python scripts/qwen/qwen3_lm_sensitivity.py -m \
  output_layout=publish \
  quant_pair=wfp8_afp16 \
  dataset.size=small,medium,large
```
