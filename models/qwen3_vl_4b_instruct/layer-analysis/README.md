# Qwen3-VL-4B Layer Analysis

This directory holds local analysis artifacts for Qwen3-VL-4B-Instruct (for
example, per-layer quantization sensitivity reports). These artifacts are
generated outputs intended to be committed for reproducibility.

Git ignores only `.pt` artifacts under `models/qwen3_vl_4b_instruct/layer-analysis/`
(for example, serialized AutoQuant state).

## Available analyses

- INT8 (W8A8) per-layer sensitivity: `weight-int8-act-int8/`
- FP8 weights + FP16 activations (weight-only) LM-only sensitivity: `weight-fp8-act-fp16/`

## How to run new analyses (Hydra)

The preferred workflow for **LM-only** per-layer sensitivity is the Hydra runner:

```bash
pixi run -e rtx5090-vllm python scripts/qwen/qwen3_lm_sensitivity.py \
  output_layout=publish \
  quant_pair=wfp8_afp8 \
  dataset.size=medium
```

To sweep multiple dataset sizes:

```bash
pixi run -e rtx5090-vllm python scripts/qwen/qwen3_lm_sensitivity.py -m \
  output_layout=publish \
  quant_pair=wfp8_afp8 \
  dataset.size=small,medium,large
```

Published outputs land under:

- `models/qwen3_vl_4b_instruct/layer-analysis/weight-<weight>-act-<activation>/...`
