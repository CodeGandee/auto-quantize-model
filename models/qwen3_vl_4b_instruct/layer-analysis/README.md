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
  quant_granularity=default \
  dataset.size=medium
```

To change quantization granularity/dynamic behavior on top of the base ModelOpt
format selected by `quant_pair.format_name`, set `quant_granularity` (configs
live in `conf/quant_granularity/`):

```bash
pixi run -e rtx5090-vllm python scripts/qwen/qwen3_lm_sensitivity.py \
  output_layout=publish \
  quant_pair=wint4_aint8 \
  quant_granularity=recipe_match_channel_token \
  dataset.size=small
```

To sweep multiple dataset sizes:

```bash
pixi run -e rtx5090-vllm python scripts/qwen/qwen3_lm_sensitivity.py -m \
  output_layout=publish \
  quant_pair=wfp8_afp8 \
  quant_granularity=default \
  dataset.size=small,medium,large
```

To sweep multiple granularities (common for grid-searches):

```bash
pixi run -e rtx5090-vllm python scripts/qwen/qwen3_lm_sensitivity.py -m \
  output_layout=tmp \
  quant_pair=wint4_aint8 \
  quant_granularity=default,recipe_match_channel_token,w_group64,w_group128 \
  dataset.size=small
```

Published outputs land under:

- `models/qwen3_vl_4b_instruct/layer-analysis/weight-<weight>-act-<activation>/<quant_granularity.name>/...`

`conf/quant_granularity/*.yaml` uses this schema:

- `name: <string>` (used in output naming and manifests)
- `quant_cfg_overrides: {...}` (merged onto the base ModelOpt config’s `quant_cfg`)
