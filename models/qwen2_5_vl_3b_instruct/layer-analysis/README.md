# Qwen2.5-VL-3B Layer Analysis

This directory holds analysis artifacts for Qwen2.5-VL-3B-Instruct (for example,
per-layer quantization sensitivity reports).

Only `.pt` artifacts under `models/qwen2_5_vl_3b_instruct/layer-analysis/` are
ignored by Git (for example, serialized AutoQuant state).

## Available analyses

- INT8 (W8A8) per-layer sensitivity: `weight-int8-act-int8/`

