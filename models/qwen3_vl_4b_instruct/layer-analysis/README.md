# Qwen3-VL-4B Layer Analysis

This directory holds local analysis artifacts for Qwen3-VL-4B-Instruct (for
example, per-layer quantization sensitivity reports). These artifacts are
generated outputs intended to be committed for reproducibility.

Git ignores only `.pt` artifacts under `models/qwen3_vl_4b_instruct/layer-analysis/`
(for example, serialized AutoQuant state).

## Available analyses

- INT8 (W8A8) per-layer sensitivity: `weight-int8-act-int8/`
