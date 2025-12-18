# Qwen3-VL-4B-Instruct Model Assets

## HEADER
- **Purpose**: Track local snapshot for Qwen3-VL-4B-Instruct
- **Status**: Active
- **Date**: 2025-12-11
- **Dependencies**: Local Hugging Face snapshot storage
- **Target**: AI assistants and developers

## Content

This directory organizes a pointer to an external Qwen3-VL-4B-Instruct checkpoint downloaded from ModelScope or Hugging Face:

- `checkpoints/Qwen3-VL-4B-Instruct` — symlink to a local snapshot directory containing:
  - `config.json`, `generation_config.json`, tokenizer files
  - `model-*.safetensors`, `model.safetensors.index.json`
  - any other files required by the Qwen3-VL-4B-Instruct model

The symlink target is host-specific and should not be committed to the repository.

## Setup

On this development host, the Qwen3-VL-4B-Instruct snapshot currently lives under:

- `/workspace/llm-models/Qwen3-VL-4B-Instruct`

To mirror that layout (recommended pattern):

```bash
MODELS_ROOT=/workspace/llm-models
ln -s "${MODELS_ROOT}/Qwen3-VL-4B-Instruct" \
  models/qwen3_vl_4b_instruct/checkpoints/Qwen3-VL-4B-Instruct
```

Notes:

- Do not commit the `checkpoints/Qwen3-VL-4B-Instruct` symlink; it is environment-specific.
- Prefer downloading the model via ModelScope or Hugging Face CLI, or the respective web UI, into your chosen `${MODELS_ROOT}`.
- Keep any quantized or exported variants (e.g., ONNX, TensorRT) in dedicated experiment or export directories rather than inside this snapshot.

## Per-layer sensitivity (ModelOpt AutoQuant)

With the checkpoint symlink in place and COCO calibration subsets available, run:

```bash
# All-layers FP8 sensitivity (vision + text towers).
pixi run -e rtx5090-vllm python \
  models/qwen3_vl_4b_instruct/helpers/qwen3_vl_4b_autoquant_all_layers/run_qwen3_vl_4b_autoquant_all_layers.py

# All-layers INT8 (W8A8) sensitivity.
pixi run -e rtx5090-vllm python \
  models/qwen3_vl_4b_instruct/helpers/qwen3_vl_4b_autoquant_all_layers/run_qwen3_vl_4b_autoquant_all_layers.py \
  --quant-format int8 \
  --output-dir tmp/qwen3_vl_4b_autoquant_all_layers_int8_large

# INT8 LM-only sensitivity (text tower).
pixi run -e rtx5090-vllm python \
  models/qwen3_vl_4b_instruct/helpers/qwen3_vl_4b_autoquant_int8_lm/run_qwen3_vl_4b_autoquant_int8_lm.py
```

Outputs include `*_autoquant_state.pt`, `*_quant_manifest.json`, and
`layer-sensitivity-report.{md,json}` under the selected `tmp/` subdirectory.
