# Qwen3-VL-8B-Instruct Model Assets

## HEADER
- **Purpose**: Track local snapshot for Qwen3-VL-8B-Instruct
- **Status**: Active
- **Date**: 2026-01-21
- **Dependencies**: Local Hugging Face snapshot storage
- **Target**: AI assistants and developers

## Content

This directory organizes a pointer to an external Qwen3-VL-8B-Instruct checkpoint downloaded from ModelScope or Hugging Face:

- `checkpoints/Qwen3-VL-8B-Instruct` — symlink to a local snapshot directory containing:
  - `config.json`, `generation_config.json`, tokenizer files
  - `model-*.safetensors`, `model.safetensors.index.json`
  - any other files required by the Qwen3-VL-8B-Instruct model

The symlink target is host-specific and should not be committed to the repository.

## Setup

On this development host, the Qwen3-VL-8B-Instruct snapshot currently lives under:

- `/data1/huangzhe/llm-models/Qwen3-VL-8B-Instruct`

To mirror that layout (recommended pattern):

```bash
MODELS_ROOT=/data1/huangzhe/llm-models
ln -s "${MODELS_ROOT}/Qwen3-VL-8B-Instruct" \
  models/qwen3_vl_8b_instruct/checkpoints/Qwen3-VL-8B-Instruct
```

Notes:

- Do not commit the `checkpoints/Qwen3-VL-8B-Instruct` symlink; it is environment-specific.
- Prefer downloading the model via ModelScope or Hugging Face CLI, or the respective web UI, into your chosen `${MODELS_ROOT}`.
- Keep any quantized or exported variants (e.g., ONNX, TensorRT) in dedicated experiment or export directories rather than inside this snapshot.

## Bootstrap helper

This repo provides an interactive bootstrap script that can create/update the symlink:

```bash
./models/qwen3_vl_8b_instruct/bootstrap.sh --yes
```

## Introduce + per-layer sensitivity (ModelOpt AutoQuant)

For a minimal, reproducible end-to-end smoke test that:

1. introduces the 8B checkpoint into the repo (via symlink), and
2. runs both all-layers INT8 + LM-only INT8 per-layer sensitivity,

see:

- `docs/tutorial/howto/tut-qwen3-vl-8b-introduce-model-layer-sensitivity/`
