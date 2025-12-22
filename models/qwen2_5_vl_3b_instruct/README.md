# Qwen2.5-VL-3B-Instruct Model Assets

## HEADER
- **Purpose**: Track local ModelScope snapshot for Qwen2.5-VL-3B-Instruct
- **Status**: Active
- **Date**: 2025-12-05
- **Dependencies**: Local Hugging Face snapshot storage
- **Target**: AI assistants and developers

## Content

This directory organizes a pointer to an external Qwen2.5-VL-3B-Instruct checkpoint downloaded from ModelScope:

- `checkpoints/Qwen2.5-VL-3B-Instruct` — symlink to a local HF snapshot directory containing:
  - `config.json`, `generation_config.json`, tokenizer files
  - `model-*.safetensors`, `model.safetensors.index.json`
  - any other files required by the Qwen2.5-VL-3B-Instruct model

The symlink target is host-specific and should not be committed to the repository.

## Setup

On this development host, the Qwen2.5-VL-3B-Instruct snapshot lives under:

- `/data2/llm-models/Qwen2.5-VL-3B-Instruct`

The upstream model can be found on ModelScope:

- https://modelscope.cn/models/Qwen/Qwen2.5-VL-3B-Instruct

To mirror that layout (recommended pattern):

```bash
MODELS_ROOT=/data2/llm-models
ln -s "${MODELS_ROOT}/Qwen2.5-VL-3B-Instruct" \
  models/qwen2_5_vl_3b_instruct/checkpoints/Qwen2.5-VL-3B-Instruct
```

Notes:

- Do not commit the `checkpoints/Qwen2.5-VL-3B-Instruct` symlink; it is environment-specific.
- Prefer downloading the model via ModelScope CLI (`pip install modelscope` + `modelscope download --model Qwen/Qwen2.5-VL-3B-Instruct --local_dir <path>`) or the ModelScope web UI.
- Keep any quantized or exported variants (e.g., ONNX, TensorRT) in dedicated experiment or export directories rather than inside this snapshot.

## Per-layer sensitivity (ModelOpt AutoQuant)

Once the checkpoint symlink is in place and COCO calibration subsets are available,
you can run per-layer quantization sensitivity with NVIDIA ModelOpt:

```bash
# INT8 (W8A8) LM-only sensitivity (text tower).
pixi run -e rtx5090-vllm python \
  models/qwen2_5_vl_3b_instruct/helpers/qwen2_5_vl_3b_autoquant_fp8_schemes.py \
  --scheme-name int8_autoquant_full \
  --output-dir tmp/qwen2_5_vl_3b_autoquant_int8_lm_large

# FP8 LM-only sensitivity.
pixi run -e rtx5090-vllm python \
  models/qwen2_5_vl_3b_instruct/helpers/qwen2_5_vl_3b_autoquant_fp8_schemes.py \
  --scheme-name fp8_autoquant_full \
  --output-dir tmp/qwen2_5_vl_3b_autoquant_fp8_lm_large
```

These drivers write `layer-sensitivity-report.md` and `layer-sensitivity-report.json`
under the chosen output directory. For all-layers (vision+text) schemes, see
`models/qwen2_5_vl_3b_instruct/helpers/qwen2_5_vl_3b_autoquant_fp8_all_layers_per_scheme.py`
and the plan in `context/plans/done/plan-int8-per-layer-sensitivity-qwen-vl.md`.
