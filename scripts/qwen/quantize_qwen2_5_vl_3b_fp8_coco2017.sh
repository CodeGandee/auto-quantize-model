#!/usr/bin/env bash
set -euo pipefail

# Quantize Qwen2.5-VL-3B-Instruct to FP8 (weights/activations) with NVIDIA ModelOpt,
# using a local COCO2017 captions subset for calibration.
#
# This wraps the ModelOpt HF PTQ example (`hf_ptq.py`) and:
#   - Loads the Qwen2.5-VL-3B-Instruct checkpoint from
#       models/qwen2_5_vl_3b_instruct/checkpoints/Qwen2.5-VL-3B-Instruct
#   - Uses the text-only calibration dataset
#       datasets/vlm-quantize-calib/coco2017_captions.txt
#     wired through ModelOpt as `coco2017_captions_local`
#   - Applies FP8 quantization (`qformat=fp8`) to the language model component
#     and FP8 quantization for the KV cache
#   - Exports a quantized HF checkpoint under:
#       models/qwen2_5_vl_3b_instruct/quantized/fp8_fp8_coco2017
#
# Usage (RTX 5090 env):
#   pixi run -e rtx5090 bash scripts/qwen/quantize_qwen2_5_vl_3b_fp8_coco2017.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CKPT_PATH="${REPO_ROOT}/models/qwen2_5_vl_3b_instruct/checkpoints/Qwen2.5-VL-3B-Instruct"
CALIB_CAPTIONS_TXT="${REPO_ROOT}/datasets/vlm-quantize-calib/coco2017_captions.txt"
EXPORT_PATH="${REPO_ROOT}/models/qwen2_5_vl_3b_instruct/quantized/fp8_fp8_coco2017"

HF_PTQ_SCRIPT="${REPO_ROOT}/extern/TensorRT-Model-Optimizer/examples/llm_ptq/hf_ptq.py"
MODELOPT_ROOT="${REPO_ROOT}/extern/TensorRT-Model-Optimizer"

if [[ ! -d "${CKPT_PATH}" ]]; then
  echo "Error: Qwen2.5-VL-3B-Instruct checkpoint not found at:" >&2
  echo "  ${CKPT_PATH}" >&2
  echo "Hint: bootstrap the model under models/qwen2_5_vl_3b_instruct/checkpoints/ first." >&2
  exit 1
fi

if [[ ! -f "${CALIB_CAPTIONS_TXT}" ]]; then
  echo "Error: COCO2017 captions calibration list not found at:" >&2
  echo "  ${CALIB_CAPTIONS_TXT}" >&2
  echo "Hint: build it via:" >&2
  echo "  pixi run python scripts/build_vlm_quantize_calib_coco2017_db.py \\" >&2
  echo "    --coco-root datasets/coco2017/source-data \\" >&2
  echo "    --out datasets/vlm-quantize-calib/coco2017_vlm_calib.db \\" >&2
  echo "    --max-samples 4096 \\" >&2
  echo "    --captions-text-out datasets/vlm-quantize-calib/coco2017_captions.txt" >&2
  exit 1
fi

echo "Quantizing Qwen2.5-VL-3B-Instruct to FP8 with ModelOpt HF PTQ..."
echo "  Checkpoint    : ${CKPT_PATH}"
echo "  Calib captions: ${CALIB_CAPTIONS_TXT}"
echo "  Export path   : ${EXPORT_PATH}"

PYTHONPATH="${MODELOPT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" python "${HF_PTQ_SCRIPT}" \
  --pyt_ckpt_path "${CKPT_PATH}" \
  --device "cuda" \
  --qformat "fp8" \
  --batch_size 0 \
  --calib_size "4096" \
  --calib_seq 512 \
  --export_path "${EXPORT_PATH}" \
  --dataset "coco2017_captions_local" \
  --kv_cache_qformat "fp8" \
  --sparsity_fmt "dense" \
  --auto_quantize_bits 0 \
  --gpu_max_mem_percentage 0.8 \
  --use_seq_device_map \
  --trust_remote_code

echo "Done. FP8-quantized Qwen2.5-VL-3B-Instruct exported to:"
echo "  ${EXPORT_PATH}"

