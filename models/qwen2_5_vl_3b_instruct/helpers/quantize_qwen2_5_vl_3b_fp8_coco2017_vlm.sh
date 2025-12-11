#!/usr/bin/env bash
set -euo pipefail

# Quantize Qwen2.5-VL-3B-Instruct to FP8 (weights/activations + KV cache)
# using NVIDIA ModelOpt, with calibration on COCO2017 image+caption pairs.
#
# This script is similar to quantize_qwen2_5_vl_3b_fp8_coco2017.sh, but
# uses a VLM calibration DB that references both images and captions:
#   - datasets/vlm-quantize-calib/coco2017_vlm_calib.db
#
# WARNING:
#   - Community / official FP8 recipes for Qwen2.5-VL (ModelOpt and
#     LLM-Compressor) leave the vision tower unquantized and only
#     quantize the language model; vLLM is aligned with that layout.
#   - The FP8 checkpoint produced by this script also quantizes the
#     vision stack and is experimental / **not** vLLM-compatible as
#     of vLLM 0.10.x. See:
#       models/qwen2_5_vl_3b_instruct/reports/fp8-vlm-vs-textonly-vllm-compat.md
#
# Usage (RTX 5090 env):
#   pixi run -e rtx5090 bash scripts/qwen/quantize_qwen2_5_vl_3b_fp8_coco2017_vlm.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CKPT_PATH="${REPO_ROOT}/models/qwen2_5_vl_3b_instruct/checkpoints/Qwen2.5-VL-3B-Instruct"
COCO_ROOT="${REPO_ROOT}/datasets/coco2017/source-data"
CALIB_DB="${REPO_ROOT}/datasets/vlm-quantize-calib/coco2017_vlm_calib.db"
EXPORT_PATH="${REPO_ROOT}/models/qwen2_5_vl_3b_instruct/quantized/fp8_fp8_coco2017_vlm"

MODELOPT_ROOT="${REPO_ROOT}/extern/TensorRT-Model-Optimizer"
DRIVER_SCRIPT="${REPO_ROOT}/scripts/qwen/quantize_qwen2_5_vl_3b_fp8_coco2017_vlm.py"

if [[ ! -d "${CKPT_PATH}" ]]; then
  echo "Error: Qwen2.5-VL-3B-Instruct checkpoint not found at:" >&2
  echo "  ${CKPT_PATH}" >&2
  echo "Hint: run models/qwen2_5_vl_3b_instruct/bootstrap.sh first." >&2
  exit 1
fi

if [[ ! -d "${COCO_ROOT}" ]]; then
  echo "Error: COCO2017 root not found at:" >&2
  echo "  ${COCO_ROOT}" >&2
  echo "Hint: ensure datasets/coco2017/source-data is populated." >&2
  exit 1
fi

if [[ ! -f "${CALIB_DB}" ]]; then
  echo "Error: VLM calibration DB not found at:" >&2
  echo "  ${CALIB_DB}" >&2
  echo "Hint: build it via:" >&2
  echo "  pixi run python scripts/build_vlm_quantize_calib_coco2017_db.py \\" >&2
  echo "    --coco-root datasets/coco2017/source-data \\" >&2
  echo "    --out datasets/vlm-quantize-calib/coco2017_vlm_calib.db \\" >&2
  echo "    --max-samples 4096" >&2
  exit 1
fi

echo "Quantizing Qwen2.5-VL-3B-Instruct to FP8 using COCO2017 image+caption data..."
echo "  Checkpoint    : ${CKPT_PATH}"
echo "  COCO root     : ${COCO_ROOT}"
echo "  Calib DB      : ${CALIB_DB}"
echo "  Export path   : ${EXPORT_PATH}"

PYTHONPATH="${MODELOPT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  python "${DRIVER_SCRIPT}" \
    --ckpt-dir "${CKPT_PATH}" \
    --coco-root "${COCO_ROOT}" \
    --calib-db "${CALIB_DB}" \
    --export-path "${EXPORT_PATH}" \
    --calib-size 4096 \
    --batch-size 1 \
    --device "cuda"

echo "Done. FP8-quantized VLM-calibrated Qwen2.5-VL-3B-Instruct exported to:"
echo "  ${EXPORT_PATH}"
