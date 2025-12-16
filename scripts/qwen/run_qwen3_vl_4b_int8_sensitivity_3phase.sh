#!/usr/bin/env bash
set -euo pipefail

# Run Qwen3-VL-4B INT8 per-layer sensitivity in 3 phases (small/medium/large).
#
# Usage (from repo root, RTX 5090 env):
#   pixi run -e rtx5090-vllm bash scripts/qwen/run_qwen3_vl_4b_int8_sensitivity_3phase.sh
#
# By default this script writes outputs into the published analysis layout under:
#   models/qwen3_vl_4b_instruct/layer-analysis/weight-int8-act-int8/
#
# To write to tmp/ instead:
#   OUTPUT_MODE=tmp pixi run -e rtx5090-vllm bash scripts/qwen/run_qwen3_vl_4b_int8_sensitivity_3phase.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ENV_NAME="${ENV_NAME:-rtx5090-vllm}"
OUTPUT_MODE="${OUTPUT_MODE:-publish}"

if [[ "${OUTPUT_MODE}" != "publish" && "${OUTPUT_MODE}" != "tmp" ]]; then
  echo "[ERROR] Unsupported OUTPUT_MODE: ${OUTPUT_MODE} (expected 'publish' or 'tmp')" >&2
  exit 1
fi

MODEL_DIR="${REPO_ROOT}/models/qwen3_vl_4b_instruct/checkpoints/Qwen3-VL-4B-Instruct"
CALIB_ROOT="${REPO_ROOT}/datasets/vlm-quantize-calib"

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "[ERROR] Qwen3-VL-4B-Instruct checkpoint not found at:" >&2
  echo "  ${MODEL_DIR}" >&2
  echo "Hint: run models/qwen3_vl_4b_instruct/bootstrap.sh first." >&2
  exit 1
fi

declare -A SIZE_TO_N=(
  [small]=16
  [medium]=128
  [large]=512
)

ALL_LAYERS_DRIVER="${REPO_ROOT}/models/qwen3_vl_4b_instruct/helpers/qwen3_vl_4b_autoquant_all_layers/run_qwen3_vl_4b_autoquant_all_layers.py"
LM_HYDRA_DRIVER="${REPO_ROOT}/scripts/qwen/qwen3_lm_sensitivity.py"

for size in small medium large; do
  n="${SIZE_TO_N[${size}]}"

  vlm_db="${CALIB_ROOT}/coco2017_vlm_calib_${size}.db"
  captions_txt="${CALIB_ROOT}/coco2017_captions_${size}.txt"

  if [[ ! -f "${vlm_db}" ]]; then
    echo "[ERROR] Missing VLM calib DB: ${vlm_db}" >&2
    exit 1
  fi
  if [[ ! -f "${captions_txt}" ]]; then
    echo "[ERROR] Missing captions subset: ${captions_txt}" >&2
    exit 1
  fi

  if [[ "${OUTPUT_MODE}" == "publish" ]]; then
    out_all="${REPO_ROOT}/models/qwen3_vl_4b_instruct/layer-analysis/weight-int8-act-int8/qwen3_vl_4b_autoquant_all_layers_int8_${size}"
    out_lm_mode="publish"
  else
    out_all="${REPO_ROOT}/tmp/qwen3_vl_4b_autoquant_all_layers_int8_${size}"
    out_lm="${REPO_ROOT}/tmp/qwen3_vl_4b_autoquant_int8_lm_${size}"
    out_lm_mode="tmp"
  fi

  echo
  echo "[INFO] Qwen3 INT8 all-layers sensitivity (${size}, ${n} samples)"
  pixi run -e "${ENV_NAME}" python "${ALL_LAYERS_DRIVER}" \
    --quant-format int8 \
    --model-dir "${MODEL_DIR}" \
    --output-dir "${out_all}" \
    --vlm-calib-db "${vlm_db}" \
    --max-calib-samples "${n}"

  echo
  echo "[INFO] Qwen3 INT8 LM-only sensitivity (${size}, ${n} samples)"
  if [[ "${out_lm_mode}" == "publish" ]]; then
    pixi run -e "${ENV_NAME}" python "${LM_HYDRA_DRIVER}" \
      output_layout=publish \
      quant_pair=wint8_aint8 \
      model.path="${MODEL_DIR}" \
      dataset.size="${size}" \
      dataset.captions_path="${captions_txt}" \
      dataset.max_calib_samples="${n}"
  else
    pixi run -e "${ENV_NAME}" python "${LM_HYDRA_DRIVER}" \
      output_layout=tmp \
      quant_pair=wint8_aint8 \
      runner.output_dir="${out_lm}" \
      model.path="${MODEL_DIR}" \
      dataset.size="${size}" \
      dataset.captions_path="${captions_txt}" \
      dataset.max_calib_samples="${n}"
  fi
done

echo
echo "[INFO] Completed Qwen3 INT8 3-phase sensitivity runs."
