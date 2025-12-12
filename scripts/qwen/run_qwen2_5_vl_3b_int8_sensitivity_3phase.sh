#!/usr/bin/env bash
set -euo pipefail

# Run Qwen2.5-VL-3B INT8 per-layer sensitivity in 3 phases (small/medium/large).
#
# This script always runs the LM-only INT8 sensitivity pass.
# For all-layers INT8 runs, you must provide a baseline and coverage manifest
# from an existing FP8 all-layers baseline:
#   BASELINE_DIR=... COVERAGE_MANIFEST=... pixi run -e rtx5090-vllm bash \
#     scripts/qwen/run_qwen2_5_vl_3b_int8_sensitivity_3phase.sh
#
# Outputs are written under tmp/ with one subdir per phase.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ENV_NAME="${ENV_NAME:-rtx5090-vllm}"

MODEL_DIR="${REPO_ROOT}/models/qwen2_5_vl_3b_instruct/checkpoints/Qwen2.5-VL-3B-Instruct"
CALIB_ROOT="${REPO_ROOT}/datasets/vlm-quantize-calib"

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "[ERROR] Qwen2.5-VL-3B-Instruct checkpoint not found at:" >&2
  echo "  ${MODEL_DIR}" >&2
  echo "Hint: run models/qwen2_5_vl_3b_instruct/bootstrap.sh first." >&2
  exit 1
fi

declare -A SIZE_TO_N=(
  [small]=16
  [medium]=128
  [large]=512
)

LM_DRIVER="${REPO_ROOT}/models/qwen2_5_vl_3b_instruct/helpers/qwen2_5_vl_3b_autoquant_fp8_schemes.py"
ALL_LAYERS_DRIVER="${REPO_ROOT}/models/qwen2_5_vl_3b_instruct/helpers/qwen2_5_vl_3b_autoquant_fp8_all_layers_per_scheme.py"

BASELINE_DIR="${BASELINE_DIR:-}"
COVERAGE_MANIFEST="${COVERAGE_MANIFEST:-}"

if [[ -n "${COVERAGE_MANIFEST}" && ! -f "${COVERAGE_MANIFEST}" ]]; then
  echo "[ERROR] COVERAGE_MANIFEST is set but file does not exist:" >&2
  echo "  ${COVERAGE_MANIFEST}" >&2
  exit 1
fi

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

  out_lm="${REPO_ROOT}/tmp/qwen2_5_vl_3b_autoquant_int8_lm_${size}"

  echo
  echo "[INFO] Qwen2.5 INT8 LM-only sensitivity (${size}, ${n} samples)"
  pixi run -e "${ENV_NAME}" python "${LM_DRIVER}" \
    --scheme-name int8_autoquant_full \
    --model-dir "${MODEL_DIR}" \
    --output-dir "${out_lm}" \
    --captions-path "${captions_txt}" \
    --max-calib-samples "${n}"

  if [[ -n "${BASELINE_DIR}" && -n "${COVERAGE_MANIFEST}" ]]; then
    out_all="${REPO_ROOT}/tmp/qwen2_5_vl_3b_autoquant_all_layers_int8_${size}"
    echo
    echo "[INFO] Qwen2.5 INT8 all-layers sensitivity (${size}, ${n} samples)"
    pixi run -e "${ENV_NAME}" python "${ALL_LAYERS_DRIVER}" \
      --quant-format int8 \
      --model-dir "${MODEL_DIR}" \
      --baseline-dir "${BASELINE_DIR}" \
      --coverage-manifest "${COVERAGE_MANIFEST}" \
      --out-dir "${out_all}" \
      --vlm-calib-db "${vlm_db}" \
      --max-calib-samples "${n}"
  else
    echo
    echo "[WARN] Skipping Qwen2.5 INT8 all-layers runs for ${size};"
    echo "       set BASELINE_DIR and COVERAGE_MANIFEST to enable."
  fi
done

echo
echo "[INFO] Completed Qwen2.5 INT8 3-phase sensitivity runs."

