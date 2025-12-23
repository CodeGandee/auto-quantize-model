#!/usr/bin/env bash
set -euo pipefail

# Quantize YOLOv10m ONNX to INT4 Q/DQ (weight-only) with NVIDIA ModelOpt.
#
# This script supports a "mixed" scheme via `nodes_to_exclude` to keep selected
# Conv nodes in higher precision (fp16/fp32) while quantizing the rest to INT4.
#
# Usage:
#   RUN_ROOT="tmp/yolov10m_lowbit/$(date +%Y-%m-%d_%H-%M-%S)" \
#   CALIB_PATH="$RUN_ROOT/calib/calib_yolov10m_640.npy" \
#   OUTPUT_PATH="$RUN_ROOT/candidates/yolov10m-int4-k10.onnx" \
#   NODES_TO_EXCLUDE_FILE="$RUN_ROOT/schemes/nodes_to_exclude_k10.txt" \
#   pixi run -e rtx5090 bash scripts/cv-models/quantize_yolov10m_int4_onnx.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/tmp/yolov10m_lowbit/$(date +%Y-%m-%d_%H-%M-%S)}"
ONNX_PATH="${ONNX_PATH:-${REPO_ROOT}/models/cv-models/yolov10m/checkpoints/yolov10m.onnx}"
CALIB_PATH="${CALIB_PATH:-${RUN_ROOT}/calib/calib_yolov10m_640.npy}"
OUTPUT_PATH="${OUTPUT_PATH:-${RUN_ROOT}/onnx/yolov10m-int4-qdq.onnx}"
CALIBRATION_METHOD="${CALIBRATION_METHOD:-awq_clip}"
CALIBRATION_EPS="${CALIBRATION_EPS:-cuda:0 cpu}"
HIGH_PRECISION_DTYPE="${HIGH_PRECISION_DTYPE:-fp16}"
NODES_TO_EXCLUDE_FILE="${NODES_TO_EXCLUDE_FILE:-}"

LOG_DIR="${LOG_DIR:-${RUN_ROOT}/quantize-int4}"
LOG_FILE="${LOG_DIR}/modelopt-onnx-ptq.log"

mkdir -p "$(dirname "${OUTPUT_PATH}")" "${LOG_DIR}"

read -r -a CALIBRATION_EPS_ARR <<< "${CALIBRATION_EPS}"

if [[ ! -f "${ONNX_PATH}" ]]; then
  echo "Error: ONNX model not found at ${ONNX_PATH}" >&2
  exit 1
fi

if [[ ! -f "${CALIB_PATH}" ]]; then
  echo "Error: calibration tensor not found at ${CALIB_PATH}" >&2
  exit 1
fi

NODES_TO_EXCLUDE_ARR=()
if [[ -n "${NODES_TO_EXCLUDE_FILE}" ]]; then
  if [[ ! -f "${NODES_TO_EXCLUDE_FILE}" ]]; then
    echo "Error: NODES_TO_EXCLUDE_FILE not found: ${NODES_TO_EXCLUDE_FILE}" >&2
    exit 1
  fi
  while IFS= read -r line; do
    raw="$(echo "${line}" | sed 's/#.*$//g' | xargs || true)"
    if [[ -n "${raw}" ]]; then
      NODES_TO_EXCLUDE_ARR+=("${raw}")
    fi
  done < "${NODES_TO_EXCLUDE_FILE}"
fi

cat > "${LOG_DIR}/run-config.txt" <<EOF
RUN_ROOT=${RUN_ROOT}
ONNX_PATH=${ONNX_PATH}
CALIB_PATH=${CALIB_PATH}
OUTPUT_PATH=${OUTPUT_PATH}
CALIBRATION_METHOD=${CALIBRATION_METHOD}
CALIBRATION_EPS=${CALIBRATION_EPS}
HIGH_PRECISION_DTYPE=${HIGH_PRECISION_DTYPE}
NODES_TO_EXCLUDE_FILE=${NODES_TO_EXCLUDE_FILE}
NODES_TO_EXCLUDE_COUNT=${#NODES_TO_EXCLUDE_ARR[@]}
EOF

echo "Quantizing YOLOv10m ONNX with ModelOpt (INT4)..."
echo "  ONNX input  : ${ONNX_PATH}"
echo "  Calibration : ${CALIB_PATH}"
echo "  Output      : ${OUTPUT_PATH}"
echo "  Exclusions  : ${#NODES_TO_EXCLUDE_ARR[@]} nodes"
echo "  Logs        : ${LOG_FILE}"

CMD=(python -m modelopt.onnx.quantization
  --onnx_path "${ONNX_PATH}"
  --quantize_mode int4
  --calibration_data_path "${CALIB_PATH}"
  --calibration_method "${CALIBRATION_METHOD}"
  --calibration_eps "${CALIBRATION_EPS_ARR[@]}"
  --high_precision_dtype "${HIGH_PRECISION_DTYPE}"
  --output_path "${OUTPUT_PATH}"
)

if [[ ${#NODES_TO_EXCLUDE_ARR[@]} -gt 0 ]]; then
  CMD+=(--nodes_to_exclude "${NODES_TO_EXCLUDE_ARR[@]}")
fi

"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"

echo "Done. Quantized model written to:"
echo "  ${OUTPUT_PATH}"
