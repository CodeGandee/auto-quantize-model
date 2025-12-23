#!/usr/bin/env bash
set -euo pipefail

# Quantize YOLOv10m ONNX to INT8 Q/DQ with NVIDIA ModelOpt.
#
# This is a thin wrapper around:
#   python -m modelopt.onnx.quantization
#
# Defaults are chosen to match the "first-pass" ModelOpt recipe:
#   - quantize_mode=int8
#   - calibration_method=entropy
#   - use_zero_point=True (override via USE_ZERO_POINT)
#   - calibration_eps="cuda:0 cpu" (override via CALIBRATION_EPS)
#
# Usage:
#   RUN_ROOT="tmp/yolov10m_lowbit/$(date +%Y-%m-%d_%H-%M-%S)" \
#   CALIB_PATH="$RUN_ROOT/calib/calib_yolov10m_640.npy" \
#   pixi run -e rtx5090 bash scripts/cv-models/quantize_yolov10m_int8_onnx.sh
#
# Outputs:
#   - QDQ ONNX: $RUN_ROOT/onnx/yolov10m-int8-qdq.onnx
#   - Logs:     $RUN_ROOT/quantize-int8/modelopt-onnx-ptq.log

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/tmp/yolov10m_lowbit/$(date +%Y-%m-%d_%H-%M-%S)}"
ONNX_PATH="${ONNX_PATH:-${REPO_ROOT}/models/cv-models/yolov10m/checkpoints/yolov10m.onnx}"
CALIB_PATH="${CALIB_PATH:-${RUN_ROOT}/calib/calib_yolov10m_640.npy}"
OUTPUT_PATH="${OUTPUT_PATH:-${RUN_ROOT}/onnx/yolov10m-int8-qdq.onnx}"
CALIBRATION_METHOD="${CALIBRATION_METHOD:-entropy}"
CALIBRATION_EPS="${CALIBRATION_EPS:-cuda:0 cpu}"
HIGH_PRECISION_DTYPE="${HIGH_PRECISION_DTYPE:-fp16}"
NODES_TO_EXCLUDE_FILE="${NODES_TO_EXCLUDE_FILE:-}"
OP_TYPES_TO_EXCLUDE="${OP_TYPES_TO_EXCLUDE:-}"
OP_TYPES_TO_QUANTIZE="${OP_TYPES_TO_QUANTIZE:-}"
USE_ZERO_POINT="${USE_ZERO_POINT:-True}"

LOG_DIR="${LOG_DIR:-${RUN_ROOT}/quantize-int8}"
LOG_FILE="${LOG_DIR}/modelopt-onnx-ptq.log"

mkdir -p "${RUN_ROOT}/onnx" "${LOG_DIR}"

read -r -a CALIBRATION_EPS_ARR <<< "${CALIBRATION_EPS}"
read -r -a OP_TYPES_TO_EXCLUDE_ARR <<< "${OP_TYPES_TO_EXCLUDE}"
read -r -a OP_TYPES_TO_QUANTIZE_ARR <<< "${OP_TYPES_TO_QUANTIZE}"

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

if [[ ! -f "${ONNX_PATH}" ]]; then
  echo "Error: ONNX model not found at ${ONNX_PATH}" >&2
  exit 1
fi

if [[ ! -f "${CALIB_PATH}" ]]; then
  echo "Error: calibration tensor not found at ${CALIB_PATH}" >&2
  echo "Hint: generate it with scripts/cv-models/make_yolov10m_calib_npy.py" >&2
  exit 1
fi

cat > "${LOG_DIR}/run-config.txt" <<EOF
RUN_ROOT=${RUN_ROOT}
ONNX_PATH=${ONNX_PATH}
CALIB_PATH=${CALIB_PATH}
OUTPUT_PATH=${OUTPUT_PATH}
CALIBRATION_METHOD=${CALIBRATION_METHOD}
CALIBRATION_EPS=${CALIBRATION_EPS}
HIGH_PRECISION_DTYPE=${HIGH_PRECISION_DTYPE}
OP_TYPES_TO_EXCLUDE=${OP_TYPES_TO_EXCLUDE}
OP_TYPES_TO_QUANTIZE=${OP_TYPES_TO_QUANTIZE}
USE_ZERO_POINT=${USE_ZERO_POINT}
NODES_TO_EXCLUDE_FILE=${NODES_TO_EXCLUDE_FILE}
NODES_TO_EXCLUDE_COUNT=${#NODES_TO_EXCLUDE_ARR[@]}
EOF

echo "Quantizing YOLOv10m ONNX with ModelOpt (INT8)..."
echo "  ONNX input  : ${ONNX_PATH}"
echo "  Calibration : ${CALIB_PATH}"
echo "  Output      : ${OUTPUT_PATH}"
echo "  Logs        : ${LOG_FILE}"

CMD=(python -m modelopt.onnx.quantization
  --onnx_path "${ONNX_PATH}"
  --quantize_mode int8
  --calibration_data_path "${CALIB_PATH}"
  --calibration_method "${CALIBRATION_METHOD}"
  --calibration_eps "${CALIBRATION_EPS_ARR[@]}"
  --high_precision_dtype "${HIGH_PRECISION_DTYPE}"
  --output_path "${OUTPUT_PATH}"
)

if [[ ${#OP_TYPES_TO_EXCLUDE_ARR[@]} -gt 0 && -n "${OP_TYPES_TO_EXCLUDE_ARR[0]}" ]]; then
  CMD+=(--op_types_to_exclude "${OP_TYPES_TO_EXCLUDE_ARR[@]}")
fi

if [[ ${#OP_TYPES_TO_QUANTIZE_ARR[@]} -gt 0 && -n "${OP_TYPES_TO_QUANTIZE_ARR[0]}" ]]; then
  CMD+=(--op_types_to_quantize "${OP_TYPES_TO_QUANTIZE_ARR[@]}")
fi

if [[ ${#NODES_TO_EXCLUDE_ARR[@]} -gt 0 ]]; then
  CMD+=(--nodes_to_exclude "${NODES_TO_EXCLUDE_ARR[@]}")
fi

if [[ -n "${USE_ZERO_POINT}" ]]; then
  CMD+=(--use_zero_point "${USE_ZERO_POINT}")
fi

"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"

echo "Done. Quantized model written to:"
echo "  ${OUTPUT_PATH}"
