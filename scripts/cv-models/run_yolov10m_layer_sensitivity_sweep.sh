#!/usr/bin/env bash
set -euo pipefail

# Run a focused YOLOv10m layer sensitivity sweep (Torch / ModelOpt AutoQuant).
#
# This wraps:
#   tests/manual/yolo10_layer_sensitivity_sweep/scripts/run_layer_sensitivity_sweep.py
#
# Outputs (per run):
#   tmp/yolov10m_layer_sensitivity/<run-id>/{outputs,logs}/...
#
# Usage:
#   pixi run -e rtx5090 bash scripts/cv-models/run_yolov10m_layer_sensitivity_sweep.sh
#
# Customize:
#   DEVICE=cpu|cuda
#   WEIGHT_DTYPES="fp8 int8"
#   ACT_DTYPES="fp16 fp8"
#   GRANULARITIES="per_layer"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
RUN_ROOT="${REPO_ROOT}/tmp/yolov10m_layer_sensitivity/${RUN_ID}"
OUT_ROOT="${RUN_ROOT}/outputs"
LOG_ROOT="${RUN_ROOT}/logs"

DEVICE="${DEVICE:-cuda}"
WEIGHT_DTYPES="${WEIGHT_DTYPES:-fp8 int8}"
ACT_DTYPES="${ACT_DTYPES:-fp16 fp8}"
GRANULARITIES="${GRANULARITIES:-per_layer}"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

echo "[INFO] Bootstrapping YOLOv10 assets (if missing)..."
bash "${REPO_ROOT}/models/yolo10/bootstrap.sh"

echo "[INFO] Running YOLOv10m layer sensitivity sweep..."
echo "[INFO] run_root=${RUN_ROOT}"
echo "[INFO] device=${DEVICE}"
echo "[INFO] weight_dtypes=${WEIGHT_DTYPES}"
echo "[INFO] act_dtypes=${ACT_DTYPES}"
echo "[INFO] granularities=${GRANULARITIES}"

pixi run -e rtx5090 python "${REPO_ROOT}/tests/manual/yolo10_layer_sensitivity_sweep/scripts/run_layer_sensitivity_sweep.py" \
  --output-root "${OUT_ROOT}" \
  --log-root "${LOG_ROOT}" \
  --device "${DEVICE}" \
  --models yolov10m \
  --weight-dtypes ${WEIGHT_DTYPES} \
  --act-dtypes ${ACT_DTYPES} \
  --granularities ${GRANULARITIES} \
  --max-calib-samples 100 \
  --batch-size 1 \
  --imgsz 640 \
  --auto-quantize-score-size 16

echo "[INFO] Done. Outputs:"
echo "  ${OUT_ROOT}"
